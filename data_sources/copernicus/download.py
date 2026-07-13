"""从 Copernicus Data Space OData 下载 Sentinel-2 SAFE 产品。

读取 search.py 输出的 manifest，通过 OData 产品 API 下载完整 .SAFE zip，
解压到本地并在 manifest 中写入 `_local_safe_path`，供 extract.py 继续提取波段。
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from data_sources.copernicus.auth import get_access_token
from data_sources.copernicus.config import (
    COPERNICUS_ODATA_DOWNLOAD_URL,
    COPERNICUS_ODATA_URL,
    MAX_RETRIES,
)


def _odata_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _product_name_candidates(scene_id: str) -> list[str]:
    name = scene_id.strip()
    if not name:
        return []
    if name.endswith(".SAFE"):
        return [name, name[:-5]]
    return [name + ".SAFE", name]


def _query_odata_products(filter_expr: str, token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{COPERNICUS_ODATA_URL}/Products",
        params={"$filter": filter_expr, "$select": "Id,Name,Online", "$top": "5"},
        headers=_odata_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("value", [])


def _get_odata_product(scene_id: str, token: str) -> dict[str, Any]:
    """Find the OData product matching a STAC scene id/product name."""
    for candidate in _product_name_candidates(scene_id):
        escaped = candidate.replace("'", "''")
        products = _query_odata_products(f"Name eq '{escaped}'", token)
        if products:
            return products[0]

    escaped_scene = scene_id.replace("'", "''")
    products = _query_odata_products(f"startswith(Name,'{escaped_scene}')", token)
    if products:
        return products[0]

    raise FileNotFoundError(f"未在 Copernicus OData 中找到产品: {scene_id}")


def _download_product_zip(product_id: str, output_path: Path, token: str) -> None:
    """Download a complete SAFE product zip through the authenticated OData API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{COPERNICUS_ODATA_DOWNLOAD_URL}/Products({product_id})/$value"
    headers = _odata_headers(token)

    for attempt in range(MAX_RETRIES):
        try:
            current_url = url
            response = None
            for _ in range(8):
                response = requests.get(
                    current_url,
                    headers=headers,
                    stream=True,
                    timeout=180,
                    allow_redirects=False,
                )
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("Location")
                if not location:
                    break
                current_url = urljoin(current_url, location)

            if response is None:
                raise RuntimeError("下载响应为空")
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return
        except (requests.RequestException, IOError) as exc:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 10
                print(f"  重试 {attempt + 1}/{MAX_RETRIES}（{wait}s）...", flush=True)
                time.sleep(wait)
            else:
                raise


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract a SAFE zip and return the extracted .SAFE directory."""
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        safe_roots = {
            Path(name).parts[0]
            for name in archive.namelist()
            if Path(name).parts and Path(name).parts[0].endswith(".SAFE")
        }
        if not safe_roots:
            raise FileNotFoundError(f"{zip_path} 中没有找到 .SAFE 目录")
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if output_dir != target and output_dir not in target.parents:
                raise RuntimeError(f"ZIP 包含不安全路径: {member.filename}")
        archive.extractall(output_dir)

    for safe_root in sorted(safe_roots):
        safe_dir = output_dir / safe_root
        if safe_dir.exists():
            return safe_dir
    raise FileNotFoundError(f"解压后没有找到 .SAFE 目录: {zip_path}")


def _safe_dir_has_data(safe_dir: Path) -> bool:
    return safe_dir.exists() and any(safe_dir.rglob("*.jp2"))


def _zip_is_valid(zip_path: Path) -> bool:
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _download_scene_safe(
    scene_id: str,
    output_dir: Path,
    token: str,
    skip_existing: bool = True,
) -> Path:
    """Download one scene as a complete SAFE product through OData."""
    product = _get_odata_product(scene_id, token)
    product_name = str(product.get("Name") or scene_id)
    product_id = str(product.get("Id") or "")
    if not product_id:
        raise ValueError(f"OData 产品缺少 Id: {product}")

    safe_dir = output_dir / product_name
    if skip_existing and _safe_dir_has_data(safe_dir):
        return safe_dir

    zip_path = output_dir / f"{product_name}.zip"
    if not (skip_existing and _zip_is_valid(zip_path)):
        print(f"    下载 OData 产品 {product_name}", flush=True)
        _download_product_zip(product_id, zip_path, token)

    print(f"    解压 {zip_path.name}", flush=True)
    return _safe_extract_zip(zip_path, output_dir)


def download_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    token: str,
    skip_existing: bool = True,
    timepoints: list[str] | None = None,
) -> dict[str, Any]:
    """根据 manifest 下载所有场景的波段。"""
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)

    tp_labels = set(timepoints) if timepoints else None
    total = 0
    success = 0

    for tp in manifest.get("timepoints", []):
        if tp_labels and tp["label"] not in tp_labels:
            continue

        for scene in tp.get("scenes", []):
            total += 1
            scene_id = scene.get("id", "")
            print(f"\n[{tp['label']}] {scene_id}", flush=True)

            try:
                safe_dir = _download_scene_safe(
                    scene_id, output_dir, token, skip_existing
                )
                scene["_local_safe_path"] = str(safe_dir)
                scene["_downloaded"] = True
                scene.pop("_download_error", None)
                success += 1
                print(f"  完成: {safe_dir}", flush=True)
            except Exception as exc:
                print(f"  失败: {exc}", flush=True)
                scene["_download_error"] = str(exc)[:500]
                scene["_downloaded"] = False

    print(f"\n下载完成: {success}/{total} 场景", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Copernicus OData 下载 Sentinel-2 SAFE 产品。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/source/copernicus"))
    parser.add_argument("--timepoints", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest 不存在: {args.manifest}")

    print("获取 Copernicus access token...", flush=True)
    token = get_access_token()
    print("Token 已就绪。", flush=True)

    updated = download_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        token=token,
        skip_existing=not args.force,
        timepoints=args.timepoints,
    )

    output_path = args.output_manifest or args.manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"Manifest 已更新: {output_path}")


if __name__ == "__main__":
    main()
