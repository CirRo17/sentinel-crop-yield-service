"""汇总作物产量统计并生成报告。

本步骤读取估产输出的产量统计和可选的地块 Shapefile，对每个地块
统计各作物产量、面积和单产，并生成 JSON 与 HTML 汇总报告。

示例：
    python -m pipeline.yield_estimation.02_yield_summary \
        --yield-stats data/output/yield_estimation/yield_stats.json

    python -m pipeline.yield_estimation.02_yield_summary \
        --yield-raster data/output/yield_estimation/yield_all.tif \
        --parcels data/output/parcel_postprocess/parcel_majority.shp \
        --classification data/output/crop_classification/crop_classification_clean.tif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

import importlib

from crop_domain.labels import TARGET_LABELS
from configs.paths import ProjectPaths

_yield = importlib.import_module("pipeline.yield_estimation.01_yield_estimation")
CROP_CODE_TO_NAME = _yield.CROP_CODE_TO_NAME
CROP_MODELS = _yield.CROP_MODELS

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总产量统计，可选地块级聚合。"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--yield-stats",
        type=Path,
        default=None,
        help="估产输出的 yield_stats.json。",
    )
    parser.add_argument(
        "--yield-raster",
        type=Path,
        default=None,
        help="估产输出的综合产量栅格，地块模式需要。",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=None,
        help="分类栅格，地块模式需要。",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help="地块 Shapefile。指定后启用逐地块产量统计。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="汇总 JSON 输出路径。",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="HTML 报告输出路径。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 地块级聚合

def _parcel_yield_stats(
    yield_raster_path: Path,
    classification_path: Path,
    parcels_path: Path,
) -> list[dict[str, Any]]:
    """对每个地块提取各类作物产量统计。"""
    gdf = gpd.read_file(parcels_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    results: list[dict[str, Any]] = []

    with rasterio.open(yield_raster_path) as yield_src:
        with rasterio.open(classification_path) as class_src:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                geom_crs = gdf.crs
                geom_list = [geom.__geo_interface__]

                props: dict[str, Any] = {
                    "parcel_index": int(idx),
                }
                # 保留原始属性中的关键字段
                for col in gdf.columns:
                    if col != "geometry" and col in row.index:
                        val = row[col]
                        if isinstance(val, (str, int, float, bool, type(None))):
                            props[col] = val

                try:
                    class_data, class_transform = mask(
                        class_src, geom_list, crop=True, filled=True, nodata=0
                    )
                    class_arr = class_data[0].astype("int16")
                except Exception:
                    results.append({**props, "error": "分类栅格读取失败"})
                    continue

                try:
                    yield_data, _ = mask(
                        yield_src, geom_list, crop=True, filled=True,
                        nodata=yield_src.nodata
                    )
                    yield_arr = yield_data[0].astype("float32")
                except Exception:
                    results.append({**props, "error": "产量栅格读取失败"})
                    continue

                pixel_area = abs(class_transform.a * class_transform.e) / 10000.0

                crops_detail: dict[str, dict[str, Any]] = {}
                for code in sorted(CROP_CODE_TO_NAME):
                    crop_name = CROP_CODE_TO_NAME[code]
                    crop_mask = class_arr == code
                    if not np.any(crop_mask):
                        continue

                    crop_yield = np.where(
                        crop_mask & np.isfinite(yield_arr) & (yield_arr > 0),
                        yield_arr,
                        np.nan,
                    )
                    valid = np.isfinite(crop_yield)
                    if not np.any(valid):
                        continue

                    values = crop_yield[valid]
                    area_ha = float(np.count_nonzero(valid) * pixel_area)
                    crops_detail[crop_name] = {
                        "area_ha": area_ha,
                        "total_yield_kg": float(np.sum(values * pixel_area)),
                        "mean_yield_kg_ha": float(np.mean(values)),
                        "median_yield_kg_ha": float(np.median(values)),
                        "pixel_count": int(np.count_nonzero(valid)),
                    }

                if crops_detail:
                    results.append({**props, "crops": crops_detail})

    return results


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------

def _html_report(stats: dict, parcel_results: Optional[list[dict]]) -> str:
    """生成简单 HTML 产量报告。"""
    summary = stats.get("summary", {})
    crops = stats.get("crops", [])

    rows = ""
    for crop in crops:
        if crop.get("mean_yield_kg_ha") is None:
            continue
        rows += f"""<tr>
            <td>{crop['label']}</td>
            <td>{crop['area_ha']:.1f}</td>
            <td>{crop['mean_yield_kg_ha']:.1f}</td>
            <td>{crop['total_yield_kg']:.0f}</td>
            <td>{crop.get('uncertainty', {}).get('rmse_kg_ha', '-')}</td>
        </tr>"""

    parcel_html = ""
    if parcel_results:
        parcel_items = ""
        for p in parcel_results:
            crops_detail = p.get("crops", {})
            crop_lines = ""
            for cname, cd in crops_detail.items():
                crop_lines += (
                    f"<li>{cname}: {cd['area_ha']:.1f} ha, "
                    f"{cd['mean_yield_kg_ha']:.1f} kg/ha, "
                    f"{cd['total_yield_kg']:.0f} kg</li>"
                )
            parcel_items += (
                f"<tr><td>{p.get('parcel_index', '?')}</td>"
                f"<td><ul>{crop_lines}</ul></td></tr>"
            )
        parcel_html = f"""
        <h2>地块级产量</h2>
        <table border='1' cellpadding='6'>
            <tr><th>地块</th><th>产量明细</th></tr>
            {parcel_items}
        </table>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>产量报告</title>
<style>
    body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    th, td {{ padding: 8px 12px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    h1, h2 {{ color: #333; }}
    .summary {{ background: #e8f5e9; padding: 1em; border-radius: 8px; margin: 1em 0; }}
</style></head>
<body>
<h1>作物估产报告</h1>
<div class="summary">
    <strong>总种植面积:</strong> {summary.get('total_cropland_area_ha', 0):.1f} ha<br>
    <strong>总产量:</strong> {summary.get('total_yield_kg', 0):.0f} kg<br>
    <strong>平均单产:</strong> {summary.get('average_yield_kg_ha', 0):.1f} kg/ha<br>
    <strong>使用指数:</strong> {stats.get('index_used', '-')}<br>
    <strong>使用函数:</strong> {stats.get('yield_function', '-')}
</div>
<h2>分作物统计</h2>
<table border='1' cellpadding='6'>
    <tr><th>作物</th><th>面积 (ha)</th><th>均产 (kg/ha)</th><th>总产 (kg)</th><th>RMSE (kg/ha)</th></tr>
    {rows}
</table>
{parcel_html}
</body></html>"""


# ---------------------------------------------------------------------------
# 主流程

def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)

    yield_stats = args.yield_stats or paths.yield_stats
    yield_raster = yield_raster or paths.yield_raster
    classification = classification or paths.classification_clean
    output = output or paths.yield_summary
    html = html or paths.yield_report_html
    parcels = parcels

    if not yield_stats.exists():
        raise FileNotFoundError(f"yield_stats 文件不存在：{yield_stats}")
    with open(yield_stats, encoding="utf-8") as f:
        stats = json.load(f)

    parcel_results = None
    if parcels and parcels.exists():
        print(f"逐地块聚合产量：{parcels}")
        if not yield_raster.exists():
            print(f"  警告：产量栅格不存在（{yield_raster}），跳过地块聚合")
        elif not classification.exists():
            print(f"  警告：分类栅格不存在（{classification}），跳过地块聚合")
        else:
            parcel_results = _parcel_yield_stats(yield_raster, classification, parcels)
            print(f"  完成 {len(parcel_results)} 个地块的产量统计")

    output_doc = {"yield_stats": stats}
    if parcel_results is not None:
        output_doc["parcel_yields"] = parcel_results
        total_from_parcels = sum(
            sum(cd.get("total_yield_kg", 0) for cd in p.get("crops", {}).values())
            for p in parcel_results
        )
        output_doc["parcel_summary"] = {
            "parcel_count": len(parcel_results),
            "total_yield_from_parcels_kg": total_from_parcels,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2, ensure_ascii=False)
    print(f"汇总 JSON：{output}")

    html_content = _html_report(stats, parcel_results)
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(html_content, encoding="utf-8")
    print(f"HTML 报告：{html}")
if __name__ == "__main__":
    main()




