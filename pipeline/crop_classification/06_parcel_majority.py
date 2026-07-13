"""生成地块级作物类型统计结果。

读取像素级作物分类图和地块 Shapefile，对每个地块内部像元做多数投票，
把占比最高的作物类别写入地块属性，并输出新的 Shapefile、压缩包和统计
摘要 JSON。
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

from crop_domain.labels import TARGET_LABELS, normalize_output_classes
from configs.paths import ProjectPaths


NODATA_CLASS = 255
OUTPUT_NODATA_CLASS = 0
DEFAULT_CLASSIFICATION = Path("data/output/crop_classification/tuanlinpu_2026_06_classification_t49req.tif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将分类栅格的地块内多数类别写入地块矢量。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--classification",
        type=Path,
        default=DEFAULT_CLASSIFICATION,
        help="像素级分类 GeoTIFF，通常来自预测或后处理步骤。",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help="地块 Shapefile。不指定则从配置文件 project.parcels 推导。",
    )
    parser.add_argument(
        "--output-shp",
        type=Path,
        default=None,
        help="输出的地块级 Shapefile。",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="可选：输出 Shapefile 压缩包，便于接口下载或成果分发。不指定则不生成 ZIP。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="输出统计摘要 JSON。",
    )
    parser.add_argument(
        "--field",
        default="crop_type",
        help="写入地块属性表的类别字段名。Shapefile 最终会截断到 10 个字符。",
    )
    parser.add_argument(
        "--raster-band",
        type=int,
        default=1,
        help="分类 GeoTIFF 中用于统计的波段编号，1 表示第一波段。",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="额外写入每个地块内所有类别计数的 JSON 字段。",
    )
    return parser.parse_args()


def _dominant_value(values: np.ndarray) -> tuple[int, int, dict[int, int]]:
    """计算地块内的主类别、主类别像元数和所有类别计数。"""

    if values.size == 0:
        return NODATA_CLASS, 0, {}

    values = values[np.isfinite(values)]
    values = values[values != NODATA_CLASS]
    if values.size == 0:
        return NODATA_CLASS, 0, {}

    # 统一归一化到项目约定的输出类别，避免旧编码或异常编码混入结果。
    normalized = normalize_output_classes(values.astype("int64")).astype("int16")
    classes, counts = np.unique(normalized, return_counts=True)
    count_by_class = {int(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())}
    index = int(np.argmax(counts))
    return int(classes[index]), int(counts[index]), count_by_class


def _zip_shapefile(output_shp: Path, zip_path: Path) -> None:
    """打包 Shapefile 的全部伴随文件，便于 API 或人工下载。"""

    shapefile_suffixes = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_shp.parent.iterdir()):
            if path.is_file() and path.stem == output_shp.stem and path.suffix.lower() in shapefile_suffixes:
                archive.write(path, arcname=path.name)


def attach_raster_majority_to_parcels(
    raster_path: Path,
    parcels_path: Path,
    output_shp: Path,
    zip_path: Path | None,
    *,
    field: str = "crop_type",
    raster_band: int = 1,
    include_all: bool = False,
) -> dict[str, Any]:
    """把分类栅格的地块内主类别写入地块 Shapefile。"""

    if not raster_path.exists():
        raise FileNotFoundError(f"Missing classification raster: {raster_path}")
    if not parcels_path.exists():
        raise FileNotFoundError(f"Missing parcel shapefile: {parcels_path}")

    output_shp.parent.mkdir(parents=True, exist_ok=True)
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(parcels_path)
    if gdf.empty:
        raise ValueError(f"Parcel shapefile contains no features: {parcels_path}")

    with rasterio.open(raster_path) as src:
        if raster_band < 1 or raster_band > src.count:
            raise ValueError(f"Raster band {raster_band} is outside 1..{src.count}.")

        # 地块和分类图坐标系不一致时，先把地块投影到分类图坐标系。
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        class_values: list[int] = []
        dominant_counts: list[int] = []
        total_counts: list[int] = []
        all_counts: list[str] = []

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                class_values.append(NODATA_CLASS)
                dominant_counts.append(0)
                total_counts.append(0)
                all_counts.append("{}")
                continue

            try:
                # 按地块面裁剪分类栅格，只统计面内像元。
                data, _ = mask(
                    src,
                    [geom.__geo_interface__],
                    crop=True,
                    filled=False,
                    indexes=raster_band,
                )
            except ValueError:
                # 地块不与分类图相交时，标记为 NoData。
                class_values.append(NODATA_CLASS)
                dominant_counts.append(0)
                total_counts.append(0)
                all_counts.append("{}")
                continue

            values = data.compressed() if np.ma.isMaskedArray(data) else data.reshape(-1)
            dominant, dominant_count, counts = _dominant_value(values)
            class_values.append(dominant)
            dominant_counts.append(dominant_count)
            total_counts.append(int(values.size))
            all_counts.append(json.dumps(counts, ensure_ascii=False, sort_keys=True))

    # Shapefile 字段名最长 10 个字符，因此这里主动截断，避免驱动自动改名不可控。
    label_field = f"{field[:7]}_lbl"
    count_field = f"{field[:7]}_cnt"
    total_field = f"{field[:7]}_tot"
    all_field = f"{field[:6]}_all"

    output_class_values = [
        OUTPUT_NODATA_CLASS if value == NODATA_CLASS else value
        for value in class_values
    ]

    gdf[field[:10]] = output_class_values
    gdf[label_field[:10]] = [TARGET_LABELS.get(value, "NoData") for value in output_class_values]
    gdf[count_field[:10]] = dominant_counts
    gdf[total_field[:10]] = total_counts
    if include_all:
        gdf[all_field[:10]] = all_counts

    gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
    if zip_path is not None:
        _zip_shapefile(output_shp, zip_path)

    valid = [value for value, total in zip(output_class_values, total_counts) if total > 0]
    return {
        "parcel_shp": str(parcels_path),
        "output_shp": str(output_shp),
        "zip": str(zip_path) if zip_path is not None else None,
        "parcel_count": int(len(gdf)),
        "valid_parcel_count": int(len(valid)),
        "no_valid_pixel_parcel_count": int(sum(total == 0 for total in total_counts)),
        "class_codes": sorted({int(value) for value in valid}),
        "field": field[:10],
        "include_all": include_all,
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)

    classification = args.classification
    output_shp = args.output_shp or paths.parcel_majority_shp
    output_zip = args.output_zip
    summary = args.summary or paths.parcel_majority_summary
    parcels = args.parcels or paths.parcels

    if not classification.exists():
        raise FileNotFoundError(
            f"Missing classification raster: {classification}. "
            "Run python -m pipeline.crop_classification.03_predict_classify or "
            "python -m pipeline.crop_classification.04_postprocess first."
        )
    if not parcels or not parcels.exists():
        raise FileNotFoundError(
            f"Missing parcel shapefile. Pass --parcels explicitly."
        )

    result = attach_raster_majority_to_parcels(
        classification,
        parcels,
        output_shp,
        output_zip,
        field=args.field,
        raster_band=args.raster_band,
        include_all=args.include_all,
    )

    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved parcel Shapefile: {output_shp}")
    if output_zip is not None:
        print(f"Saved parcel Shapefile ZIP: {output_zip}")
    print(f"Saved parcel majority summary: {summary}")
    print(f"Valid parcels: {result['valid_parcel_count']} / {result['parcel_count']}")


if __name__ == "__main__":
    main()
