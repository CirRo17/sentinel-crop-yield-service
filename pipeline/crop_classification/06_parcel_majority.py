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


DEFAULT_CLASSIFICATION = Path("data/output/crop_classification.tif")
DEFAULT_PARCELS = Path("shp_Files/Caobuhu_Parcel_shp/草埠湖镇修改.shp")
DEFAULT_OUTPUT_SHP = Path("data/output/parcel_postprocess/parcel_majority.shp")
DEFAULT_OUTPUT_ZIP = Path("data/output/parcel_postprocess/parcel_majority.zip")
DEFAULT_SUMMARY = Path("data/output/parcel_postprocess/parcel_majority_summary.json")
NODATA_CLASS = -9999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将分类栅格的地块内多数类别写入地块矢量。")
    parser.add_argument(
        "--classification",
        type=Path,
        default=DEFAULT_CLASSIFICATION,
        help="像素级分类 GeoTIFF，通常来自预测或后处理步骤。",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=DEFAULT_PARCELS,
        help="地块 Shapefile。API 场景由调用方上传，离线场景可手动指定。",
    )
    parser.add_argument(
        "--output-shp",
        type=Path,
        default=DEFAULT_OUTPUT_SHP,
        help="输出的地块级 Shapefile。",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help="输出 Shapefile 压缩包，便于接口下载或成果分发。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
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

    # 统一归一化到项目约定的输出类别，避免旧编码或异常编码混入结果。
    normalized = normalize_output_classes(values.astype("int64")).astype("int16")
    classes, counts = np.unique(normalized, return_counts=True)
    count_by_class = {int(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())}
    index = int(np.argmax(counts))
    return int(classes[index]), int(counts[index]), count_by_class


def _zip_shapefile(output_shp: Path, zip_path: Path) -> None:
    """打包 Shapefile 的全部伴随文件，便于 API 或人工下载。"""

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_shp.parent.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)


def attach_raster_majority_to_parcels(
    raster_path: Path,
    parcels_path: Path,
    output_shp: Path,
    zip_path: Path,
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

    gdf[field[:10]] = class_values
    gdf[label_field[:10]] = [TARGET_LABELS.get(value, "NoData") for value in class_values]
    gdf[count_field[:10]] = dominant_counts
    gdf[total_field[:10]] = total_counts
    if include_all:
        gdf[all_field[:10]] = all_counts

    gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
    _zip_shapefile(output_shp, zip_path)

    valid = [value for value in class_values if value != NODATA_CLASS]
    return {
        "parcel_shp": str(parcels_path),
        "output_shp": str(output_shp),
        "zip": str(zip_path),
        "parcel_count": int(len(gdf)),
        "valid_parcel_count": int(len(valid)),
        "class_codes": sorted({int(value) for value in valid}),
        "field": field[:10],
        "include_all": include_all,
    }


def main() -> None:
    args = parse_args()

    if not args.classification.exists():
        raise FileNotFoundError(
            f"Missing classification raster: {args.classification}. "
            "Run python -m pipeline.crop_classification.03_predict_classify or "
            "python -m pipeline.crop_classification.04_postprocess first."
        )
    if not args.parcels.exists():
        raise FileNotFoundError(f"Missing parcel shapefile: {args.parcels}.")

    # API 也调用同一个函数，避免维护两套地块统计逻辑。
    summary = attach_raster_majority_to_parcels(
        args.classification,
        args.parcels,
        args.output_shp,
        args.output_zip,
        field=args.field,
        raster_band=args.raster_band,
        include_all=args.include_all,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved parcel Shapefile: {args.output_shp}")
    print(f"Saved parcel Shapefile ZIP: {args.output_zip}")
    print(f"Saved parcel majority summary: {args.summary}")
    print(f"Valid parcels: {summary['valid_parcel_count']} / {summary['parcel_count']}")


if __name__ == "__main__":
    main()



