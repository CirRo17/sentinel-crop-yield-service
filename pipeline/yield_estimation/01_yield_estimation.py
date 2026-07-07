"""基于分类结果和多光谱特征估算作物产量。

核心估产逻辑内嵌在本文件中，使用分类栅格和特征栈中的植被指数
生成分作物产量栅格与统计 JSON。

示例：
    python -m pipeline.yield_estimation.01_yield_estimation \
        --classification data/output/crop_classification_clean.tif \
        --feature-stack data/exported/feature_stack.tif \
        --metadata data/exported/feature_stack_metadata.json \
        --timepoint t2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import rasterio

from crop_domain.labels import TARGET_LABELS

# ===================================================================
# 估产模型参数

CROP_MODELS = {
    "maize": {
        "label": "玉米",
        "code": 3,
        "formula": [77763.098231, -80886.711657, 24626.403283],
        "rmse_kg_ha": 450.0,
    },
    "wheat": {
        "label": "小麦",
        "code": 2,
        "formula": [92997.916893, -114110.350163, 38604.301220],
        "rmse_kg_ha": 520.0,
    },
    "rice": {
        "label": "水稻（中稻）",
        "code": 1,
        "formula": [-10128.830316, 13475.878556, 4032.546616],
        "rmse_kg_ha": 380.0,
    },
}

LAI_DEFAULTS = {
    "maize": {"k": 0.44, "m": 0.9},
    "wheat": {"k": 0.40, "m": 0.9},
    "rice": {"k": 0.42, "m": 0.9},
}

CROP_CODE_TO_NAME = {v["code"]: name for name, v in CROP_MODELS.items()}
SUPPORTED_CODES = {v["code"] for v in CROP_MODELS.values()}

# ===================================================================
# 估产核心函数
# ===================================================================


def _coefficient_values(override: Optional[Iterable[float]]) -> Optional[list[float]]:
    if override is not None:
        values = list(override)
        if not values:
            raise ValueError("自定义产量函数的 model_coefficients 不能为空。")
        return [float(v) for v in values]
    return None


def estimate_yield(
    index_values,
    crop: str,
    override: Optional[Iterable[float]] = None,
    function_type: str = "default",
):
    function_type = (function_type or "default").lower()
    if function_type in {"default", "best"}:
        if override is not None:
            values = list(override)
            if len(values) != 3:
                raise ValueError("默认多项式模型的 model_coefficients 必须包含 [a, b, c]。")
            a, b, c = [float(v) for v in values]
        else:
            if crop not in CROP_MODELS:
                raise ValueError(f"不支持的作物：{crop}")
            a, b, c = CROP_MODELS[crop]["formula"]
        return a * np.square(index_values) + b * index_values + c

    coefficients = _coefficient_values(override)
    if coefficients is None:
        raise ValueError("yield_function 不是 default 时必须提供 model_coefficients。")

    if function_type == "custom":
        function_type = "polynomial"

    if function_type == "linear":
        if len(coefficients) != 2:
            raise ValueError("linear 产量函数需要系数 [a, b]，形式为 y=a*x+b。")
        a, b = coefficients
        return a * index_values + b

    if function_type == "exponential":
        if len(coefficients) not in {2, 3}:
            raise ValueError("exponential 产量函数需要 [a, b] 或 [a, b, c]，形式为 y=a*exp(b*x)+c。")
        a, b = coefficients[:2]
        c = coefficients[2] if len(coefficients) == 3 else 0.0
        return a * np.exp(b * index_values) + c

    if function_type == "power":
        if len(coefficients) not in {2, 3}:
            raise ValueError("power 产量函数需要 [a, b] 或 [a, b, c]，形式为 y=a*x^b+c。")
        a, b = coefficients[:2]
        c = coefficients[2] if len(coefficients) == 3 else 0.0
        safe_x = np.where(index_values > 0, index_values, np.nan)
        return a * np.power(safe_x, b) + c

    if function_type == "logarithmic":
        if len(coefficients) != 2:
            raise ValueError("logarithmic 产量函数需要系数 [a, b]，形式为 y=a*ln(x)+b。")
        a, b = coefficients
        safe_x = np.where(index_values > 0, index_values, np.nan)
        return a * np.log(safe_x) + b

    if function_type == "polynomial":
        if len(coefficients) < 2:
            raise ValueError("polynomial 产量函数至少需要两个系数。")
        return np.polyval(coefficients, index_values)

    raise ValueError(f"不支持的 yield_function：{function_type}")


def lai_from_ci(ci_values, crop: str, k: Optional[float] = None, m: Optional[float] = None):
    params = LAI_DEFAULTS.get(crop, {"k": 0.44, "m": 0.9})
    kk = float(k if k is not None else params["k"])
    mm = float(m if m is not None else params["m"])
    clean_ci = np.maximum(ci_values, 0)
    return kk * np.power(clean_ci, mm)


def uncertainty(crop: str, mean_yield: float):
    rmse = float(CROP_MODELS[crop]["rmse_kg_ha"])
    relative = rmse / mean_yield if mean_yield else None
    return {
        "rmse_kg_ha": rmse,
        "relative_error": relative,
        "confidence_interval_95_kg_ha": [
            max(0.0, mean_yield - 1.96 * rmse),
            mean_yield + 1.96 * rmse,
        ],
    }


# ===================================================================
# 管线适配
# ===================================================================

DEFAULT_CLASSIFICATION = Path("data/output/crop_classification_clean.tif")
DEFAULT_FEATURE_STACK = Path("data/exported/feature_stack.tif")
DEFAULT_METADATA = Path("data/exported/feature_stack_metadata.json")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_STATS = Path("data/output/yield_stats.json")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于分类结果和植被指数估算作物产量。"
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=DEFAULT_CLASSIFICATION,
        help="分类或后处理输出的分类栅格。",
    )
    parser.add_argument(
        "--feature-stack",
        type=Path,
        default=DEFAULT_FEATURE_STACK,
        help="特征栈 GeoTIFF。",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="特征栈元数据 JSON。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="产量栅格输出目录。",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help="产量统计 JSON 输出路径。",
    )
    parser.add_argument(
        "--index",
        choices=("ndvi", "lai"),
        default="ndvi",
        help="产量估算所用的植被指数。ndvi 取所有时相的逐像素均值；lai 需要指定 --timepoint。",
    )
    parser.add_argument(
        "--timepoint",
        default=None,
        help="LAI 模式必须指定时相前缀，例如 t2；NDVI 模式忽略此参数。",
    )
    parser.add_argument(
        "--yield-function",
        default="default",
        help="产量函数类型：default, linear, exponential, power, logarithmic, polynomial。",
    )
    parser.add_argument(
        "--lai-k",
        type=float,
        default=None,
        help="LAI 模型参数 k，用于覆盖默认值。",
    )
    parser.add_argument(
        "--lai-m",
        type=float,
        default=None,
        help="LAI 模型参数 m，用于覆盖默认值。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 特征栈工具

def _load_band_names(metadata_path: Path) -> list[str]:
    """从元数据 JSON 读取波段名称列表。"""
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        names = meta.get("band_names", [])
        if names:
            return [str(n) for n in names]
    # 回退：直接从栅格读取
    with rasterio.open(DEFAULT_FEATURE_STACK) as src:
        return [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]


def _timepoint_prefixes(band_names: list[str]) -> list[str]:
    """从波段名列表提取所有唯一时相前缀，例如 t1、t2、2025_07。"""
    seen: dict[str, bool] = {}
    prefixes: list[str] = []
    for name in band_names:
        parts = name.split("_", 1)
        if len(parts) == 2 and parts[0] not in seen:
            seen[parts[0]] = True
            prefixes.append(parts[0])
    return prefixes


def _find_band_index(band_names: list[str], timepoint: str, suffix: str) -> Optional[int]:
    """查找指定时相和后缀对应的 1-based 波段索引。"""
    target = f"{timepoint}_{suffix}"
    try:
        return band_names.index(target) + 1
    except ValueError:
        return None


def _select_timepoint(
    band_names: list[str],
    classification: np.ndarray,
    feature_stack_path: Path,
    specified: Optional[str],
) -> str:
    """确定用于估产的时相前缀。"""
    prefixes = _timepoint_prefixes(band_names)
    if not prefixes:
        raise ValueError("特征栈中未找到任何时相前缀。")

    if specified is not None:
        if specified not in prefixes:
            raise ValueError(
                f"指定的时相 {specified!r} 不在特征栈中。可用时相：{', '.join(prefixes)}"
            )
        return specified

    if len(prefixes) == 1:
        return prefixes[0]

    # 自动选择作物区域 NDVI 均值最高的时相。
    crop_mask = np.isin(classification, list(SUPPORTED_CODES))
    if not np.any(crop_mask):
        return prefixes[-1]  # 没有作物像元时取最后一个时相
    best_prefix = prefixes[0]
    best_mean = -999.0
    with rasterio.open(feature_stack_path) as src:
        for prefix in prefixes:
            ndvi_idx = _find_band_index(band_names, prefix, "ndvi")
            if ndvi_idx is None:
                continue
            ndvi = src.read(ndvi_idx, masked=False).astype("float32")
            masked = ndvi[crop_mask]
            valid = masked[np.isfinite(masked)]
            if valid.size == 0:
                continue
            mean_ndvi = float(np.mean(valid))
            if mean_ndvi > best_mean:
                best_mean = mean_ndvi
                best_prefix = prefix

    print(f"自动选择时相：{best_prefix}（作物区 NDVI 均值 = {best_mean:.4f}）")
    return best_prefix


# ---------------------------------------------------------------------------
# 像素面积
# ---------------------------------------------------------------------------

def _pixel_area_ha(transform) -> float:
    area = abs(transform.a * transform.e)
    return area / 10000.0


# ---------------------------------------------------------------------------
# 主流程

def main() -> None:
    args = parse_args()

    for path, label in [
        (args.classification, "classification"),
        (args.feature_stack, "feature-stack"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 文件不存在：{path}")

    print(f"读取分类栅格：{args.classification}")
    with rasterio.open(args.classification) as src:
        classification = src.read(1, masked=False).astype("int16")
        class_profile = src.profile.copy()
        class_transform = src.transform

    pixel_area = _pixel_area_ha(class_transform)
    band_names = _load_band_names(args.metadata)
    print(f"特征栈共 {len(band_names)} 个波段")

    with rasterio.open(args.feature_stack) as fs_src:
        if args.index == "lai":
            if args.timepoint is None:
                raise ValueError("LAI 模式需要指定 --timepoint。")
            red_idx = _find_band_index(band_names, args.timepoint, "red")
            nir_idx = _find_band_index(band_names, args.timepoint, "nir")
            rededge_idx = _find_band_index(band_names, args.timepoint, "rededge")
            if any(i is None for i in (red_idx, nir_idx, rededge_idx)):
                raise ValueError(f"时相 {args.timepoint} 缺少 LAI 所需的 red/nir/rededge 波段。")
            rededge = fs_src.read(rededge_idx, masked=False).astype("float32")
            nir = fs_src.read(nir_idx, masked=False).astype("float32")
            ci = np.divide(nir, rededge, out=np.full_like(nir, np.nan), where=rededge > 0) - 1.0
            predictor = None
        else:
            prefixes = _timepoint_prefixes(band_names)
            ndvi_arrays = []
            for prefix in prefixes:
                ndvi_idx = _find_band_index(band_names, prefix, "ndvi")
                if ndvi_idx is not None:
                    ndvi_arrays.append(fs_src.read(ndvi_idx, masked=False).astype("float32"))
            if not ndvi_arrays:
                raise ValueError("特征栈中未找到任何 NDVI 波段。")
            predictor = np.nanmean(np.stack(ndvi_arrays, axis=0), axis=0).astype("float32")
            ci = None
            print(f"NDVI 均值使用 {len(ndvi_arrays)} 个时相")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yield_profile = class_profile.copy()
    yield_profile.update(dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)

    crop_results: list[dict[str, Any]] = []
    combined_yield = np.full((class_profile["height"], class_profile["width"]), np.nan, dtype="float32")

    for code in sorted(SUPPORTED_CODES):
        crop_name = CROP_CODE_TO_NAME[code]
        label = TARGET_LABELS.get(code, str(code))
        mask = classification == code
        pixel_count = int(np.count_nonzero(mask))

        if pixel_count == 0:
            print(f"  {label}: 无有效像元，跳过")
            crop_results.append({
                "crop_code": code,
                "crop_name": crop_name,
                "label": label,
                "area_ha": 0.0,
                "pixel_count": 0,
                "mean_yield_kg_ha": None,
                "total_yield_kg": 0.0,
                "warning": "no pixels found",
            })
            continue

        crop_predictor = lai_from_ci(ci, crop_name, args.lai_k, args.lai_m) if args.index == "lai" and ci is not None else predictor
        masked_predictor = np.where(mask, crop_predictor, np.nan)

        try:
            pixel_yield = estimate_yield(masked_predictor, crop_name, function_type=args.yield_function)
        except ValueError as exc:
            print(f"  {label}: 估产失败 - {exc}")
            crop_results.append({
                "crop_code": code,
                "crop_name": crop_name,
                "label": label,
                "area_ha": 0.0,
                "pixel_count": pixel_count,
                "error": str(exc),
            })
            continue

        valid_yield = np.isfinite(pixel_yield) & (pixel_yield > 0)
        if not np.any(valid_yield):
            print(f"  {label}: 无有效产量像元")
            crop_results.append({
                "crop_code": code,
                "crop_name": crop_name,
                "label": label,
                "area_ha": float(pixel_count * pixel_area),
                "pixel_count": pixel_count,
                "mean_yield_kg_ha": None,
                "total_yield_kg": 0.0,
                "warning": "no valid yield pixels",
            })
            continue

        yield_values = pixel_yield[valid_yield]
        area_ha = float(np.count_nonzero(valid_yield) * pixel_area)
        total_yield = float(np.sum(yield_values * pixel_area))
        mean_yield = float(np.mean(yield_values))
        median_yield = float(np.median(yield_values))
        std_yield = float(np.std(yield_values))

        crop_yield_raster = np.where(valid_yield, pixel_yield, yield_profile["nodata"]).astype("float32")
        out_path = args.output_dir / f"yield_{crop_name}.tif"
        with rasterio.open(out_path, "w", **yield_profile) as dst:
            dst.write(crop_yield_raster, 1)
            dst.set_band_description(1, f"yield_{crop_name}_kg_ha")

        combined_yield = np.where(valid_yield, pixel_yield, combined_yield)
        hist_counts, hist_edges = np.histogram(yield_values, bins=6)
        histogram = [
            {
                "min": float(hist_edges[i]),
                "max": float(hist_edges[i + 1]),
                "area_ha": float(count * pixel_area),
                "percent": float(count / len(yield_values) * 100.0),
            }
            for i, count in enumerate(hist_counts)
        ]

        crop_result = {
            "crop_code": code,
            "crop_name": crop_name,
            "label": label,
            "area_ha": area_ha,
            "pixel_count": int(np.count_nonzero(valid_yield)),
            "mean_yield_kg_ha": mean_yield,
            "median_yield_kg_ha": median_yield,
            "std_yield_kg_ha": std_yield,
            "total_yield_kg": total_yield,
            "histogram": histogram,
            "uncertainty": uncertainty(crop_name, mean_yield),
        }
        crop_results.append(crop_result)
        print(f"  {label}: 面积 {area_ha:.1f} ha, 均产 {mean_yield:.1f} kg/ha, 总产 {total_yield:.0f} kg")

    combined_out = args.output_dir / "yield_all.tif"
    combined_data = np.where(np.isfinite(combined_yield), combined_yield, yield_profile["nodata"]).astype("float32")
    with rasterio.open(combined_out, "w", **yield_profile) as dst:
        dst.write(combined_data, 1)
        dst.set_band_description(1, "yield_all_kg_ha")

    total_area = sum(r["area_ha"] for r in crop_results)
    total_yield_all = sum(r["total_yield_kg"] for r in crop_results)
    stats = {
        "timepoint_used": args.timepoint,
        "index_used": args.index,
        "yield_function": args.yield_function,
        "pixel_area_ha": pixel_area,
        "crops": crop_results,
        "summary": {
            "total_cropland_area_ha": total_area,
            "total_yield_kg": total_yield_all,
            "average_yield_kg_ha": total_yield_all / total_area if total_area > 0 else 0.0,
        },
        "outputs": {
            "combined": str(combined_out),
            "per_crop": {
                r["crop_name"]: str(args.output_dir / f"yield_{r['crop_name']}.tif")
                for r in crop_results
                if r.get("crop_name")
            },
        },
    }

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== 估产完成 ===")
    print(f"总种植面积：{total_area:.1f} ha")
    print(f"总产量：{total_yield_all:.0f} kg")
    print(f"统计输出：{args.stats}")
    print(f"产量栅格：{combined_out}")
if __name__ == "__main__":
    main()




