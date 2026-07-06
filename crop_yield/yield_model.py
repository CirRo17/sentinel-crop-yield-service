from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .config import CROP_MODELS, LAI_DEFAULTS


def _coefficient_values(override: Optional[Iterable[float]]) -> Optional[list[float]]:
    if override is not None:
        values = list(override)
        if not values:
            raise ValueError("model_coefficients cannot be empty for a custom yield function.")
        return [float(v) for v in values]
    return None


def coefficients_for(crop: str, override: Optional[Iterable[float]] = None):
    values = _coefficient_values(override)
    if values is not None:
        if len(values) != 3:
            raise ValueError("model_coefficients must contain [a, b, c] for the default polynomial model.")
        return values
    if crop not in CROP_MODELS:
        raise ValueError(f"Unsupported crop: {crop}")
    return CROP_MODELS[crop]["formula"]


def estimate_yield(
    index_values,
    crop: str,
    override: Optional[Iterable[float]] = None,
    function_type: str = "default",
):
    function_type = (function_type or "default").lower()
    if function_type in {"default", "best"}:
        a, b, c = coefficients_for(crop, override)
        return a * np.square(index_values) + b * index_values + c

    coefficients = _coefficient_values(override)
    if coefficients is None:
        raise ValueError("model_coefficients is required when yield_function is not default.")

    if function_type == "custom":
        function_type = "polynomial"

    if function_type == "linear":
        if len(coefficients) != 2:
            raise ValueError("linear yield function requires coefficients [a, b] for y=a*x+b.")
        a, b = coefficients
        return a * index_values + b

    if function_type == "exponential":
        if len(coefficients) not in {2, 3}:
            raise ValueError("exponential yield function requires [a, b] or [a, b, c] for y=a*exp(b*x)+c.")
        a, b = coefficients[:2]
        c = coefficients[2] if len(coefficients) == 3 else 0.0
        return a * np.exp(b * index_values) + c

    if function_type == "power":
        if len(coefficients) not in {2, 3}:
            raise ValueError("power yield function requires [a, b] or [a, b, c] for y=a*x^b+c.")
        a, b = coefficients[:2]
        c = coefficients[2] if len(coefficients) == 3 else 0.0
        safe_x = np.where(index_values > 0, index_values, np.nan)
        return a * np.power(safe_x, b) + c

    if function_type == "logarithmic":
        if len(coefficients) != 2:
            raise ValueError("logarithmic yield function requires coefficients [a, b] for y=a*ln(x)+b.")
        a, b = coefficients
        safe_x = np.where(index_values > 0, index_values, np.nan)
        return a * np.log(safe_x) + b

    if function_type == "polynomial":
        if len(coefficients) < 2:
            raise ValueError("polynomial yield function requires at least two coefficients.")
        return np.polyval(coefficients, index_values)

    raise ValueError(f"Unsupported yield_function: {function_type}")


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
