"""标准光谱指数计算。

所有指数共享同一个底层 safe_ratio 实现和 epsilon 常量，
确保 pipeline 离线训练、本地多波段适配、API 在线推理三处的
特征计算完全一致。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """归一化差值: (a - b) / (a + b + ε)."""
    return ((a - b) / (a + b + _EPS)).astype("float32")


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """归一化植被指数 NDVI。"""
    return safe_ratio(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """归一化水体指数 NDWI。"""
    return safe_ratio(green, nir)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """增强植被指数 EVI。"""
    return (
        2.5
        * (nir - red)
        / (nir + 6.0 * red - 7.5 * blue + 1.0 + _EPS)
    ).astype("float32")


def ndre(nir: np.ndarray, rededge: np.ndarray) -> np.ndarray:
    """归一化红边指数 NDRE。"""
    return safe_ratio(nir, rededge)


def ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """归一化水分指数 NDMI。"""
    return safe_ratio(nir, swir)


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """归一化燃烧指数 NBR。"""
    return safe_ratio(nir, swir)
