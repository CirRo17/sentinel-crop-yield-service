"""标准特征栈 schema 校验工具。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


BASE_FEATURE_NAMES = ("blue", "green", "red", "rededge", "nir", "swir", "ndvi", "ndwi", "evi", "ndre", "nbr")


@dataclass(frozen=True)
class FeatureSchemaCheck:
    model_features: list[str]
    stack_features: list[str]
    selected_band_indexes: list[int]
    selected_stack_features: list[str]
    missing_features: list[str]
    duplicate_stack_features: list[str]
    schema_hash: str
    matched_by_suffix: bool = False


def schema_hash(feature_names: list[str]) -> str:
    payload = json.dumps(feature_names, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def band_names_from_dataset(src: Any, metadata: dict[str, Any] | None = None) -> list[str]:
    if metadata and metadata.get("band_names"):
        names = [str(name) for name in metadata["band_names"]]
    else:
        names = [src.descriptions[index - 1] or f"band_{index}" for index in range(1, src.count + 1)]
    if len(names) != src.count:
        raise ValueError(f"特征名数量 {len(names)} 与栅格波段数 {src.count} 不一致。")
    return names


def duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


def base_feature_name(name: str) -> str:
    if name in BASE_FEATURE_NAMES:
        return name
    for suffix in BASE_FEATURE_NAMES:
        if name.endswith(f"_{suffix}"):
            return suffix
    return name


def feature_prefix(name: str) -> str | None:
    base = base_feature_name(name)
    if base == name:
        return None
    return name[: -(len(base) + 1)]


def check_feature_stack_schema(
    model_features: list[str],
    stack_features: list[str],
    *,
    allow_suffix_match: bool = False,
    suffix_prefix: str | None = None,
) -> FeatureSchemaCheck:
    model_features = [str(name) for name in model_features]
    stack_features = [str(name) for name in stack_features]
    duplicates = duplicate_names(stack_features)
    by_name = {name: index + 1 for index, name in enumerate(stack_features)}
    selected: list[int] = []
    selected_names: list[str] = []
    missing: list[str] = []
    matched_by_suffix = False

    for feature in model_features:
        if feature in by_name:
            selected.append(by_name[feature])
            selected_names.append(feature)
            continue

        if allow_suffix_match:
            if suffix_prefix:
                candidate = f"{suffix_prefix}_{feature}"
                if candidate in by_name:
                    selected.append(by_name[candidate])
                    selected_names.append(candidate)
                    matched_by_suffix = True
                    continue

            suffix_matches = [
                (name, index + 1)
                for index, name in enumerate(stack_features)
                if name.endswith(f"_{feature}")
            ]
            if len(suffix_matches) == 1:
                name, index = suffix_matches[0]
                selected.append(index)
                selected_names.append(name)
                matched_by_suffix = True
                continue

        missing.append(feature)

    return FeatureSchemaCheck(
        model_features=model_features,
        stack_features=stack_features,
        selected_band_indexes=selected,
        selected_stack_features=selected_names,
        missing_features=missing,
        duplicate_stack_features=duplicates,
        schema_hash=schema_hash(model_features),
        matched_by_suffix=matched_by_suffix,
    )


def require_feature_stack_schema(
    model_features: list[str],
    stack_features: list[str],
    *,
    allow_suffix_match: bool = False,
    suffix_prefix: str | None = None,
) -> FeatureSchemaCheck:
    check = check_feature_stack_schema(
        model_features,
        stack_features,
        allow_suffix_match=allow_suffix_match,
        suffix_prefix=suffix_prefix,
    )
    problems: list[str] = []
    if check.duplicate_stack_features:
        problems.append(f"特征栈存在重复 band 名：{', '.join(check.duplicate_stack_features)}")
    if check.missing_features:
        problems.append(f"特征栈缺少模型所需特征：{', '.join(check.missing_features)}")
    if problems:
        available = ", ".join(check.stack_features)
        required = ", ".join(check.model_features)
        raise ValueError(
            "标准特征栈 schema 校验失败。"
            f"{'；'.join(problems)}。"
            f"模型需要：{required}。"
            f"特征栈提供：{available}。"
        )
    return check
