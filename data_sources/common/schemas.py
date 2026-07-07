"""数据源场景检索和兼容分类请求的数据结构。"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from data_sources.common.config import DEFAULT_LIMIT, DEFAULT_MAX_CLOUD


class SceneRequest(BaseModel):
    geometry: dict[str, Any] = Field(..., description="EPSG:4326 坐标系下的 GeoJSON geometry。")
    start_date: date
    end_date: date
    max_cloud: float = Field(DEFAULT_MAX_CLOUD, ge=0, le=100)
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=30)


class ClassifyRequest(SceneRequest):
    model_path: Optional[str] = Field(
        default=None,
        description="服务所在机器上的可选本地 joblib 模型路径。",
    )


class SceneSummary(BaseModel):
    id: str
    datetime: Optional[str] = None
    cloud_cover: Optional[float] = None
    assets: dict[str, str]


class ClassifyResponse(BaseModel):
    crop_type: str
    confidence: float
    method: str
    scene_count: int
    features: dict[str, float]
    scenes: list[SceneSummary]
