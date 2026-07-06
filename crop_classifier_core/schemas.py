from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import DEFAULT_LIMIT, DEFAULT_MAX_CLOUD


class SceneRequest(BaseModel):
    geometry: dict[str, Any] = Field(..., description="GeoJSON geometry in EPSG:4326.")
    start_date: date
    end_date: date
    max_cloud: float = Field(DEFAULT_MAX_CLOUD, ge=0, le=100)
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=30)


class ClassifyRequest(SceneRequest):
    model_path: Optional[str] = Field(
        default=None,
        description="Optional local joblib model path on the service host.",
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

