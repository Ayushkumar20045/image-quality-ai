from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IssueResponse(BaseModel):
    type: str
    severity: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class AnalysisResponse(BaseModel):
    id: int | None = None

    image: str

    degradation: str

    severity: str

    quality_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    quality_label: str

    issues: list[IssueResponse]

    image_statistics: dict[str, Any]

    degradation_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    severity_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    degradation_probabilities: dict[str, float]

    severity_probabilities: dict[str, float]

    created_at: str | None = None


class HistoryResponse(BaseModel):
    total: int

    analyses: list[AnalysisResponse]