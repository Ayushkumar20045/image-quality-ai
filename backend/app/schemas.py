from __future__ import annotations

from pydantic import BaseModel, Field


class ProbabilityResponse(BaseModel):
    label: str
    probability: float = Field(
        ge=0.0,
        le=1.0,
    )


class AnalysisResponse(BaseModel):
    image: str
    degradation: str
    severity: str
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
    )
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