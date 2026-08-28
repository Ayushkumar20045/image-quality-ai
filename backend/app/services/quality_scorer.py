from __future__ import annotations

from typing import Any

from ml.src.predict import (
    calculate_quality_label,
    calculate_quality_score,
)


def score_prediction(
    degradation: str,
    severity: str,
    degradation_confidence: float,
    severity_confidence: float,
    issues: list[dict[str, Any]],
) -> float:
    """
    Calculate the final 0-100 image quality score using
    the project's existing scoring logic.
    """
    return calculate_quality_score(
        degradation=degradation,
        severity=severity,
        degradation_confidence=degradation_confidence,
        severity_confidence=severity_confidence,
        issues=issues,
    )


def classify_quality(
    quality_score: float,
    severity: str,
) -> str:
    """
    Convert the numerical score into the project's quality label.
    """
    return calculate_quality_label(
        quality_score=quality_score,
        severity=severity,
    )
