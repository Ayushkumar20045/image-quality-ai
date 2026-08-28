from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.src.predict import predict_image


def predict(image_path: str | Path) -> dict[str, Any]:
    """
    Run the project's trained ML inference pipeline.

    The actual model loading, feature extraction, prediction,
    confidence calculation, issue detection, and quality scoring
    remain centralized in ml/src/predict.py.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    return dict(predict_image(path))
