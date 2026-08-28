from __future__ import annotations

from pathlib import Path
from typing import Any

from .prediction_service import predict


def analyze_image(
    image_path: str | Path,
    image_name: str | None = None,
) -> dict[str, Any]:
    """
    Execute the complete image-quality analysis pipeline.

    The ML implementation remains in ml/src/predict.py.
    This service provides the backend-facing abstraction.
    """
    result = predict(image_path)

    if image_name:
        result["image"] = image_name

    return result
