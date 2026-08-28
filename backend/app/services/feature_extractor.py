from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.src.feature_extraction import extract_features


def extract_image_features(
    image_path: str | Path,
) -> dict[str, float | int]:
    """
    Extract the same 13 features used by the trained ML models.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return extract_features(path)


def build_statistics(
    features: dict[str, float | int],
) -> dict[str, Any]:
    """
    Convert raw ML features into API-friendly image statistics.
    """
    return {
        "width": int(features["width"]),
        "height": int(features["height"]),
        "aspect_ratio": round(
            float(features["aspect_ratio"]), 4
        ),
        "sharpness": round(
            float(features["sharpness"]), 4
        ),
        "gradient_magnitude": round(
            float(features["gradient_magnitude"]), 4
        ),
        "mean_brightness": round(
            float(features["mean_brightness"]), 4
        ),
        "brightness_std": round(
            float(features["brightness_std"]), 4
        ),
        "dark_pixel_ratio": round(
            float(features["dark_pixel_ratio"]), 4
        ),
        "bright_pixel_ratio": round(
            float(features["bright_pixel_ratio"]), 4
        ),
        "high_frequency_residual": round(
            float(features["high_frequency_residual"]), 4
        ),
        "local_intensity_variation": round(
            float(features["local_intensity_variation"]), 4
        ),
        "mean_saturation": round(
            float(features["mean_saturation"]), 4
        ),
        "saturation_std": round(
            float(features["saturation_std"]), 4
        ),
    }
