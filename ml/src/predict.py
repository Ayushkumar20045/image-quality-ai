from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "artifacts"

DEGRADATION_MODEL_PATH = (
    ARTIFACT_ROOT / "degradation_model.joblib"
)

SEVERITY_MODEL_PATH = (
    ARTIFACT_ROOT / "severity_model.joblib"
)

FEATURE_COLUMNS = [
    "width",
    "height",
    "aspect_ratio",
    "sharpness",
    "gradient_magnitude",
    "mean_brightness",
    "brightness_std",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "high_frequency_residual",
    "local_intensity_variation",
    "mean_saturation",
    "saturation_std",
]


def load_models() -> tuple[object, object]:
    required_files = [
        DEGRADATION_MODEL_PATH,
        SEVERITY_MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    degradation_model = joblib.load(
        DEGRADATION_MODEL_PATH
    )

    severity_model = joblib.load(
        SEVERITY_MODEL_PATH
    )

    return (
        degradation_model,
        severity_model,
    )


def build_feature_dataframe(
    image_path: Path,
) -> pd.DataFrame:
    features = extract_features(image_path)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing:
        raise RuntimeError(
            f"Missing required features: {missing}"
        )

    values = {
        feature: features[feature]
        for feature in FEATURE_COLUMNS
    }

    return pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS,
    )


def get_prediction_confidence(
    model: object,
    features: pd.DataFrame,
) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0

    probabilities = model.predict_proba(features)

    return float(np.max(probabilities[0]))


def get_class_probabilities(
    model: object,
    features: pd.DataFrame,
) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    return {
        str(label): float(probability)
        for label, probability in zip(
            classes,
            probabilities,
        )
    }


def calculate_quality_score(
    degradation: str,
    severity: str,
    degradation_confidence: float,
    severity_confidence: float,
) -> float:
    severity_penalty = {
        "low": 15.0,
        "medium": 40.0,
        "high": 70.0,
    }

    degradation_penalty = {
        "blur": 1.00,
        "noise": 0.90,
        "compression": 0.80,
        "underexposure": 0.85,
        "overexposure": 0.85,
    }

    base_penalty = severity_penalty.get(
        severity,
        40.0,
    )

    degradation_factor = degradation_penalty.get(
        degradation,
        0.85,
    )

    confidence_factor = (
        0.5
        + 0.5
        * (
            degradation_confidence
            + severity_confidence
        )
        / 2.0
    )

    penalty = (
        base_penalty
        * degradation_factor
        * confidence_factor
    )

    score = 100.0 - penalty

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def predict_image(
    image_path: str | Path,
) -> dict[str, object]:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    (
        degradation_model,
        severity_model,
    ) = load_models()

    feature_dataframe = build_feature_dataframe(
        image_path
    )

    degradation_prediction = (
        degradation_model.predict(
            feature_dataframe
        )[0]
    )

    severity_prediction = (
        severity_model.predict(
            feature_dataframe
        )[0]
    )

    degradation_confidence = (
        get_prediction_confidence(
            degradation_model,
            feature_dataframe,
        )
    )

    severity_confidence = (
        get_prediction_confidence(
            severity_model,
            feature_dataframe,
        )
    )

    degradation_probabilities = (
        get_class_probabilities(
            degradation_model,
            feature_dataframe,
        )
    )

    severity_probabilities = (
        get_class_probabilities(
            severity_model,
            feature_dataframe,
        )
    )

    quality_score = calculate_quality_score(
        degradation=str(
            degradation_prediction
        ),
        severity=str(
            severity_prediction
        ),
        degradation_confidence=(
            degradation_confidence
        ),
        severity_confidence=(
            severity_confidence
        ),
    )

    return {
        "image": image_path.name,
        "degradation": str(
            degradation_prediction
        ),
        "severity": str(
            severity_prediction
        ),
        "quality_score": round(
            quality_score,
            2,
        ),
        "degradation_confidence": round(
            degradation_confidence,
            4,
        ),
        "severity_confidence": round(
            severity_confidence,
            4,
        ),
        "degradation_probabilities": (
            degradation_probabilities
        ),
        "severity_probabilities": (
            severity_probabilities
        ),
    }


def print_prediction(
    result: dict[str, object],
) -> None:
    print()
    print("=" * 60)
    print("IMAGE QUALITY PREDICTION")
    print("=" * 60)

    print(
        f"Image: {result['image']}"
    )

    print(
        f"Degradation: {result['degradation']}"
    )

    print(
        f"Severity: {result['severity']}"
    )

    print(
        f"Quality Score: {result['quality_score']}/100"
    )

    print(
        "Degradation Confidence: "
        f"{result['degradation_confidence']:.2%}"
    )

    print(
        "Severity Confidence: "
        f"{result['severity_confidence']:.2%}"
    )

    print()
    print("Degradation Probabilities:")

    degradation_probabilities = (
        result["degradation_probabilities"]
    )

    for label, probability in sorted(
        degradation_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {label:15s} "
            f"{probability:.2%}"
        )

    print()
    print("Severity Probabilities:")

    severity_probabilities = (
        result["severity_probabilities"]
    )

    for label, probability in sorted(
        severity_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {label:15s} "
            f"{probability:.2%}"
        )

    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict image quality using the "
            "trained Image Quality AI models."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the image to analyze.",
    )

    args = parser.parse_args()

    result = predict_image(
        args.image
    )

    print_prediction(result)


if __name__ == "__main__":
    main()