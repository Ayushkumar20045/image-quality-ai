from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


DEGRADATION_PENALTIES = {
    "blur": 1.00,
    "noise": 0.90,
    "compression": 0.80,
    "underexposure": 0.85,
    "overexposure": 0.85,
}


SEVERITY_PENALTIES = {
    "low": 15.0,
    "medium": 40.0,
    "high": 70.0,
}


def load_models() -> tuple[object, object]:
    required_files = [
        DEGRADATION_MODEL_PATH,
        SEVERITY_MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}"
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
) -> tuple[pd.DataFrame, dict[str, float | int]]:

    features = extract_features(
        image_path
    )

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

    dataframe = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS,
    )

    return dataframe, features


def get_prediction_confidence(
    model: object,
    features: pd.DataFrame,
) -> float:

    if not hasattr(
        model,
        "predict_proba",
    ):
        return 0.0

    probabilities = model.predict_proba(
        features
    )

    return float(
        np.max(
            probabilities[0]
        )
    )


def get_class_probabilities(
    model: object,
    features: pd.DataFrame,
) -> dict[str, float]:

    if not hasattr(
        model,
        "predict_proba",
    ):
        return {}

    probabilities = model.predict_proba(
        features
    )[0]

    classes = model.classes_

    return {
        str(label): float(probability)
        for label, probability in zip(
            classes,
            probabilities,
        )
    }


def calculate_quality_label(
    quality_score: float,
    severity: str,
) -> str:

    if severity == "high" or quality_score < 45:
        return "DEFECTIVE"

    if severity == "medium" or quality_score < 75:
        return "DEGRADED"

    return "ACCEPTABLE"


def create_issue(
    issue_type: str,
    severity: str,
    confidence: float,
) -> dict[str, Any]:

    return {
        "type": issue_type,
        "severity": severity,
        "confidence": round(
            float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            ),
            4,
        ),
    }


def build_issues(
    degradation: str,
    severity: str,
    degradation_confidence: float,
    features: dict[str, float | int],
    degradation_probabilities: dict[str, float],
) -> list[dict[str, Any]]:

    issues: list[dict[str, Any]] = []

    primary_confidence = (
        degradation_confidence
    )

    if degradation in DEGRADATION_PENALTIES:
        issues.append(
            create_issue(
                degradation,
                severity,
                primary_confidence,
            )
        )

    probability_threshold = 0.20

    secondary_candidates = [
        (
            label,
            probability,
        )
        for label, probability
        in degradation_probabilities.items()
        if (
            label != degradation
            and probability >= probability_threshold
        )
    ]

    for label, probability in sorted(
        secondary_candidates,
        key=lambda item: item[1],
        reverse=True,
    ):

        if len(issues) >= 3:
            break

        issue_severity = "low"

        if probability >= 0.70:
            issue_severity = "high"

        elif probability >= 0.40:
            issue_severity = "medium"

        issues.append(
            create_issue(
                label,
                issue_severity,
                probability,
            )
        )

    brightness = float(
        features["mean_brightness"]
    )

    dark_ratio = float(
        features["dark_pixel_ratio"]
    )

    bright_ratio = float(
        features["bright_pixel_ratio"]
    )

    sharpness = float(
        features["sharpness"]
    )

    residual = float(
        features["high_frequency_residual"]
    )

    if (
        brightness < 0.08
        and dark_ratio > 0.55
        and not any(
            issue["type"]
            == "underexposure"
            for issue in issues
        )
    ):
        confidence = min(
            0.99,
            0.60
            + dark_ratio
            * 0.35,
        )

        issues.append(
            create_issue(
                "underexposure",
                "high",
                confidence,
            )
        )

    elif (
        brightness < 0.18
        and dark_ratio > 0.35
        and not any(
            issue["type"]
            == "underexposure"
            for issue in issues
        )
    ):
        confidence = min(
            0.95,
            0.45
            + dark_ratio
            * 0.40,
        )

        issues.append(
            create_issue(
                "underexposure",
                "medium",
                confidence,
            )
        )

    if (
        brightness > 0.92
        and bright_ratio > 0.55
        and not any(
            issue["type"]
            == "overexposure"
            for issue in issues
        )
    ):
        confidence = min(
            0.99,
            0.60
            + bright_ratio
            * 0.35,
        )

        issues.append(
            create_issue(
                "overexposure",
                "high",
                confidence,
            )
        )

    elif (
        brightness > 0.82
        and bright_ratio > 0.35
        and not any(
            issue["type"]
            == "overexposure"
            for issue in issues
        )
    ):
        confidence = min(
            0.95,
            0.45
            + bright_ratio
            * 0.40,
        )

        issues.append(
            create_issue(
                "overexposure",
                "medium",
                confidence,
            )
        )

    if (
        sharpness < 20
        and not any(
            issue["type"] == "blur"
            for issue in issues
        )
    ):
        blur_confidence = min(
            0.98,
            0.65
            + (
                20
                - max(sharpness, 0)
            )
            / 40,
        )

        issues.append(
            create_issue(
                "blur",
                "high"
                if sharpness < 8
                else "medium",
                blur_confidence,
            )
        )

    if (
        residual > 0.10
        and not any(
            issue["type"] == "noise"
            for issue in issues
        )
    ):
        noise_confidence = min(
            0.95,
            0.50
            + residual,
        )

        issues.append(
            create_issue(
                "noise",
                "medium",
                noise_confidence,
            )
        )

    return issues[:5]


def calculate_quality_score(
    degradation: str,
    severity: str,
    degradation_confidence: float,
    severity_confidence: float,
    issues: list[dict[str, Any]],
) -> float:

    base_penalty = SEVERITY_PENALTIES.get(
        severity,
        40.0,
    )

    degradation_factor = (
        DEGRADATION_PENALTIES.get(
            degradation,
            0.85,
        )
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

    primary_penalty = (
        base_penalty
        * degradation_factor
        * confidence_factor
    )

    additional_penalty = 0.0

    if len(issues) > 1:
        additional_penalty = min(
            15.0,
            (
                len(issues) - 1
            )
            * 3.0,
        )

    score = (
        100.0
        - primary_penalty
        - additional_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def build_image_statistics(
    features: dict[str, float | int],
) -> dict[str, Any]:

    return {
        "width": int(
            features["width"]
        ),
        "height": int(
            features["height"]
        ),
        "aspect_ratio": round(
            float(
                features["aspect_ratio"]
            ),
            4,
        ),
        "sharpness": round(
            float(
                features["sharpness"]
            ),
            4,
        ),
        "gradient_magnitude": round(
            float(
                features[
                    "gradient_magnitude"
                ]
            ),
            4,
        ),
        "mean_brightness": round(
            float(
                features[
                    "mean_brightness"
                ]
            ),
            4,
        ),
        "brightness_std": round(
            float(
                features[
                    "brightness_std"
                ]
            ),
            4,
        ),
        "dark_pixel_ratio": round(
            float(
                features[
                    "dark_pixel_ratio"
                ]
            ),
            4,
        ),
        "bright_pixel_ratio": round(
            float(
                features[
                    "bright_pixel_ratio"
                ]
            ),
            4,
        ),
        "high_frequency_residual": round(
            float(
                features[
                    "high_frequency_residual"
                ]
            ),
            4,
        ),
        "local_intensity_variation": round(
            float(
                features[
                    "local_intensity_variation"
                ]
            ),
            4,
        ),
        "mean_saturation": round(
            float(
                features[
                    "mean_saturation"
                ]
            ),
            4,
        ),
        "saturation_std": round(
            float(
                features[
                    "saturation_std"
                ]
            ),
            4,
        ),
    }


def predict_image(
    image_path: str | Path,
) -> dict[str, object]:

    image_path = Path(
        image_path
    )

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

    (
        feature_dataframe,
        features,
    ) = build_feature_dataframe(
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

    degradation = str(
        degradation_prediction
    )

    severity = str(
        severity_prediction
    )

    issues = build_issues(
        degradation=degradation,
        severity=severity,
        degradation_confidence=(
            degradation_confidence
        ),
        features=features,
        degradation_probabilities=(
            degradation_probabilities
        ),
    )

    quality_score = calculate_quality_score(
        degradation=degradation,
        severity=severity,
        degradation_confidence=(
            degradation_confidence
        ),
        severity_confidence=(
            severity_confidence
        ),
        issues=issues,
    )

    quality_label = calculate_quality_label(
        quality_score=quality_score,
        severity=severity,
    )

    image_statistics = (
        build_image_statistics(
            features
        )
    )

    return {
        "image": image_path.name,

        "degradation": degradation,

        "severity": severity,

        "quality_score": round(
            quality_score,
            2,
        ),

        "quality_label": quality_label,

        "issues": issues,

        "image_statistics": image_statistics,

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

    print(
        "=" * 70
    )

    print(
        "IMAGE QUALITY AI PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Image: {result['image']}"
    )

    print(
        f"Quality: {result['quality_label']}"
    )

    print(
        f"Quality Score: "
        f"{result['quality_score']}/100"
    )

    print(
        f"Primary Degradation: "
        f"{result['degradation']}"
    )

    print(
        f"Severity: "
        f"{result['severity']}"
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

    print(
        "Detected Issues:"
    )

    issues = result[
        "issues"
    ]

    if issues:
        for issue in issues:
            print(
                f"  - {issue['type']:15s} "
                f"{issue['severity']:8s} "
                f"{issue['confidence']:.2%}"
            )
    else:
        print(
            "  None detected"
        )

    print()

    print(
        "Image Statistics:"
    )

    statistics = result[
        "image_statistics"
    ]

    for name, value in statistics.items():
        print(
            f"  {name:28s}: {value}"
        )

    print()

    print(
        "Degradation Probabilities:"
    )

    degradation_probabilities = (
        result[
            "degradation_probabilities"
        ]
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

    print(
        "Severity Probabilities:"
    )

    severity_probabilities = (
        result[
            "severity_probabilities"
        ]
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

    print(
        "=" * 70
    )

    print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Predict image quality using "
            "the trained Image Quality AI "
            "models."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help=(
            "Path to the image to analyze."
        ),
    )

    args = parser.parse_args()

    result = predict_image(
        args.image
    )

    print_prediction(
        result
    )


if __name__ == "__main__":
    main()