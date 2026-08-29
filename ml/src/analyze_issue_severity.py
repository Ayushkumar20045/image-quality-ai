from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "features"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "evaluation"
)

TEST_FILE = FEATURE_ROOT / "test.csv"

SEVERITY_MODEL_PATH = (
    ARTIFACT_ROOT / "severity_model.joblib"
)

IDENTIFIER_COLUMNS = {
    "source_id",
    "split",
    "filename",
    "degradation",
    "severity",
}

ISSUES = [
    "blur",
    "noise",
    "compression",
    "underexposure",
    "overexposure",
]

SEVERITIES = [
    "low",
    "medium",
    "high",
]


ISSUE_FEATURES = {
    "blur": [
        "sharpness",
        "gradient_magnitude",
    ],
    "noise": [
        "high_frequency_residual",
        "local_intensity_variation",
    ],
    "compression": [
        "local_intensity_variation",
        "high_frequency_residual",
        "gradient_magnitude",
    ],
    "underexposure": [
        "mean_brightness",
        "dark_pixel_ratio",
        "brightness_std",
    ],
    "overexposure": [
        "mean_brightness",
        "bright_pixel_ratio",
        "brightness_std",
    ],
}


def load_test_data() -> pd.DataFrame:

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}"
        )

    data = pd.read_csv(TEST_FILE)

    if data.empty:
        raise RuntimeError(
            "Test dataset is empty."
        )

    return data


def get_feature_columns(
    data: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in data.columns
        if column not in IDENTIFIER_COLUMNS
    ]


def load_model():

    if not SEVERITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Severity model not found: "
            f"{SEVERITY_MODEL_PATH}"
        )

    return joblib.load(
        SEVERITY_MODEL_PATH
    )


def predict(
    data: pd.DataFrame,
    feature_columns: list[str],
    model,
) -> pd.DataFrame:

    x = data[feature_columns]

    predictions = model.predict(x)

    probabilities = model.predict_proba(x)

    confidence = np.max(
        probabilities,
        axis=1,
    )

    result = data.copy()

    result[
        "predicted_severity"
    ] = predictions

    result[
        "severity_confidence"
    ] = confidence

    result[
        "severity_correct"
    ] = (
        result["severity"]
        == result["predicted_severity"]
    )

    return result


def analyze_actual_severity(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in ISSUES:

        subset = data[
            data["degradation"] == issue
        ]

        if subset.empty:
            continue

        for severity in SEVERITIES:

            severity_subset = subset[
                subset["severity"] == severity
            ]

            if severity_subset.empty:
                continue

            for feature in ISSUE_FEATURES[
                issue
            ]:

                rows.append(
                    {
                        "issue": issue,
                        "severity": severity,
                        "feature": feature,
                        "samples": len(
                            severity_subset
                        ),
                        "mean": (
                            severity_subset[
                                feature
                            ].mean()
                        ),
                        "median": (
                            severity_subset[
                                feature
                            ].median()
                        ),
                        "std": (
                            severity_subset[
                                feature
                            ].std()
                        ),
                        "min": (
                            severity_subset[
                                feature
                            ].min()
                        ),
                        "max": (
                            severity_subset[
                                feature
                            ].max()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def analyze_predicted_severity(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in ISSUES:

        subset = data[
            data["predicted_degradation"]
            == issue
        ] if "predicted_degradation" in data else data[
            data["degradation"] == issue
        ]

        if subset.empty:
            continue

        for severity in SEVERITIES:

            severity_subset = subset[
                subset["predicted_severity"]
                == severity
            ]

            if severity_subset.empty:
                continue

            for feature in ISSUE_FEATURES[
                issue
            ]:

                rows.append(
                    {
                        "issue": issue,
                        "predicted_severity": severity,
                        "samples": len(
                            severity_subset
                        ),
                        "mean": (
                            severity_subset[
                                feature
                            ].mean()
                        ),
                        "median": (
                            severity_subset[
                                feature
                            ].median()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def calculate_severity_ordering(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    severity_rank = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    for issue in ISSUES:

        subset = data[
            data["degradation"] == issue
        ]

        if subset.empty:
            continue

        for feature in ISSUE_FEATURES[
            issue
        ]:

            means = (
                subset
                .groupby("severity")[
                    feature
                ]
                .mean()
            )

            if not all(
                severity in means.index
                for severity in SEVERITIES
            ):
                continue

            values = [
                means[severity]
                for severity in SEVERITIES
            ]

            increasing = (
                values[0]
                <= values[1]
                <= values[2]
            )

            decreasing = (
                values[0]
                >= values[1]
                >= values[2]
            )

            ordered = (
                increasing
                or decreasing
            )

            correlation = (
                subset["severity"]
                .map(severity_rank)
                .corr(
                    subset[feature]
                )
            )

            rows.append(
                {
                    "issue": issue,
                    "feature": feature,
                    "low_mean": means["low"],
                    "medium_mean": means["medium"],
                    "high_mean": means["high"],
                    "ordered": ordered,
                    "direction": (
                        "increasing"
                        if increasing
                        else (
                            "decreasing"
                            if decreasing
                            else "mixed"
                        )
                    ),
                    "severity_correlation": correlation,
                }
            )

    return pd.DataFrame(rows)


def analyze_prediction_accuracy(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in ISSUES:

        subset = data[
            data["degradation"] == issue
        ]

        if subset.empty:
            continue

        for severity in SEVERITIES:

            severity_subset = subset[
                subset["severity"] == severity
            ]

            if severity_subset.empty:
                continue

            accuracy = (
                severity_subset[
                    "severity_correct"
                ].mean()
                * 100
            )

            confidence = (
                severity_subset[
                    "severity_confidence"
                ].mean()
                * 100
            )

            rows.append(
                {
                    "issue": issue,
                    "actual_severity": severity,
                    "samples": len(
                        severity_subset
                    ),
                    "accuracy": round(
                        accuracy,
                        2,
                    ),
                    "mean_confidence": round(
                        confidence,
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_summary(
    actual: pd.DataFrame,
    ordering: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print(
        "PHASE 3.3 - ISSUE SEVERITY VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "ACTUAL SEVERITY FEATURE ANALYSIS"
    )
    print("-" * 70)

    for issue in ISSUES:

        subset = actual[
            actual["issue"] == issue
        ]

        if subset.empty:
            continue

        print()
        print(
            f"[{issue.upper()}]"
        )

        print(
            subset[
                [
                    "severity",
                    "feature",
                    "samples",
                    "mean",
                    "median",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "SEVERITY ORDERING CHECK"
    )
    print("-" * 70)

    print(
        ordering.to_string(
            index=False
        )
    )

    total = len(ordering)

    ordered = int(
        ordering["ordered"].sum()
    )

    if total:

        print()
        print(
            f"Ordered relationships: "
            f"{ordered}/{total} "
            f"({ordered / total * 100:.2f}%)"
        )

    print()
    print(
        "SEVERITY MODEL ACCURACY BY ISSUE"
    )
    print("-" * 70)

    print(
        accuracy.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)


def save_reports(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    ordering: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> None:

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    actual.to_csv(
        REPORT_ROOT
        / "issue_severity_feature_analysis.csv",
        index=False,
    )

    predicted.to_csv(
        REPORT_ROOT
        / "issue_predicted_severity_features.csv",
        index=False,
    )

    ordering.to_csv(
        REPORT_ROOT
        / "issue_severity_ordering.csv",
        index=False,
    )

    accuracy.to_csv(
        REPORT_ROOT
        / "issue_severity_accuracy.csv",
        index=False,
    )

    print()
    print(
        "Reports saved:"
    )

    print(
        REPORT_ROOT
        / "issue_severity_feature_analysis.csv"
    )

    print(
        REPORT_ROOT
        / "issue_predicted_severity_features.csv"
    )

    print(
        REPORT_ROOT
        / "issue_severity_ordering.csv"
    )

    print(
        REPORT_ROOT
        / "issue_severity_accuracy.csv"
    )


def main() -> None:

    print(
        "Image Quality AI - "
        "Issue Severity Validation"
    )

    data = load_test_data()

    feature_columns = get_feature_columns(
        data
    )

    if len(feature_columns) != 13:
        raise RuntimeError(
            f"Expected 13 features, "
            f"found {len(feature_columns)}."
        )

    model = load_model()

    predictions = model.predict(
        data[feature_columns]
    )

    data[
        "predicted_severity"
    ] = predictions

    probabilities = model.predict_proba(
        data[feature_columns]
    )

    data[
        "severity_confidence"
    ] = np.max(
        probabilities,
        axis=1,
    )

    data[
        "severity_correct"
    ] = (
        data["severity"]
        == data["predicted_severity"]
    )

    actual = analyze_actual_severity(
        data
    )

    predicted = analyze_predicted_severity(
        data
    )

    ordering = calculate_severity_ordering(
        data
    )

    accuracy = analyze_prediction_accuracy(
        data
    )

    print_summary(
        actual=actual,
        ordering=ordering,
        accuracy=accuracy,
    )

    save_reports(
        actual=actual,
        predicted=predicted,
        ordering=ordering,
        accuracy=accuracy,
    )


if __name__ == "__main__":
    main()