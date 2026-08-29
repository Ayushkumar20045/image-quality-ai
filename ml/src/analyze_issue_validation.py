from __future__ import annotations

from pathlib import Path
from typing import Any

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

DEGRADATION_MODEL_PATH = (
    ARTIFACT_ROOT / "degradation_model.joblib"
)

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


DEGRADATIONS = [
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


def load_test_data() -> pd.DataFrame:

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}"
        )

    test = pd.read_csv(TEST_FILE)

    if test.empty:
        raise RuntimeError(
            "Test dataset is empty."
        )

    return test


def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in dataframe.columns
        if column not in IDENTIFIER_COLUMNS
    ]


def validate_test_data(
    test: pd.DataFrame,
    feature_columns: list[str],
) -> None:

    if len(test) != 2700:
        raise RuntimeError(
            f"Expected 2700 test rows, "
            f"found {len(test)}."
        )

    if len(feature_columns) != 13:
        raise RuntimeError(
            f"Expected 13 features, "
            f"found {len(feature_columns)}."
        )

    if test[feature_columns].isnull().sum().sum() != 0:
        raise RuntimeError(
            "Test feature matrix contains "
            "missing values."
        )


def load_models():

    if not DEGRADATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: "
            f"{DEGRADATION_MODEL_PATH}"
        )

    if not SEVERITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: "
            f"{SEVERITY_MODEL_PATH}"
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


def predict_test_set(
    test: pd.DataFrame,
    feature_columns: list[str],
    degradation_model: Any,
    severity_model: Any,
) -> pd.DataFrame:

    x_test = test[feature_columns]

    degradation_predictions = (
        degradation_model.predict(x_test)
    )

    severity_predictions = (
        severity_model.predict(x_test)
    )

    degradation_probabilities = (
        degradation_model.predict_proba(x_test)
    )

    degradation_classes = (
        degradation_model.classes_
    )

    severity_probabilities = (
        severity_model.predict_proba(x_test)
    )

    severity_classes = (
        severity_model.classes_
    )

    results = test.copy()

    results[
        "predicted_degradation"
    ] = degradation_predictions

    results[
        "predicted_severity"
    ] = severity_predictions

    results[
        "degradation_confidence"
    ] = np.max(
        degradation_probabilities,
        axis=1,
    )

    results[
        "severity_confidence"
    ] = np.max(
        severity_probabilities,
        axis=1,
    )

    for index, label in enumerate(
        degradation_classes
    ):

        results[
            f"prob_{label}"
        ] = degradation_probabilities[
            :, index
        ]

    for index, label in enumerate(
        severity_classes
    ):

        results[
            f"severity_prob_{label}"
        ] = severity_probabilities[
            :, index
        ]

    results[
        "degradation_correct"
    ] = (
        results["degradation"]
        == results["predicted_degradation"]
    )

    results[
        "severity_correct"
    ] = (
        results["severity"]
        == results["predicted_severity"]
    )

    return results


def validate_feature_relationships(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    feature_map = {
        "blur": "sharpness",
        "noise": "high_frequency_residual",
        "underexposure": "mean_brightness",
        "overexposure": "mean_brightness",
        "compression": "local_intensity_variation",
    }

    for issue, feature in feature_map.items():

        issue_rows = results[
            results["degradation"] == issue
        ]

        non_issue_rows = results[
            results["degradation"] != issue
        ]

        issue_mean = (
            issue_rows[feature].mean()
        )

        non_issue_mean = (
            non_issue_rows[feature].mean()
        )

        issue_median = (
            issue_rows[feature].median()
        )

        non_issue_median = (
            non_issue_rows[feature].median()
        )

        difference = (
            issue_mean
            - non_issue_mean
        )

        rows.append(
            {
                "issue": issue,
                "feature": feature,
                "issue_mean": issue_mean,
                "non_issue_mean": non_issue_mean,
                "issue_median": issue_median,
                "non_issue_median": non_issue_median,
                "mean_difference": difference,
            }
        )

    return pd.DataFrame(rows)


def validate_issue_confidence(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in DEGRADATIONS:

        issue_rows = results[
            results["predicted_degradation"]
            == issue
        ].copy()

        if issue_rows.empty:
            continue

        issue_rows["confidence_bucket"] = pd.cut(
            issue_rows[
                "degradation_confidence"
            ],
            bins=[
                0.0,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
            ],
            labels=[
                "<40%",
                "40-50%",
                "50-60%",
                "60-70%",
                "70-80%",
                "80-90%",
                "90%+",
            ],
            include_lowest=True,
        )

        grouped = (
            issue_rows
            .groupby(
                "confidence_bucket",
                observed=False,
            )
        )

        for bucket, group in grouped:

            if len(group) == 0:
                continue

            accuracy = (
                group[
                    "degradation_correct"
                ].mean()
                * 100
            )

            mean_confidence = (
                group[
                    "degradation_confidence"
                ].mean()
                * 100
            )

            rows.append(
                {
                    "issue": issue,
                    "confidence_bucket": str(
                        bucket
                    ),
                    "samples": len(group),
                    "accuracy": round(
                        accuracy,
                        2,
                    ),
                    "mean_confidence": round(
                        mean_confidence,
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_issue_severity(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in DEGRADATIONS:

        issue_rows = results[
            results["predicted_degradation"]
            == issue
        ]

        if issue_rows.empty:
            continue

        severity_distribution = (
            pd.crosstab(
                issue_rows[
                    "predicted_severity"
                ],
                columns="count",
                normalize=False,
            )
        )

        for severity in SEVERITIES:

            count = int(
                severity_distribution
                .get(
                    "count",
                    pd.Series(
                        dtype=float
                    ),
                )
                .get(
                    severity,
                    0,
                )
            )

            rows.append(
                {
                    "issue": issue,
                    "predicted_severity": severity,
                    "samples": count,
                }
            )

    return pd.DataFrame(rows)


def validate_rule_based_signals(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for issue in DEGRADATIONS:

        subset = results[
            results["degradation"] == issue
        ]

        if subset.empty:
            continue

        if issue == "blur":

            signal = (
                subset["sharpness"] < 20
            )

        elif issue == "noise":

            signal = (
                subset[
                    "high_frequency_residual"
                ] > 0.10
            )

        elif issue == "underexposure":

            signal = (
                (subset["mean_brightness"] < 0.18)
                &
                (subset["dark_pixel_ratio"] > 0.35)
            )

        elif issue == "overexposure":

            signal = (
                (subset["mean_brightness"] > 0.82)
                &
                (subset["bright_pixel_ratio"] > 0.35)
            )

        else:

            signal = (
                subset[
                    "local_intensity_variation"
                ] > subset[
                    "local_intensity_variation"
                ].median()
            )

        rows.append(
            {
                "issue": issue,
                "samples": len(subset),
                "feature_signal_rate": (
                    signal.mean() * 100
                ),
                "feature_signal_count": int(
                    signal.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    results: pd.DataFrame,
    feature_validation: pd.DataFrame,
    confidence_validation: pd.DataFrame,
    severity_validation: pd.DataFrame,
    rule_validation: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print(
        "PHASE 3.3 - ISSUE VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "OVERALL ISSUE PREDICTION ACCURACY"
    )
    print("-" * 70)

    issue_accuracy = (
        results["degradation_correct"]
        .mean()
        * 100
    )

    print(
        f"Degradation accuracy: "
        f"{issue_accuracy:.2f}%"
    )

    print()
    print(
        "ISSUE-LEVEL ACCURACY"
    )
    print("-" * 70)

    for issue in DEGRADATIONS:

        subset = results[
            results[
                "predicted_degradation"
            ] == issue
        ]

        if subset.empty:
            continue

        accuracy = (
            subset[
                "degradation_correct"
            ].mean()
            * 100
        )

        confidence = (
            subset[
                "degradation_confidence"
            ].mean()
            * 100
        )

        print(
            f"{issue:15s} | "
            f"samples: {len(subset):4d} | "
            f"accuracy: {accuracy:6.2f}% | "
            f"confidence: {confidence:6.2f}%"
        )

    print()
    print(
        "FEATURE SIGNAL VALIDATION"
    )
    print("-" * 70)

    print(
        feature_validation.to_string(
            index=False
        )
    )

    print()
    print(
        "RULE-BASED SIGNAL VALIDATION"
    )
    print("-" * 70)

    print(
        rule_validation.to_string(
            index=False
        )
    )

    print()
    print(
        "ISSUE CONFIDENCE VS CORRECTNESS"
    )
    print("-" * 70)

    if not confidence_validation.empty:

        print(
            confidence_validation.to_string(
                index=False
            )
        )

    print()
    print(
        "ISSUE × PREDICTED SEVERITY"
    )
    print("-" * 70)

    print(
        severity_validation.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)


def save_reports(
    results: pd.DataFrame,
    feature_validation: pd.DataFrame,
    confidence_validation: pd.DataFrame,
    severity_validation: pd.DataFrame,
    rule_validation: pd.DataFrame,
) -> None:

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_validation.to_csv(
        REPORT_ROOT
        / "issue_validation.csv",
        index=False,
    )

    confidence_validation.to_csv(
        REPORT_ROOT
        / "issue_confidence_validation.csv",
        index=False,
    )

    severity_validation.to_csv(
        REPORT_ROOT
        / "issue_severity_validation.csv",
        index=False,
    )

    rule_validation.to_csv(
        REPORT_ROOT
        / "issue_rule_signal_validation.csv",
        index=False,
    )

    results[
        [
            "filename",
            "degradation",
            "predicted_degradation",
            "degradation_correct",
            "degradation_confidence",
            "severity",
            "predicted_severity",
            "severity_correct",
            "severity_confidence",
        ]
    ].to_csv(
        REPORT_ROOT
        / "issue_prediction_details.csv",
        index=False,
    )

    print()
    print(
        "Reports saved:"
    )

    print(
        f"  {REPORT_ROOT / 'issue_validation.csv'}"
    )

    print(
        f"  {REPORT_ROOT / 'issue_confidence_validation.csv'}"
    )

    print(
        f"  {REPORT_ROOT / 'issue_severity_validation.csv'}"
    )

    print(
        f"  {REPORT_ROOT / 'issue_rule_signal_validation.csv'}"
    )

    print(
        f"  {REPORT_ROOT / 'issue_prediction_details.csv'}"
    )


def main() -> None:

    print(
        "Image Quality AI - "
        "Phase 3.3 Issue Validation"
    )

    test = load_test_data()

    feature_columns = get_feature_columns(
        test
    )

    validate_test_data(
        test,
        feature_columns,
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    (
        degradation_model,
        severity_model,
    ) = load_models()

    print(
        "Models loaded successfully."
    )

    results = predict_test_set(
        test=test,
        feature_columns=feature_columns,
        degradation_model=(
            degradation_model
        ),
        severity_model=severity_model,
    )

    feature_validation = (
        validate_feature_relationships(
            results
        )
    )

    confidence_validation = (
        validate_issue_confidence(
            results
        )
    )

    severity_validation = (
        validate_issue_severity(
            results
        )
    )

    rule_validation = (
        validate_rule_based_signals(
            results
        )
    )

    print_summary(
        results=results,
        feature_validation=feature_validation,
        confidence_validation=confidence_validation,
        severity_validation=severity_validation,
        rule_validation=rule_validation,
    )

    save_reports(
        results=results,
        feature_validation=feature_validation,
        confidence_validation=confidence_validation,
        severity_validation=severity_validation,
        rule_validation=rule_validation,
    )


if __name__ == "__main__":
    main()