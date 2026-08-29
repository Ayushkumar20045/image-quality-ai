from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.predict import (
    build_issues,
    calculate_quality_label,
    calculate_quality_score,
    get_class_probabilities,
    get_prediction_confidence,
)


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
    ARTIFACT_ROOT
    / "degradation_model.joblib"
)

SEVERITY_MODEL_PATH = (
    ARTIFACT_ROOT
    / "severity_model.joblib"
)

IDENTIFIER_COLUMNS = {
    "source_id",
    "split",
    "filename",
    "degradation",
    "severity",
}


def load_test_data() -> pd.DataFrame:
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test feature file not found: {TEST_FILE}"
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
            f"Expected 2700 test rows, found {len(test)}."
        )

    if len(feature_columns) != 13:
        raise RuntimeError(
            f"Expected 13 features, found {len(feature_columns)}."
        )

    if test[feature_columns].isnull().sum().sum() != 0:
        raise RuntimeError(
            "Test feature matrix contains missing values."
        )

    if set(test["split"].unique()) != {"test"}:
        raise RuntimeError(
            "Test file contains rows from a split other than 'test'."
        )


def load_models() -> tuple[object, object]:

    if not DEGRADATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {DEGRADATION_MODEL_PATH}"
        )

    if not SEVERITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {SEVERITY_MODEL_PATH}"
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


def analyze_scores(
    test: pd.DataFrame,
    feature_columns: list[str],
    degradation_model: object,
    severity_model: object,
) -> pd.DataFrame:

    x_test = test[feature_columns]

    degradation_predictions = (
        degradation_model.predict(x_test)
    )

    severity_predictions = (
        severity_model.predict(x_test)
    )

    rows: list[dict] = []

    for index in range(len(test)):

        feature_row = test.iloc[index]

        features = {
            column: feature_row[column]
            for column in feature_columns
        }

        feature_dataframe = pd.DataFrame(
            [features],
            columns=feature_columns,
        )

        degradation = str(
            degradation_predictions[index]
        )

        severity = str(
            severity_predictions[index]
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

        rows.append(
            {
                "actual_degradation": str(
                    test.iloc[index]["degradation"]
                ),
                "predicted_degradation": degradation,
                "actual_severity": str(
                    test.iloc[index]["severity"]
                ),
                "predicted_severity": severity,
                "degradation_confidence": (
                    degradation_confidence
                ),
                "severity_confidence": (
                    severity_confidence
                ),
                "quality_score": quality_score,
                "quality_label": quality_label,
                "issue_count": len(issues),
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    results: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("QUALITY SCORE CALIBRATION ANALYSIS")
    print("=" * 70)

    print()
    print("OVERALL SCORE DISTRIBUTION")
    print("-" * 70)

    print(
        f"Mean score   : "
        f"{results['quality_score'].mean():.2f}"
    )

    print(
        f"Median score : "
        f"{results['quality_score'].median():.2f}"
    )

    print(
        f"Minimum      : "
        f"{results['quality_score'].min():.2f}"
    )

    print(
        f"Maximum      : "
        f"{results['quality_score'].max():.2f}"
    )

    print()
    print("AVERAGE SCORE BY ACTUAL SEVERITY")
    print("-" * 70)

    severity_summary = (
        results
        .groupby("actual_severity")["quality_score"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reindex(
            ["low", "medium", "high"]
        )
    )

    print(
        severity_summary.round(2)
    )

    print()
    print("AVERAGE SCORE BY PREDICTED SEVERITY")
    print("-" * 70)

    predicted_severity_summary = (
        results
        .groupby("predicted_severity")["quality_score"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reindex(
            ["low", "medium", "high"]
        )
    )

    print(
        predicted_severity_summary.round(2)
    )

    print()
    print("AVERAGE SCORE BY ACTUAL DEGRADATION")
    print("-" * 70)

    degradation_summary = (
        results
        .groupby("actual_degradation")["quality_score"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .sort_values(
            "mean",
            ascending=True,
        )
    )

    print(
        degradation_summary.round(2)
    )

    print()
    print("QUALITY LABEL DISTRIBUTION")
    print("-" * 70)

    print(
        results["quality_label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("QUALITY LABEL BY ACTUAL SEVERITY")
    print("-" * 70)

    label_by_severity = pd.crosstab(
        results["actual_severity"],
        results["quality_label"],
        normalize="index",
    ) * 100

    print(
        label_by_severity.round(2)
    )

    print()
    print("QUALITY SCORE BY ACTUAL SEVERITY")
    print("-" * 70)

    severity_score_means = (
        results
        .groupby("actual_severity")[
            "quality_score"
        ]
        .mean()
    )

    for severity in [
        "low",
        "medium",
        "high",
    ]:

        if severity in severity_score_means:
            print(
                f"{severity:8s}: "
                f"{severity_score_means[severity]:.2f}"
            )

    print()
    print("SCORE ORDERING CHECK")
    print("-" * 70)

    low_score = severity_score_means.get(
        "low",
        np.nan,
    )

    medium_score = severity_score_means.get(
        "medium",
        np.nan,
    )

    high_score = severity_score_means.get(
        "high",
        np.nan,
    )

    if (
        low_score > medium_score
        > high_score
    ):
        print(
            "PASS: Average quality score decreases "
            "as actual severity increases."
        )
    else:
        print(
            "WARNING: Average quality score does "
            "not decrease monotonically with severity."
        )

    print()
    print("SCORE RANGE OVERLAP")
    print("-" * 70)

    for severity in [
        "low",
        "medium",
        "high",
    ]:

        group = results[
            results["actual_severity"]
            == severity
        ]

        print(
            f"{severity:8s}: "
            f"{group['quality_score'].min():.2f}"
            f" - "
            f"{group['quality_score'].max():.2f}"
        )

    print()
    print("CORRELATION")
    print("-" * 70)

    severity_mapping = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    numeric_severity = (
        results["actual_severity"]
        .map(severity_mapping)
    )

    correlation = (
        numeric_severity
        .corr(results["quality_score"])
    )

    print(
        "Actual severity vs quality score: "
        f"{correlation:.4f}"
    )

    print(
        "Expected direction: negative"
    )

    print()
    print("=" * 70)


def save_results(
    results: pd.DataFrame,
) -> None:

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_ROOT
        / "quality_score_calibration.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Detailed calibration results saved: "
        f"{output_path}"
    )


def main() -> None:

    test = load_test_data()

    feature_columns = get_feature_columns(
        test
    )

    validate_test_data(
        test,
        feature_columns,
    )

    degradation_model, severity_model = (
        load_models()
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        "Generating quality scores..."
    )

    results = analyze_scores(
        test=test,
        feature_columns=feature_columns,
        degradation_model=degradation_model,
        severity_model=severity_model,
    )

    print_summary(
        results
    )

    save_results(
        results
    )


if __name__ == "__main__":
    main()