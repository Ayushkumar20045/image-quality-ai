from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "features"
ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "artifacts"
REPORT_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "evaluation"

TEST_FILE = FEATURE_ROOT / "test.csv"

MODELS = {
    "degradation": ARTIFACT_ROOT / "degradation_model.joblib",
    "severity": ARTIFACT_ROOT / "severity_model.joblib",
}

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
        raise RuntimeError("Test dataset is empty.")

    return test


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
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


def evaluate_target(
    target: str,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    model_path = MODELS[target]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    print(f"\n{'=' * 60}")
    print(f"EVALUATING TARGET: {target.upper()}")
    print(f"{'=' * 60}")

    print(f"Loading model: {model_path.name}")

    model = joblib.load(model_path)

    x_test = test[feature_columns]
    y_test = test[target]

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nOverall metrics:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall:    {recall:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Weighted F1:     {weighted_f1:.4f}")

    labels = sorted(y_test.unique())

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    confusion = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    return {
        "target": target,
        "model": model_path.name,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion,
    }


def save_results(
    results: list[dict],
) -> None:
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "target": result["target"],
                "model": result["model"],
                "accuracy": result["accuracy"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            }
        )

        confusion_path = (
            REPORT_ROOT
            / f"{result['target']}_confusion_matrix.csv"
        )

        result["confusion_matrix"].to_csv(
            confusion_path,
            index=True,
        )

    summary_path = REPORT_ROOT / "test_metrics.csv"

    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
    )

    print("\nEvaluation reports saved:")
    print(f"  Summary: {summary_path}")

    for result in results:
        confusion_path = (
            REPORT_ROOT
            / f"{result['target']}_confusion_matrix.csv"
        )

        print(
            f"  {result['target']} confusion matrix: "
            f"{confusion_path}"
        )


def main() -> None:
    print("Image Quality Model Evaluation")

    test = load_test_data()
    feature_columns = get_feature_columns(test)

    validate_test_data(
        test,
        feature_columns,
    )

    print(f"Test rows: {len(test)}")
    print(f"Features: {len(feature_columns)}")
    print("Test split verified: 2700 rows")

    results = []

    for target in MODELS:
        result = evaluate_target(
            target=target,
            test=test,
            feature_columns=feature_columns,
        )

        results.append(result)

    save_results(results)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"{result['target'].capitalize():12s} | "
            f"Model: {result['model']:28s} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f}"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()