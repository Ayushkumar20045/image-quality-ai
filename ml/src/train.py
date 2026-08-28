from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "features"
ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "artifacts"

IDENTIFIER_COLUMNS = {
    "source_id",
    "split",
    "filename",
    "degradation",
    "severity",
}

TARGETS = {
    "degradation": "degradation",
    "severity": "severity",
}


def load_split(split: str) -> pd.DataFrame:
    path = FEATURE_ROOT / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    return pd.read_csv(path)


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column not in IDENTIFIER_COLUMNS
    ]


def build_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    skewed_features = [
        "sharpness",
        "gradient_magnitude",
    ]

    standard_features = [
        feature
        for feature in feature_columns
        if feature not in skewed_features
    ]

    return ColumnTransformer(
        transformers=[
            (
                "skewed",
                Pipeline(
                    steps=[
                        (
                            "log",
                            FunctionTransformer(
                                np.log1p,
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                skewed_features,
            ),
            (
                "standard",
                StandardScaler(),
                standard_features,
            ),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def build_pipeline(
    feature_columns: list[str],
    model: object,
) -> Pipeline:
    preprocessor = build_preprocessor(feature_columns)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    predictions = pipeline.predict(x_validation)

    return {
        "accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
            )
        ),
    }


def train_target(
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    x_train = train[feature_columns]
    y_train = train[target]

    x_validation = validation[feature_columns]
    y_validation = validation[target]

    results: dict[str, dict[str, float]] = {}

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -1.0

    print(f"\n{'=' * 60}")
    print(f"TRAINING TARGET: {target.upper()}")
    print(f"{'=' * 60}")

    for name, model in build_models().items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(
            feature_columns,
            model,
        )

        pipeline.fit(
            x_train,
            y_train,
        )

        metrics = evaluate_model(
            pipeline,
            x_validation,
            y_validation,
        )

        results[name] = metrics

        print(
            f"  Accuracy: {metrics['accuracy']:.4f}"
        )
        print(
            f"  Macro F1: {metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError(
            f"No model was successfully trained for {target}."
        )

    artifact_path = (
        ARTIFACT_ROOT / f"{target}_model.joblib"
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_pipeline,
        artifact_path,
    )

    print(
        f"\nBest {target} model: {best_name}"
    )
    print(
        f"Validation Macro F1: {best_score:.4f}"
    )
    print(
        f"Saved: {artifact_path}"
    )

    return best_name, best_pipeline, results


def main() -> None:
    train = load_split("train")
    validation = load_split("validation")

    feature_columns = get_feature_columns(train)

    if len(feature_columns) != 13:
        raise RuntimeError(
            f"Expected 13 features, found {len(feature_columns)}."
        )

    if get_feature_columns(validation) != feature_columns:
        raise RuntimeError(
            "Training and validation feature columns do not match."
        )

    print("Image Quality Model Training")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Features: {len(feature_columns)}")

    for target in TARGETS.values():
        train_target(
            target=target,
            train=train,
            validation=validation,
            feature_columns=feature_columns,
        )

    print("\nTraining complete.")
    print(f"Artifacts: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()