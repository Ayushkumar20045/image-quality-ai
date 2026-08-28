from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "features"
ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "artifacts"

TARGET_COLUMNS = (
    "degradation",
    "severity",
)

IDENTIFIER_COLUMNS = (
    "source_id",
    "split",
    "filename",
    "degradation",
    "severity",
)


class FeaturePreprocessor:
    """Prepare extracted image features for model training and inference."""

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()

    def fit(self, dataframe: pd.DataFrame) -> None:
        self.feature_columns = [
            column
            for column in dataframe.columns
            if column not in IDENTIFIER_COLUMNS
        ]

        if not self.feature_columns:
            raise ValueError("No numerical feature columns were found.")

        features = dataframe[self.feature_columns].astype(float)

        if features.isna().any().any():
            raise ValueError("Training features contain missing values.")

        self.scaler.fit(features)

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError(
                "Preprocessor must be fitted before transformation."
            )

        features = dataframe[self.feature_columns].astype(float)

        if features.isna().any().any():
            raise ValueError("Input features contain missing values.")

        return self.scaler.transform(features)

    def fit_transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        self.fit(dataframe)
        return self.transform(dataframe)


def load_split(split: str) -> pd.DataFrame:
    path = FEATURE_ROOT / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    return pd.read_csv(path)


def save_preprocessor(preprocessor: FeaturePreprocessor) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    path = ARTIFACT_ROOT / "feature_preprocessor.joblib"

    joblib.dump(preprocessor, path)

    return path


def main() -> None:
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    preprocessor = FeaturePreprocessor()

    train_features = preprocessor.fit_transform(train)
    validation_features = preprocessor.transform(validation)
    test_features = preprocessor.transform(test)

    artifact_path = save_preprocessor(preprocessor)

    print("Preprocessing complete.")
    print(f"Features: {len(preprocessor.feature_columns)}")
    print(f"Training matrix: {train_features.shape}")
    print(f"Validation matrix: {validation_features.shape}")
    print(f"Test matrix: {test_features.shape}")
    print(f"Preprocessor: {artifact_path}")


if __name__ == "__main__":
    main()