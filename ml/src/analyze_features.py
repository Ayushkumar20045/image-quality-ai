from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "features"
REPORT_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "analysis"

SPLITS = ("train", "validation", "test")

IDENTIFIER_COLUMNS = {
    "source_id",
    "split",
    "degradation",
    "severity",
    "filename",
}


def load_features(split: str) -> pd.DataFrame:
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


def create_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    summary = dataframe[feature_columns].describe().T

    summary["missing"] = dataframe[feature_columns].isna().sum()

    summary["unique"] = dataframe[feature_columns].nunique()

    return summary.reset_index(names="feature")


def create_degradation_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    return (
        dataframe.groupby("degradation")[feature_columns]
        .mean()
        .reset_index()
    )


def create_severity_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    severity_order = ["low", "medium", "high"]

    summary = (
        dataframe.groupby("severity")[feature_columns]
        .mean()
        .reindex(severity_order)
        .reset_index()
    )

    return summary


def create_correlation_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    return dataframe[feature_columns].corr()


def write_report(
    split: str,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = create_summary(dataframe, feature_columns)
    degradation_summary = create_degradation_summary(
        dataframe,
        feature_columns,
    )
    severity_summary = create_severity_summary(
        dataframe,
        feature_columns,
    )
    correlation_matrix = create_correlation_matrix(
        dataframe,
        feature_columns,
    )

    summary.to_csv(
        REPORT_ROOT / f"{split}_feature_summary.csv",
        index=False,
    )

    degradation_summary.to_csv(
        REPORT_ROOT / f"{split}_degradation_means.csv",
        index=False,
    )

    severity_summary.to_csv(
        REPORT_ROOT / f"{split}_severity_means.csv",
        index=False,
    )

    correlation_matrix.to_csv(
        REPORT_ROOT / f"{split}_correlation.csv",
    )


def print_overview(
    split: str,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"{split.upper()} FEATURE ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Rows: {len(dataframe)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Missing values: {int(dataframe[feature_columns].isna().sum().sum())}")

    print("\nFeature ranges:")

    for feature in feature_columns:
        minimum = dataframe[feature].min()
        maximum = dataframe[feature].max()

        print(
            f"  {feature:28s} "
            f"min={minimum:.6f} "
            f"max={maximum:.6f}"
        )

    print("\nMean feature values by degradation:")

    degradation_means = create_degradation_summary(
        dataframe,
        feature_columns,
    )

    print(degradation_means.to_string(index=False))

    print("\nMean feature values by severity:")

    severity_means = create_severity_summary(
        dataframe,
        feature_columns,
    )

    print(severity_means.to_string(index=False))


def main() -> None:
    train = load_features("train")

    feature_columns = get_feature_columns(train)

    if len(feature_columns) != 13:
        raise RuntimeError(
            f"Expected 13 image features, found {len(feature_columns)}."
        )

    print(
        f"Analyzing {len(feature_columns)} image-quality features..."
    )

    for split in SPLITS:
        dataframe = load_features(split)

        split_features = get_feature_columns(dataframe)

        if split_features != feature_columns:
            raise RuntimeError(
                f"{split}: feature columns do not match training data."
            )

        print_overview(
            split,
            dataframe,
            feature_columns,
        )

        write_report(
            split,
            dataframe,
            feature_columns,
        )

    print("\nFeature analysis complete.")
    print(f"Reports: {REPORT_ROOT}")


if __name__ == "__main__":
    main()