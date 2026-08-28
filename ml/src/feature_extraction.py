from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "quality"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "features"
)

MANIFEST_PATH = DATASET_ROOT / "manifest.csv"

SPLITS = ("train", "validation", "test")


def calculate_sharpness(gray: np.ndarray) -> float:
    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def calculate_gradient_magnitude(
    gray: np.ndarray,
) -> float:
    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32),
    )

    return float(magnitude.mean())


def calculate_exposure_features(
    gray: np.ndarray,
) -> tuple[float, float, float, float]:
    brightness = gray / 255.0

    mean_brightness = float(
        brightness.mean()
    )

    brightness_std = float(
        brightness.std()
    )

    dark_pixel_ratio = float(
        np.mean(brightness < 0.10)
    )

    bright_pixel_ratio = float(
        np.mean(brightness > 0.90)
    )

    return (
        mean_brightness,
        brightness_std,
        dark_pixel_ratio,
        bright_pixel_ratio,
    )


def calculate_high_frequency_residual(
    gray: np.ndarray,
) -> float:
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    return float(
        np.std(residual) / 255.0
    )


def calculate_local_intensity_variation(
    gray: np.ndarray,
) -> float:
    local_mean = cv2.GaussianBlur(
        gray.astype(np.float32),
        (7, 7),
        0,
    )

    local_variation = np.abs(
        gray.astype(np.float32)
        - local_mean
    )

    return float(
        local_variation.mean() / 255.0
    )


def calculate_color_features(
    image: np.ndarray,
) -> tuple[float, float]:
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(np.float32)
        / 255.0
    )

    return (
        float(saturation.mean()),
        float(saturation.std()),
    )


def extract_features(
    image_path: Path,
) -> dict[str, float | int]:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    (
        mean_brightness,
        brightness_std,
        dark_pixel_ratio,
        bright_pixel_ratio,
    ) = calculate_exposure_features(gray)

    (
        mean_saturation,
        saturation_std,
    ) = calculate_color_features(image)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": float(width / height),
        "sharpness": calculate_sharpness(gray),
        "gradient_magnitude": (
            calculate_gradient_magnitude(gray)
        ),
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "dark_pixel_ratio": dark_pixel_ratio,
        "bright_pixel_ratio": bright_pixel_ratio,
        "high_frequency_residual": (
            calculate_high_frequency_residual(gray)
        ),
        "local_intensity_variation": (
            calculate_local_intensity_variation(gray)
        ),
        "mean_saturation": mean_saturation,
        "saturation_std": saturation_std,
    }


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def create_feature_directories() -> None:
    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def process_split(
    split: str,
    rows: list[dict[str, str]],
) -> Path:
    output_path = FEATURE_ROOT / f"{split}.csv"

    feature_rows: list[dict[str, object]] = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        image_path = (
            DATASET_ROOT
            / split
            / row["filename"]
        )

        features = extract_features(
            image_path
        )

        feature_row: dict[str, object] = {
            "source_id": row["source_id"],
            "split": row["split"],
            "degradation": row["degradation"],
            "severity": row["severity"],
            "filename": row["filename"],
            **features,
        }

        feature_rows.append(feature_row)

        print(
            f"\r{split}: {index}/{len(rows)}",
            end="",
            flush=True,
        )

    print()

    fieldnames = list(
        feature_rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(feature_rows)

    return output_path


def main() -> None:
    create_feature_directories()

    manifest_rows = load_manifest()

    for split in SPLITS:
        split_rows = [
            row
            for row in manifest_rows
            if row["split"] == split
        ]

        if not split_rows:
            raise RuntimeError(
                f"No manifest entries found for {split}."
            )

        print(
            f"\nExtracting {split} features "
            f"({len(split_rows)} images)..."
        )

        output_path = process_split(
            split,
            split_rows,
        )

        print(
            f"Saved: {output_path}"
        )

    print(
        "\nFeature extraction complete."
    )
    print(
        f"Feature directory: {FEATURE_ROOT}"
    )


if __name__ == "__main__":
    main()