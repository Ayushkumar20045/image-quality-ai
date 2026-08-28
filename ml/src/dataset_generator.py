from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "ml" / "data" / "raw" / "coco"
OUTPUT_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "quality"

SEED = 42

SPLITS = {
    "train": 840,
    "validation": 180,
    "test": 180,
}

SEVERITIES = ("low", "medium", "high")

DEGRADATIONS = (
    "blur",
    "noise",
    "underexposure",
    "overexposure",
    "compression",
)


def apply_blur(image: Image.Image, severity: str) -> Image.Image:
    radius = {
        "low": 1.2,
        "medium": 2.5,
        "high": 4.0,
    }[severity]

    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(
    image: Image.Image,
    severity: str,
    rng: np.random.Generator,
) -> Image.Image:
    standard_deviation = {
        "low": 8.0,
        "medium": 18.0,
        "high": 32.0,
    }[severity]

    array = np.asarray(image).astype(np.float32)

    noise = rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=array.shape,
    )

    noisy = np.clip(
        array + noise,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(noisy, mode="RGB")


def apply_exposure(
    image: Image.Image,
    severity: str,
    direction: str,
) -> Image.Image:
    factors = {
        "low": 0.75,
        "medium": 0.50,
        "high": 0.30,
    }

    factor = factors[severity]

    if direction == "underexposure":
        return ImageEnhance.Brightness(image).enhance(factor)

    if direction == "overexposure":
        return ImageEnhance.Brightness(image).enhance(
            1.0 + (1.0 - factor)
        )

    raise ValueError(
        f"Unsupported exposure direction: {direction}"
    )


def apply_compression(
    image: Image.Image,
    severity: str,
) -> Image.Image:
    quality = {
        "low": 70,
        "medium": 40,
        "high": 15,
    }[severity]

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def apply_degradation(
    image: Image.Image,
    degradation: str,
    severity: str,
    rng: np.random.Generator,
) -> Image.Image:
    if degradation == "blur":
        return apply_blur(image, severity)

    if degradation == "noise":
        return apply_noise(
            image,
            severity,
            rng,
        )

    if degradation in {"underexposure", "overexposure"}:
        return apply_exposure(
            image,
            severity,
            degradation,
        )

    if degradation == "compression":
        return apply_compression(
            image,
            severity,
        )

    raise ValueError(
        f"Unknown degradation: {degradation}"
    )


def create_output_directories() -> None:
    for split in SPLITS:
        (OUTPUT_ROOT / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def load_source_images(split: str) -> list[Path]:
    source_directory = SOURCE_ROOT / split

    images = sorted(
        source_directory.glob("*.jpg")
    )

    expected_count = SPLITS[split]

    if len(images) != expected_count:
        raise RuntimeError(
            f"{split}: expected {expected_count} source images, "
            f"found {len(images)}."
        )

    return images


def generate_split(
    split: str,
    images: list[Path],
) -> list[dict[str, object]]:
    output_directory = OUTPUT_ROOT / split
    manifest_rows: list[dict[str, object]] = []

    rng = np.random.default_rng(SEED)

    for index, source_path in enumerate(
        images,
        start=1,
    ):
        with Image.open(source_path) as source:
            image = source.convert("RGB")

        source_id = int(source_path.stem)

        for degradation in DEGRADATIONS:
            for severity in SEVERITIES:
                degraded = apply_degradation(
                    image=image,
                    degradation=degradation,
                    severity=severity,
                    rng=rng,
                )

                filename = (
                    f"{source_id:012d}_"
                    f"{degradation}_"
                    f"{severity}.jpg"
                )

                output_path = (
                    output_directory / filename
                )

                degraded.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                )

                manifest_rows.append(
                    {
                        "source_id": source_id,
                        "split": split,
                        "degradation": degradation,
                        "severity": severity,
                        "filename": filename,
                    }
                )

        print(
            f"\r{split}: {index}/{len(images)}",
            end="",
            flush=True,
        )

    print()

    return manifest_rows


def write_manifest(
    rows: list[dict[str, object]],
) -> None:
    manifest_path = OUTPUT_ROOT / "manifest.csv"

    fieldnames = [
        "source_id",
        "split",
        "degradation",
        "severity",
        "filename",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    create_output_directories()

    all_rows: list[dict[str, object]] = []

    for split in SPLITS:
        images = load_source_images(split)

        print(
            f"\nGenerating {split} "
            f"({len(images)} source images)..."
        )

        rows = generate_split(
            split,
            images,
        )

        all_rows.extend(rows)

    write_manifest(all_rows)

    expected_samples = (
        sum(SPLITS.values())
        * len(DEGRADATIONS)
        * len(SEVERITIES)
    )

    print("\nDataset generation complete.")
    print(
        f"Source images: {sum(SPLITS.values())}"
    )
    print(
        f"Degradation types: {len(DEGRADATIONS)}"
    )
    print(
        f"Severity levels: {len(SEVERITIES)}"
    )
    print(
        f"Generated samples: {len(all_rows)}"
    )
    print(
        f"Expected samples: {expected_samples}"
    )
    print(
        f"Manifest: {OUTPUT_ROOT / 'manifest.csv'}"
    )


if __name__ == "__main__":
    main()