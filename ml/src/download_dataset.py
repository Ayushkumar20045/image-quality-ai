from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "ml" / "data" / "raw" / "coco"
TEMP_ROOT = DATASET_ROOT / "_download"

ARCHIVE_URL = "http://images.cocodataset.org/zips/val2017.zip"

TOTAL_IMAGES = 1_200
TRAIN_SIZE = 840
VALIDATION_SIZE = 180
TEST_SIZE = 180
RANDOM_SEED = 42

EXPECTED_SOURCE_IMAGES = 5_000
CHUNK_SIZE = 1024 * 1024


def create_directories() -> None:
    for split in ("train", "validation", "test"):
        (DATASET_ROOT / split).mkdir(parents=True, exist_ok=True)

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def download_archive() -> Path:
    archive_path = TEMP_ROOT / "val2017.zip"

    if archive_path.exists():
        print("Using existing COCO archive.")
        return archive_path

    print("Downloading COCO val2017 archive...")

    with requests.get(
        ARCHIVE_URL,
        stream=True,
        timeout=(15, 120),
    ) as response:
        response.raise_for_status()

        with archive_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file.write(chunk)

    print("Download complete.")
    return archive_path


def extract_archive(archive_path: Path) -> Path:
    extracted_root = TEMP_ROOT / "val2017"

    if extracted_root.exists():
        return extracted_root

    print("Extracting COCO archive...")

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(TEMP_ROOT)

    if not extracted_root.is_dir():
        raise RuntimeError("COCO archive did not contain val2017/.")

    return extracted_root


def get_source_images(extracted_root: Path) -> list[Path]:
    images = sorted(extracted_root.glob("*.jpg"))

    if len(images) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_IMAGES} COCO images, "
            f"found {len(images)}."
        )

    return images


def create_splits(images: list[Path]) -> dict[str, list[Path]]:
    generator = random.Random(RANDOM_SEED)

    selected = images.copy()
    generator.shuffle(selected)

    selected = selected[:TOTAL_IMAGES]

    return {
        "train": selected[:TRAIN_SIZE],
        "validation": selected[
            TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE
        ],
        "test": selected[
            TRAIN_SIZE + VALIDATION_SIZE :
        ],
    }


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        return image.size


def copy_images(
    splits: dict[str, list[Path]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    for split, images in splits.items():
        destination = DATASET_ROOT / split

        print(f"\nPreparing {split} ({len(images)} images)...")

        for index, source in enumerate(images, start=1):
            target = destination / source.name

            if not target.exists():
                shutil.copy2(source, target)

            width, height = verify_image(target)

            entries.append(
                {
                    "filename": source.name,
                    "source_id": int(source.stem),
                    "split": split,
                    "width": width,
                    "height": height,
                    "sha256": calculate_sha256(target),
                }
            )

            print(
                f"\r{split}: {index}/{len(images)}",
                end="",
                flush=True,
            )

        print()

    return entries


def write_manifest(entries: list[dict[str, object]]) -> None:
    manifest = {
        "dataset": "COCO 2017 Validation",
        "source": ARCHIVE_URL,
        "random_seed": RANDOM_SEED,
        "selection_size": TOTAL_IMAGES,
        "split_sizes": {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE,
        },
        "images": entries,
    }

    manifest_path = DATASET_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def cleanup() -> None:
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)


def main() -> None:
    create_directories()

    try:
        archive_path = download_archive()
        extracted_root = extract_archive(archive_path)

        source_images = get_source_images(extracted_root)
        splits = create_splits(source_images)

        entries = copy_images(splits)
        write_manifest(entries)

    finally:
        cleanup()

    print("\nDataset preparation complete.")
    print(f"Total images: {len(entries)}")
    print(f"Train: {TRAIN_SIZE}")
    print(f"Validation: {VALIDATION_SIZE}")
    print(f"Test: {TEST_SIZE}")
    print(f"Dataset: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
