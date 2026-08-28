from __future__ import annotations

from pathlib import Path

from PIL import Image


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_extension(filename: str) -> str:
    """
    Validate and return a normalized image extension.
    """
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. "
            "Use JPG, JPEG, PNG, or WEBP."
        )

    return extension


def validate_image_file(path: str | Path) -> None:
    """
    Verify that the file exists and is a readable image.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(
            f"Invalid or unreadable image: {exc}"
        ) from exc


def validate_file_size(size: int) -> None:
    """
    Enforce the backend's 10 MB upload limit.
    """
    if size > MAX_FILE_SIZE:
        raise ValueError(
            "Image is too large. "
            "Maximum allowed size is 10 MB."
        )
