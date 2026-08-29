from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def validate_extension(filename: str) -> str:
    """
    Validate the uploaded filename extension.

    Returns:
        The normalized lowercase extension.

    Raises:
        ValueError: If the extension is unsupported.
    """

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. "
            "Use JPG, JPEG, PNG, or WEBP."
        )

    return extension


def validate_image_file(path: Path) -> None:
    """
    Verify that the uploaded file is an actual readable image.

    This validation does not trust the filename or MIME type.
    The file contents are inspected using Pillow.

    Raises:
        FileNotFoundError: If the temporary file does not exist.
        ValueError: If the file is not a valid/readable image.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Uploaded image file could not be found."
        )

    if not path.is_file():
        raise ValueError(
            "Uploaded image is not a valid file."
        )

    if path.stat().st_size == 0:
        raise ValueError(
            "Uploaded image is empty."
        )

    try:
        # First pass verifies the image structure without
        # decoding the entire image into memory.
        with Image.open(path) as image:
            image.verify()

        # Pillow's verify() invalidates the image object,
        # therefore reopen the file and actually load it.
        with Image.open(path) as image:
            image.load()

            if image.width <= 0 or image.height <= 0:
                raise ValueError(
                    "Image has invalid dimensions."
                )

    except UnidentifiedImageError as exc:
        raise ValueError(
            "Uploaded file is not a valid image."
        ) from exc

    except (OSError, SyntaxError) as exc:
        raise ValueError(
            "Image is corrupt or unreadable."
        ) from exc