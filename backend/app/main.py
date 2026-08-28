from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml.src.predict import predict_image

from .schemas import AnalysisResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


app = FastAPI(
    title="Image Quality AI",
    description=(
        "AI-powered image quality analysis API "
        "using machine learning."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Image Quality AI",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.post(
    "/api/analyze",
    response_model=AnalysisResponse,
)
async def analyze_image(
    file: UploadFile = File(...),
) -> AnalysisResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, or WEBP."
            ),
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            shutil.copyfileobj(
                file.file,
                temporary_file,
            )

        result = predict_image(
            temporary_path
        )

        result["image"] = file.filename

        return AnalysisResponse(
            **result
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed: "
                f"{exc}"
            ),
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )