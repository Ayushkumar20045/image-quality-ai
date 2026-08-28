from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.app.database import (
    get_analysis,
    get_analysis_history,
    save_analysis,
)
from backend.app.schemas import (
    AnalysisResponse,
    HistoryResponse,
)
from backend.app.services.image_analyzer import analyze_image
from backend.app.utils.image import (
    validate_extension,
    validate_image_file,
)

router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
async def analyze(
    file: UploadFile = File(...),
) -> AnalysisResponse:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    try:
        extension = validate_extension(
            file.filename
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            total_size = 0

            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image is too large. "
                            "Maximum allowed size is 10 MB."
                        ),
                    )

                temporary_file.write(chunk)

        try:
            validate_image_file(
                temporary_path
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        result = analyze_image(
            temporary_path,
            image_name=file.filename,
        )

        analysis_id = save_analysis(
            result
        )

        saved_analysis = get_analysis(
            analysis_id
        )

        if saved_analysis is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analysis was saved but "
                    "could not be retrieved."
                ),
            )

        return AnalysisResponse(
            **saved_analysis
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed. "
                "Please try another image."
            ),
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )


@router.get(
    "/history",
    response_model=HistoryResponse,
)
def history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> HistoryResponse:

    analyses = get_analysis_history(
        limit=limit
    )

    return HistoryResponse(
        total=len(analyses),
        analyses=analyses,
    )


@router.get(
    "/history/{analysis_id}",
    response_model=AnalysisResponse,
)
def history_by_id(
    analysis_id: int,
) -> AnalysisResponse:

    analysis = get_analysis(
        analysis_id
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return AnalysisResponse(
        **analysis
    )
