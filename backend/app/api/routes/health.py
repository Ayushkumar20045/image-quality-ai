from fastapi import APIRouter

router = APIRouter(
    prefix="/api",
    tags=["Health"],
)


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }
