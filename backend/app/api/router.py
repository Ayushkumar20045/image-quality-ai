from fastapi import APIRouter

from backend.app.api.routes.analysis import router as analysis_router
from backend.app.api.routes.health import router as health_router

router = APIRouter()

router.include_router(
    health_router
)

router.include_router(
    analysis_router
)
