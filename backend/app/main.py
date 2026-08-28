from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.analysis import router as analysis_router
from backend.app.api.routes.health import router as health_router
from backend.app.database import init_database


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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_database()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Image Quality AI",
        "status": "running",
        "version": "1.0.0",
    }


app.include_router(health_router)
app.include_router(analysis_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
