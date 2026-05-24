from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as rag_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.rag_service import RagService


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    settings = custom_settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(debug=settings.debug)
        app.state.rag_service = RagService(settings=settings)
        yield

    app = FastAPI(
        title=settings.project_name,
        description=settings.project_description,
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.include_router(rag_router)
    return app


app = create_app()
