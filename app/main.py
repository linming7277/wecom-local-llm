"""FastAPI 入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings, warn_example_secrets
from app.llm.router import LLMRouter, build_router
from app.logging import setup_logging


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    llm_model: str
    fallback_provider: str | None = None
    fallback_enabled: bool = False


def health_payload(settings: Settings) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        fallback_provider=settings.llm_fallback_provider or None,
        fallback_enabled=settings.llm_fallback_enabled,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = get_settings()
    warn_example_secrets(settings)
    app.state.settings = settings
    app.state.llm = build_router(settings)
    try:
        yield
    finally:
        router: LLMRouter = app.state.llm
        await router.aclose()


def create_app() -> FastAPI:
    application = FastAPI(title="wecom-local-llm", version=__version__, lifespan=lifespan)

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        settings = getattr(request.app.state, "settings", None) or get_settings()
        return health_payload(settings)

    return application


app = create_app()


def serve() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        factory=False,
    )


if __name__ == "__main__":
    serve()
