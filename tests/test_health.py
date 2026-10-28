"""服务健康检查。"""

from __future__ import annotations

import httpx
import pytest
from app import __version__
from app.config import Settings, get_settings
from app.logging import setup_logging
from app.main import create_app, health_payload, serve


async def test_health_reports_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8:27b")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openai")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "glm-4")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    get_settings.cache_clear()

    application = create_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "llm_provider": "ollama",
        "llm_model": "qwen3.8:27b",
        "fallback_provider": "openai",
        "fallback_enabled": True,
    }


def test_health_payload_omits_empty_fallback() -> None:
    payload = health_payload(Settings(_env_file=None))
    assert payload.fallback_provider is None
    assert payload.fallback_enabled is False
    assert payload.status == "ok"


def test_serve_uses_configured_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(target: str, host: str, port: int, factory: bool) -> None:
        seen["target"] = target
        seen["host"] = host
        seen["port"] = port
        seen["factory"] = factory

    monkeypatch.setattr("app.main.uvicorn.run", fake_run)
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("APP_PORT", "8123")
    get_settings.cache_clear()
    serve()
    assert seen == {
        "target": "app.main:app",
        "host": "127.0.0.1",
        "port": 8123,
        "factory": False,
    }


def test_setup_logging_accepts_unknown_level() -> None:
    setup_logging("NOT_A_LEVEL")
