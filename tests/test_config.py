"""配置：默认模型、密钥来源、非法值。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from app.config import Settings, get_settings, require_http_url
from pydantic import SecretStr, ValidationError

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_keep_model_out_of_business_code() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "qwen3.8:27b"
    assert settings.llm_base_url == "http://localhost:11434"
    assert settings.llm_num_ctx == 8192
    assert settings.llm_temperature == 0.3
    assert settings.llm_keep_alive == "30m"
    assert settings.llm_timeout == 120
    assert settings.app_port == 8100
    assert settings.llm_think is False
    assert settings.llm_fallback_enabled is False
    assert settings.llm_fallback_provider == ""
    assert settings.embed_model == "bge-m3"
    assert settings.session_max_turns == 8


def test_secrets_default_empty_and_stay_out_of_repr() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-live-secret",
        wecom_secret="corp-live-secret",
        wecom_token="callback-token-secret",
        wecom_encoding_aes_key="encoding-aes-secret",
        wecom_webhook_key="webhook-key-secret",
        admin_token="admin-token-secret",
    )
    assert settings.llm_api_key.get_secret_value() == "sk-live-secret"
    assert settings.wecom_secret.get_secret_value() == "corp-live-secret"
    rendered = repr(settings)
    assert "**********" in rendered
    for secret in (
        "sk-live-secret",
        "corp-live-secret",
        "callback-token-secret",
        "encoding-aes-secret",
        "webhook-key-secret",
        "admin-token-secret",
    ):
        assert secret not in rendered
    fresh = Settings(_env_file=None)
    for value in (
        fresh.llm_api_key,
        fresh.llm_fallback_api_key,
        fresh.wecom_secret,
        fresh.wecom_token,
        fresh.wecom_encoding_aes_key,
        fresh.wecom_webhook_key,
        fresh.admin_token,
    ):
        assert isinstance(value, SecretStr)
        assert value.get_secret_value() == ""


def test_think_and_fallback_flags_follow_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_THINK", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_think is True
    assert settings.llm_fallback_enabled is False


def test_example_wecom_values_log_warning(caplog: pytest.LogCaptureFixture) -> None:
    from app.config import warn_example_secrets

    settings = Settings(
        _env_file=None,
        wecom_corp_id="wx5823bf96d3bd56c7",
        wecom_token="QDG6eK",
        wecom_encoding_aes_key="jWmYm7qr5nMoAUwZRjGtBxmz3KA1tkAj3ykkR6q2B2C",
    )
    with caplog.at_level(logging.WARNING):
        warn_example_secrets(settings)
    assert "官方文档示例" in caplog.text
    assert "WECOM_TOKEN" in caplog.text
    assert "WECOM_ENCODING_AES_KEY" in caplog.text

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        warn_example_secrets(Settings(_env_file=None))
    assert caplog.records == []


def test_environment_overrides_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    monkeypatch.setenv("LLM_NUM_CTX", "4096")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_model == "qwen3:4b"
    assert settings.llm_provider == "openai"
    assert settings.llm_num_ctx == 4096


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("llm_provider", "azure"),
        ("llm_model", "  "),
        ("llm_base_url", "ftp://models.example"),
        ("llm_temperature", 3),
        ("llm_num_ctx", 0),
        ("llm_timeout", 0),
        ("llm_fallback_provider", "local"),
        ("llm_fallback_base_url", "not-a-url"),
    ],
)
def test_invalid_settings_raise(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field_name: value})


def test_blank_fallback_url_is_allowed() -> None:
    settings = Settings(_env_file=None, llm_fallback_base_url="  ")
    assert settings.llm_fallback_base_url == ""


def test_require_http_url_strips_slash() -> None:
    assert require_http_url("http://localhost:11434/") == "http://localhost:11434"
    with pytest.raises(ValueError):
        require_http_url("localhost:11434")


def test_model_name_is_not_hardcoded_outside_config() -> None:
    offenders: list[str] = []
    for path in (ROOT / "app").rglob("*.py"):
        if path.name == "config.py":
            continue
        if "qwen3.8:27b" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_gitignore_contains_required_entries() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (
        ".venv/",
        ".env",
        "data/",
        "bench/results/",
        "__pycache__/",
        ".coverage",
        ".pytest_cache/",
    ):
        assert entry in lines
    assert ".env.example" not in lines


def test_gitignore_excludes_env_and_databases() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert ".env.example" not in lines
    assert any(line.endswith(".db") or "*.db" in line for line in lines)
