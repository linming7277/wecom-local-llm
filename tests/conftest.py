"""测试隔离：清掉进程里的配置，避免开发机环境变量串进断言。"""

from __future__ import annotations

import os

import pytest
from app.config import get_settings

_PREFIXES = (
    "APP_",
    "LLM_",
    "EMBED_",
    "RAG_",
    "WECOM_",
    "REPORT_",
    "SESSION_",
    "DATABASE_",
    "ADMIN_",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch):
    for key in list(os.environ):
        if key.startswith(_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
