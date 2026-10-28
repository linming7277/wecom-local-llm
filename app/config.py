"""运行配置。密钥只从环境变量或 .env 读取。"""

from __future__ import annotations

import logging
from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 企业微信加解密文档里的公开示例。只能用于本地模拟。
EXAMPLE_WECOM_CORP_ID = "wx5823bf96d3bd56c7"
EXAMPLE_WECOM_TOKEN = "QDG6eK"
EXAMPLE_WECOM_AES_KEY = "jWmYm7qr5nMoAUwZRjGtBxmz3KA1tkAj3ykkR6q2B2C"


def require_http_url(url: str) -> str:
    """校验并规范化 http(s) 地址，去掉末尾斜杠。"""
    base = url.strip().rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("模型接口地址必须是 http 或 https URL")
    return base


class Settings(BaseSettings):
    """进程配置。

    业务代码从这里读取模型名和密钥，不要在别处写死。
    企微、知识库和日报字段会在后续里程碑使用。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8100, ge=1, le=65535)
    app_env: str = "dev"

    llm_provider: str = "ollama"
    llm_model: str = "qwen3.8:27b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: SecretStr = SecretStr("")
    llm_num_ctx: int = Field(default=8192, ge=1)
    llm_temperature: float = Field(default=0.3, ge=0, le=2)
    llm_keep_alive: str = "30m"
    llm_timeout: float = Field(default=120, gt=0)
    llm_think: bool = False

    llm_fallback_enabled: bool = False
    llm_fallback_provider: str = ""
    llm_fallback_model: str = ""
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: SecretStr = SecretStr("")

    embed_model: str = "bge-m3"
    embed_base_url: str = "http://localhost:11434"
    rag_top_k: int = Field(default=4, ge=1)

    wecom_corp_id: str = ""
    wecom_agent_id: str = ""
    wecom_secret: SecretStr = SecretStr("")
    wecom_token: SecretStr = SecretStr("")
    wecom_encoding_aes_key: SecretStr = SecretStr("")
    wecom_webhook_key: SecretStr = SecretStr("")

    report_hour: int = Field(default=18, ge=0, le=23)
    report_minute: int = Field(default=0, ge=0, le=59)
    report_timezone: str = "Asia/Shanghai"
    session_max_turns: int = Field(default=8, ge=1)
    database_url: str = "sqlite:///./data/wecom.db"
    admin_token: SecretStr = SecretStr("")

    @field_validator("llm_provider", "llm_fallback_provider")
    @classmethod
    def normalize_provider(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip().lower()
        if info.field_name == "llm_fallback_provider" and normalized == "":
            return ""
        if normalized not in {"ollama", "openai"}:
            raise ValueError("模型提供方只能是 ollama 或 openai")
        return normalized

    @field_validator("llm_model")
    @classmethod
    def model_must_be_present(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("模型名不能为空")
        return cleaned

    @field_validator("llm_base_url", "embed_base_url")
    @classmethod
    def validate_required_url(cls, value: str) -> str:
        return require_http_url(value)

    @field_validator("llm_fallback_base_url")
    @classmethod
    def validate_optional_url(cls, value: str) -> str:
        if not value.strip():
            return ""
        return require_http_url(value)


def warn_example_secrets(settings: Settings) -> None:
    """示例 CorpID、Token、EncodingAESKey 出现在配置里时提醒，避免被带到生产。"""
    found: list[str] = []
    if settings.wecom_corp_id == EXAMPLE_WECOM_CORP_ID:
        found.append("WECOM_CORP_ID")
    if settings.wecom_token.get_secret_value() == EXAMPLE_WECOM_TOKEN:
        found.append("WECOM_TOKEN")
    if settings.wecom_encoding_aes_key.get_secret_value() == EXAMPLE_WECOM_AES_KEY:
        found.append("WECOM_ENCODING_AES_KEY")
    if not found:
        return
    logger.warning(
        "检测到企微官方文档示例值：%s。这些值只能用于本地模拟，不能用于生产。",
        "、".join(found),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的配置。测试里修改环境变量后需要 cache_clear。"""
    return Settings()
