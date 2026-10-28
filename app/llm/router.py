"""按配置选择模型。主模型出现可重试失败时，自动切换一次备用模型。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.llm.base import (
    ChatChunk,
    ChatResult,
    LLMError,
    LLMProvider,
    Message,
    ToolSpec,
    close_iterator,
    iter_visible_chunks,
    reject_stream_tools,
    strip_think,
)
from app.llm.ollama import OllamaProvider
from app.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class LLMRouter(LLMProvider):
    """主模型失败时切换备用模型。

    可重试的失败：连接失败、超时、HTTP 429、HTTP 5xx、空响应、无法解析的成功响应。
    HTTP 4xx（除 429）不切换，避免把参数错误再打到备用模型。
    流式调用只在还没有产出片段时切换。已经开始输出后出错，不再重试。
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        *,
        fallback_enabled: bool = False,
    ) -> None:
        self.name = "router"
        self.primary = primary
        self.fallback = fallback
        self.fallback_enabled = fallback_enabled

    async def aclose(self) -> None:
        await self.primary.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        reject_stream_tools(tools, stream)
        try:
            return await self._call(self.primary, messages, tools, stream)
        except LLMError as exc:
            return await self._recover(exc, messages, tools, stream)

    async def _call(
        self,
        provider: LLMProvider,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        stream: bool,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        result = await provider.chat(messages, tools=tools, stream=stream)
        if not stream:
            if not isinstance(result, ChatResult):
                raise LLMError(f"{provider.name} 没有返回完整结果", retryable=True)
            cleaned = strip_think(result.content)
            if not cleaned.strip() and not result.tool_calls:
                raise LLMError(f"{provider.name} 返回空响应", retryable=True)
            result.content = cleaned
            return result
        if not hasattr(result, "__anext__"):
            raise LLMError(f"{provider.name} 没有返回流式结果", retryable=True)
        return await self._prime(provider, iter_visible_chunks(result))

    async def _prime(
        self,
        provider: LLMProvider,
        stream: AsyncIterator[ChatChunk],
    ) -> AsyncIterator[ChatChunk]:
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            await close_iterator(stream)
            raise LLMError(f"{provider.name} 返回空响应", retryable=True) from None
        except LLMError:
            await close_iterator(stream)
            raise
        except Exception as exc:
            await close_iterator(stream)
            raise LLMError(f"{provider.name} 流式调用失败：{exc}", retryable=True) from exc

        if first.done and not first.content and not first.tool_calls:
            await close_iterator(stream)
            raise LLMError(f"{provider.name} 返回空响应", retryable=True)

        async def chained() -> AsyncIterator[ChatChunk]:
            try:
                yield first
                async for chunk in stream:
                    yield chunk
            finally:
                await close_iterator(stream)

        return chained()

    async def _recover(
        self,
        exc: LLMError,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        stream: bool,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        if not self.fallback_enabled or not exc.retryable or self.fallback is None:
            raise exc
        fallback_model = getattr(self.fallback, "model", self.fallback.name)
        logger.warning("主模型调用失败，切换到备用模型 %s。原因：%s", fallback_model, exc)
        try:
            return await self._call(self.fallback, messages, tools, stream)
        except LLMError as fallback_exc:
            raise LLMError(
                f"主模型与备用模型都调用失败。主模型：{exc}；备用模型：{fallback_exc}",
                status_code=fallback_exc.status_code,
                retryable=False,
            ) from fallback_exc


def build_provider(
    kind: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    num_ctx: int,
    temperature: float,
    keep_alive: str,
    timeout: float,
    think: bool = False,
    client: httpx.AsyncClient | None = None,
) -> LLMProvider:
    if kind == "ollama":
        return OllamaProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            num_ctx=num_ctx,
            temperature=temperature,
            keep_alive=keep_alive,
            timeout=timeout,
            think=think,
            client=client,
        )
    if kind == "openai":
        return OpenAICompatProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            num_ctx=num_ctx,
            temperature=temperature,
            keep_alive=keep_alive,
            timeout=timeout,
            client=client,
        )
    raise ValueError(f"不支持的模型提供方：{kind}")


def build_router(
    settings: Settings,
    *,
    primary_client: httpx.AsyncClient | None = None,
    fallback_client: httpx.AsyncClient | None = None,
) -> LLMRouter:
    """按配置组装路由。测试可以注入 httpx 客户端，避免访问真实模型。"""
    primary = build_provider(
        settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        num_ctx=settings.llm_num_ctx,
        temperature=settings.llm_temperature,
        keep_alive=settings.llm_keep_alive,
        timeout=settings.llm_timeout,
        think=settings.llm_think,
        client=primary_client,
    )
    fallback: LLMProvider | None = None
    if settings.llm_fallback_provider and settings.llm_fallback_enabled:
        fallback_model = settings.llm_fallback_model or settings.llm_model
        fallback_base = settings.llm_fallback_base_url or settings.llm_base_url
        if not settings.llm_fallback_base_url:
            logger.info("备用模型未配置地址，沿用主模型地址 %s", settings.llm_base_url)
        fallback = build_provider(
            settings.llm_fallback_provider,
            model=fallback_model,
            base_url=fallback_base,
            api_key=settings.llm_fallback_api_key.get_secret_value(),
            num_ctx=settings.llm_num_ctx,
            temperature=settings.llm_temperature,
            keep_alive=settings.llm_keep_alive,
            timeout=settings.llm_timeout,
            think=settings.llm_think,
            client=fallback_client,
        )
    elif settings.llm_fallback_provider:
        logger.info("已配置备用模型，但 LLM_FALLBACK_ENABLED 为 false，不会自动切换")
    return LLMRouter(
        primary,
        fallback,
        fallback_enabled=settings.llm_fallback_enabled,
    )
