"""主模型失败后的备用切换。"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from app.config import Settings
from app.llm.base import (
    ChatChunk,
    ChatResult,
    LLMError,
    LLMProvider,
    Message,
    ToolCall,
    ToolSpec,
    collect_stream,
)
from app.llm.ollama import OllamaProvider
from app.llm.openai_compat import OpenAICompatProvider
from app.llm.router import LLMRouter, build_provider, build_router


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        *,
        result: ChatResult | None = None,
        error: Exception | None = None,
        chunks: list[ChatChunk] | None = None,
        fail_stream: Exception | None = None,
    ) -> None:
        self.name = name
        self.model = name
        self.result = result or ChatResult(content=name, provider=name, model=name)
        self.error = error
        self.chunks = chunks or []
        self.fail_stream = fail_stream
        self.calls = 0
        self.closed = False

    async def chat(
        self,
        messages: list[Message],
        tools=None,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        self.calls += 1
        if stream:
            fail_stream = self.fail_stream
            chunks = self.chunks
            error = self.error

            async def generate() -> AsyncIterator[ChatChunk]:
                if fail_stream is not None:
                    raise fail_stream
                if error is not None:
                    raise error
                for chunk in chunks:
                    yield chunk

            return generate()
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        self.closed = True


def _user() -> list[Message]:
    return [Message(role="user", content="你好")]


@pytest.mark.asyncio
async def test_primary_success_skips_fallback() -> None:
    primary = FakeProvider("primary", result=ChatResult(content="主回答", provider="ollama"))
    fallback = FakeProvider("fallback")
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    result = await router.chat(_user())
    assert result.content == "主回答"
    assert fallback.calls == 0
    await router.aclose()
    assert primary.closed is True
    assert fallback.closed is True


@pytest.mark.asyncio
async def test_retryable_failure_switches_once() -> None:
    primary = FakeProvider("primary", error=LLMError("连接失败", retryable=True))
    fallback = FakeProvider(
        "fallback",
        result=ChatResult(content="备用回答", provider="openai", model="glm-4"),
    )
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    result = await router.chat(_user())
    assert result.content == "备用回答"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_client_error_does_not_switch(caplog: pytest.LogCaptureFixture) -> None:
    primary = FakeProvider(
        "primary",
        error=LLMError("参数错误", status_code=400, retryable=False),
    )
    fallback = FakeProvider("fallback")
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    with caplog.at_level(logging.WARNING), pytest.raises(LLMError) as caught:
        await router.chat(_user())
    assert caught.value.status_code == 400
    assert fallback.calls == 0
    assert "切换到备用模型" not in caplog.text


@pytest.mark.asyncio
async def test_retryable_without_fallback_raises() -> None:
    router = LLMRouter(FakeProvider("primary", error=LLMError("超时", retryable=True)))
    with pytest.raises(LLMError, match="超时"):
        await router.chat(_user())


@pytest.mark.asyncio
async def test_both_providers_fail() -> None:
    router = LLMRouter(
        FakeProvider("primary", error=LLMError("主挂了", retryable=True)),
        FakeProvider("fallback", error=LLMError("备也挂了", status_code=503, retryable=True)),
        fallback_enabled=True,
    )
    with pytest.raises(LLMError) as caught:
        await router.chat(_user())
    assert caught.value.retryable is False
    assert caught.value.status_code == 503
    assert "都调用失败" in str(caught.value)


@pytest.mark.asyncio
async def test_stream_switches_before_first_chunk(caplog: pytest.LogCaptureFixture) -> None:
    primary = FakeProvider("primary", fail_stream=LLMError("流断了", retryable=True))
    fallback = FakeProvider(
        "glm-4",
        chunks=[ChatChunk(content="备", model="glm-4"), ChatChunk(content="用", done=True)],
    )
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    with caplog.at_level(logging.WARNING):
        chunks = await router.chat(_user(), stream=True)
        result = await collect_stream(chunks)
    assert result.content == "备用"
    assert "切换到备用模型 glm-4" in caplog.text
    assert "原因：流断了" in caplog.text


@pytest.mark.asyncio
async def test_stream_empty_generator_and_blank_chunk_switch() -> None:
    empty = FakeProvider("empty")
    blank = FakeProvider("blank", chunks=[ChatChunk(done=True)])
    fallback = FakeProvider("fallback", chunks=[ChatChunk(content="补上")])
    empty_router = LLMRouter(empty, fallback, fallback_enabled=True)
    result = await collect_stream(await empty_router.chat(_user(), stream=True))
    assert result.content == "补上"
    assert fallback.calls == 1

    other = FakeProvider("other", chunks=[ChatChunk(content="另")])
    blank_router = LLMRouter(blank, other, fallback_enabled=True)
    result = await collect_stream(await blank_router.chat(_user(), stream=True))
    assert result.content == "另"


@pytest.mark.asyncio
async def test_stream_tool_chunk_is_not_treated_as_empty() -> None:
    primary = FakeProvider(
        "primary",
        chunks=[
            ChatChunk(
                done=True,
                tool_calls=[ToolCall(id="call_1", name="query_appointment", arguments={})],
            )
        ],
    )
    fallback = FakeProvider("fallback", chunks=[ChatChunk(content="不该出现")])
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    result = await collect_stream(await router.chat(_user(), stream=True))
    assert result.tool_calls[0].name == "query_appointment"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_stream_runtime_error_is_retryable() -> None:
    primary = FakeProvider("primary", fail_stream=RuntimeError("boom"))
    fallback = FakeProvider("fallback", chunks=[ChatChunk(content="好")])
    router = LLMRouter(primary, fallback, fallback_enabled=True)
    result = await collect_stream(await router.chat(_user(), stream=True))
    assert result.content == "好"

    alone = LLMRouter(FakeProvider("primary", fail_stream=RuntimeError("boom")))
    with pytest.raises(LLMError, match="流式调用失败"):
        await collect_stream(await alone.chat(_user(), stream=True))


@pytest.mark.asyncio
async def test_non_stream_provider_is_rejected_for_stream_mode() -> None:
    class Plain(LLMProvider):
        name = "plain"

        async def chat(self, messages, tools=None, stream=False):
            return ChatResult(content="只有全文")

    router = LLMRouter(
        Plain(),
        FakeProvider("fallback", chunks=[ChatChunk(content="流")]),
        fallback_enabled=True,
    )
    result = await collect_stream(await router.chat(_user(), stream=True))
    assert result.content == "流"


@pytest.mark.asyncio
async def test_http_failover_from_ollama_to_openai(caplog: pytest.LogCaptureFixture) -> None:
    def primary(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "primary-model"
        assert body["options"]["num_ctx"] == 4096
        assert body["keep_alive"] == "5m"
        assert body["think"] is False
        return httpx.Response(503, json={"error": "busy"})

    def fallback(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/paas/v4/chat/completions"
        assert request.headers["authorization"] == "Bearer zk-test"
        assert body["model"] == "glm-4"
        assert body["temperature"] == 0.2
        assert "num_ctx" not in body
        assert "keep_alive" not in body
        assert "think" not in body
        return httpx.Response(
            200,
            json={
                "model": "glm-4",
                "choices": [{"message": {"content": "备用回答"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        llm_model="primary-model",
        llm_base_url="http://ollama.test",
        llm_num_ctx=4096,
        llm_temperature=0.2,
        llm_keep_alive="5m",
        llm_timeout=9,
        llm_fallback_enabled=True,
        llm_fallback_provider="openai",
        llm_fallback_model="glm-4",
        llm_fallback_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_fallback_api_key="zk-test",
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(primary)) as primary_client,
        httpx.AsyncClient(transport=httpx.MockTransport(fallback)) as fallback_client,
    ):
        router = build_router(
            settings,
            primary_client=primary_client,
            fallback_client=fallback_client,
        )
        assert isinstance(router.primary, OllamaProvider)
        assert isinstance(router.fallback, OpenAICompatProvider)
        with caplog.at_level(logging.WARNING):
            result = await router.chat(_user())
        await router.aclose()
    assert result.content == "备用回答"
    assert result.provider == "openai"
    assert "切换到备用模型 glm-4" in caplog.text
    assert "原因：" in caplog.text


def test_build_router_inherits_blank_fallback_fields(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_model="qwen3.8:27b",
        llm_base_url="http://localhost:11434",
        llm_fallback_enabled=True,
        llm_fallback_provider="ollama",
    )
    with caplog.at_level(logging.INFO):
        router = build_router(settings)
    assert isinstance(router.fallback, OllamaProvider)
    assert router.fallback.model == "qwen3.8:27b"
    assert router.fallback.base_url == "http://localhost:11434"
    assert "沿用主模型地址" in caplog.text


@pytest.mark.asyncio
async def test_fallback_disabled_does_not_switch(caplog: pytest.LogCaptureFixture) -> None:
    primary = FakeProvider("primary", error=LLMError("连接失败", retryable=True))
    fallback = FakeProvider("fallback", result=ChatResult(content="不该出现"))
    router = LLMRouter(primary, fallback, fallback_enabled=False)
    with caplog.at_level(logging.WARNING), pytest.raises(LLMError, match="连接失败"):
        await router.chat(_user())
    assert fallback.calls == 0
    assert "切换到备用模型" not in caplog.text


def test_build_router_skips_fallback_when_disabled(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        _env_file=None,
        llm_fallback_enabled=False,
        llm_fallback_provider="openai",
        llm_fallback_model="glm-4",
        llm_fallback_base_url="https://example.test/v1",
    )
    with caplog.at_level(logging.INFO):
        router = build_router(settings)
    assert router.fallback is None
    assert router.fallback_enabled is False
    assert "不会自动切换" in caplog.text


@pytest.mark.asyncio
async def test_router_strips_closed_and_unclosed_think() -> None:
    primary = FakeProvider(
        "primary",
        result=ChatResult(content="可见<think>隐藏</think>答案", provider="ollama"),
    )
    router = LLMRouter(primary, fallback_enabled=False)
    result = await router.chat(_user())
    assert result.content == "可见答案"

    thinking = FakeProvider("thinking", result=ChatResult(content="<think>还在想"))
    fallback = FakeProvider("fallback", result=ChatResult(content="备用回答"))
    switched = LLMRouter(thinking, fallback, fallback_enabled=True)
    result = await switched.chat(_user())
    assert result.content == "备用回答"
    assert thinking.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_stream_strips_split_think_tag() -> None:
    primary = FakeProvider(
        "primary",
        chunks=[
            ChatChunk(content="<thi"),
            ChatChunk(content="nk>隐藏</think>你"),
            ChatChunk(content="好", done=True),
        ],
    )
    router = LLMRouter(primary, fallback_enabled=False)
    result = await collect_stream(await router.chat(_user(), stream=True))
    assert result.content == "你好"


@pytest.mark.asyncio
async def test_stream_with_tools_is_not_implemented() -> None:
    primary = FakeProvider("primary", chunks=[ChatChunk(content="不该出现")])
    router = LLMRouter(primary, fallback_enabled=True)
    tool = ToolSpec(name="query_appointment", description="查询预约", parameters={})
    with pytest.raises(NotImplementedError, match="流式输出暂不支持同时传入工具"):
        await router.chat(_user(), tools=[tool], stream=True)
    assert primary.calls == 0


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="不支持的模型提供方"):
        build_provider(
            "azure",
            model="m",
            base_url="http://localhost:11434",
            api_key="",
            num_ctx=128,
            temperature=0,
            keep_alive="5m",
            timeout=1,
        )
