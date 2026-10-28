"""Ollama /api/chat 请求与响应。"""

from __future__ import annotations

import json

import httpx
import pytest
from app.llm.base import LLMError, Message, ToolCall, ToolSpec, collect_stream
from app.llm.ollama import OllamaProvider

TOOL = ToolSpec(
    name="query_appointment",
    description="查询预约",
    parameters={"type": "object", "properties": {"phone": {"type": "string"}}},
)


def _provider(handler, **kwargs) -> OllamaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaProvider(
        think=kwargs.get("think", False),
        model=kwargs.get("model", "qwen3.8:27b"),
        base_url=kwargs.get("base_url", "http://ollama.test"),
        num_ctx=kwargs.get("num_ctx", 4096),
        temperature=kwargs.get("temperature", 0.2),
        keep_alive=kwargs.get("keep_alive", "5m"),
        timeout=kwargs.get("timeout", 8),
        api_key=kwargs.get("api_key", ""),
        client=client,
    )


@pytest.mark.asyncio
async def test_complete_sends_native_options_and_parses_tool_call() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "query_appointment",
                                "arguments": {"phone": "13800001111"},
                            },
                        }
                    ],
                },
                "prompt_eval_count": 11,
                "eval_count": 4,
            },
        )

    provider = _provider(handler, api_key="ollama-key")
    result = await provider.chat(
        [
            Message(role="user", content="我的预约"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_old", name="query_appointment", arguments={"phone": "1"})
                ],
            ),
            Message(role="tool", name="query_appointment", content='{"time":"15:30"}'),
        ],
        tools=[TOOL],
    )
    await provider.aclose()

    assert seen["path"] == "/api/chat"
    assert seen["auth"] == "Bearer ollama-key"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen3.8:27b"
    assert body["stream"] is False
    assert body["options"] == {"num_ctx": 4096, "temperature": 0.2}
    assert body["keep_alive"] == "5m"
    assert body["think"] is False
    assert body["tools"][0]["function"]["name"] == "query_appointment"
    assert body["messages"][1]["tool_calls"][0]["id"] == "call_old"
    assert body["messages"][2]["tool_name"] == "query_appointment"
    assert result.provider == "ollama"
    assert result.tool_calls[0].arguments == {"phone": "13800001111"}
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4


@pytest.mark.asyncio
async def test_blank_keep_alive_is_omitted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "keep_alive" not in body
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"message": {"content": "好"}})

    provider = _provider(handler, keep_alive="  ")
    result = await provider.chat([Message(role="user", content="在吗")])
    assert result.content == "好"


@pytest.mark.asyncio
async def test_empty_malformed_and_error_payloads() -> None:
    async def capture(response: httpx.Response) -> LLMError:
        provider = _provider(lambda _request: response)
        with pytest.raises(LLMError) as caught:
            await provider.chat([Message(role="user", content="hi")])
        return caught.value

    empty = await capture(httpx.Response(200, json={"message": {"content": "  "}}))
    assert empty.retryable is True
    broken = await capture(httpx.Response(200, text="not-json"))
    assert broken.retryable is True
    wrong_shape = await capture(httpx.Response(200, json=["nope"]))
    assert wrong_shape.retryable is True
    bad_message = await capture(httpx.Response(200, json={"message": "x"}))
    assert bad_message.retryable is True
    missing = await capture(httpx.Response(200, json={"error": "model not found"}))
    assert missing.retryable is False
    denied = await capture(httpx.Response(404, json={"error": "model not found"}))
    assert denied.retryable is False
    assert denied.status_code == 404
    busy = await capture(httpx.Response(503, text="busy"))
    assert busy.retryable is True
    limited = await capture(httpx.Response(429, text="slow"))
    assert limited.retryable is True


@pytest.mark.asyncio
async def test_transport_failures_are_retryable() -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(timeout)
    with pytest.raises(LLMError) as caught:
        await provider.chat([Message(role="user", content="hi")])
    assert caught.value.retryable is True

    provider = _provider(refused)
    with pytest.raises(LLMError) as caught:
        await provider.chat([Message(role="user", content="hi")])
    assert "连接失败" in str(caught.value)


@pytest.mark.asyncio
async def test_stream_emits_text_then_done() -> None:
    lines = "\n".join(
        [
            json.dumps({"model": "qwen3.8:27b", "message": {"content": "你"}, "done": False}),
            "",
            json.dumps({"message": {"content": "好"}, "done": True, "eval_count": 4}),
        ]
    )

    provider = _provider(lambda _request: httpx.Response(200, text=lines))
    chunks = await provider.chat([Message(role="user", content="hi")], stream=True)
    result = await collect_stream(chunks)
    assert result.content == "你好"
    assert result.model == "qwen3.8:27b"
    assert result.completion_tokens == 4


@pytest.mark.asyncio
async def test_stream_tool_call_error_and_http_status() -> None:
    tool_line = json.dumps(
        {
            "model": "m",
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_appointment",
                            "arguments": {"phone": "13800001111"},
                        }
                    }
                ]
            },
            "done": True,
        }
    )
    provider = _provider(lambda _request: httpx.Response(200, text=tool_line))
    chunks = await provider.chat([Message(role="user", content="hi")], stream=True)
    result = await collect_stream(chunks)
    assert result.content == ""
    assert result.tool_calls[0].name == "query_appointment"

    bad = _provider(lambda _request: httpx.Response(200, text="{bad"))
    chunks = await bad.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert "无法解析" in str(caught.value)

    array_line = _provider(lambda _request: httpx.Response(200, text="[1]"))
    chunks = await array_line.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError):
        await collect_stream(chunks)

    failed = _provider(lambda _request: httpx.Response(500, text="down"))
    chunks = await failed.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert caught.value.status_code == 500
    assert caught.value.retryable is True


@pytest.mark.asyncio
async def test_stream_empty_and_upstream_error() -> None:
    empty = _provider(lambda _request: httpx.Response(200, text='{"done": true}'))
    chunks = await empty.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert caught.value.retryable is True

    error_line = _provider(lambda _request: httpx.Response(200, text='{"error":"model not found"}'))
    chunks = await error_line.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert caught.value.retryable is False

    def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(refused)
    chunks = await provider.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError):
        await collect_stream(chunks)


@pytest.mark.asyncio
async def test_think_flag_is_sent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["think"] is True
        return httpx.Response(200, json={"message": {"content": "好"}})

    provider = _provider(handler, think=True)
    result = await provider.chat([Message(role="user", content="hi")])
    assert result.content == "好"


@pytest.mark.asyncio
async def test_stream_with_tools_is_not_implemented() -> None:
    provider = _provider(lambda _request: httpx.Response(500, text="不应发出请求"))
    with pytest.raises(NotImplementedError, match="流式输出暂不支持同时传入工具"):
        await provider.chat([Message(role="user", content="hi")], tools=[TOOL], stream=True)


def test_rejects_bad_model_or_url() -> None:
    with pytest.raises(ValueError):
        OllamaProvider(
            model="  ",
            base_url="http://ollama.test",
            num_ctx=128,
            temperature=0,
            keep_alive="5m",
            timeout=1,
        )
    with pytest.raises(ValueError):
        OllamaProvider(
            model="m",
            base_url="ftp://ollama.test",
            num_ctx=128,
            temperature=0,
            keep_alive="5m",
            timeout=1,
        )


@pytest.mark.asyncio
async def test_owned_client_is_closed() -> None:
    provider = OllamaProvider(
        model="m",
        base_url="http://ollama.test",
        num_ctx=128,
        temperature=0,
        keep_alive="",
        timeout=1,
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"message": {"content": "好"}})
        )
    )
    provider._owns_client = True
    result = await provider.chat([Message(role="user", content="hi")])
    assert result.content == "好"
    await provider.aclose()
    assert provider._client is None
    await provider.aclose()
