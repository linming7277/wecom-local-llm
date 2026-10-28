"""OpenAI 兼容接口的地址、鉴权和流式工具调用。"""

from __future__ import annotations

import json

import httpx
import pytest
from app.llm.base import LLMError, Message, ToolCall, ToolSpec, collect_stream
from app.llm.openai_compat import OpenAICompatProvider, chat_completions_url

TOOL = ToolSpec(
    name="record_intent",
    description="记录意向",
    parameters={"type": "object", "properties": {"project": {"type": "string"}}},
)


def test_chat_completions_url_keeps_vendor_path() -> None:
    assert chat_completions_url("http://localhost:11434") == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert chat_completions_url("http://localhost:11434/v1/") == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert chat_completions_url("https://open.bigmodel.cn/api/paas/v4") == (
        "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    )
    with pytest.raises(ValueError):
        chat_completions_url("open.bigmodel.cn/api/paas/v4")


def _provider(handler, **kwargs) -> OpenAICompatProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(
        model=kwargs.get("model", "glm-4"),
        base_url=kwargs.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=kwargs.get("api_key", "zk-test"),
        num_ctx=8192,
        temperature=0.4,
        keep_alive="30m",
        timeout=6,
        client=client,
    )


@pytest.mark.asyncio
async def test_complete_uses_standard_fields_and_bearer_token() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "glm-4",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_intent",
                                        "arguments": '{"project":"水光","budget":"3000"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 2},
            },
        )

    provider = _provider(handler)
    result = await provider.chat(
        [
            Message(role="user", content="想做水光"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_0", name="record_intent", arguments={"project": "旧"})
                ],
            ),
            Message(role="tool", name="record_intent", tool_call_id="call_0", content="ok"),
        ],
        tools=[TOOL],
    )
    body = seen["body"]
    assert seen["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert seen["auth"] == "Bearer zk-test"
    assert isinstance(body, dict)
    assert body["model"] == "glm-4"
    assert body["temperature"] == 0.4
    assert "num_ctx" not in body
    assert "keep_alive" not in body
    assert body["tool_choice"] == "auto"
    encoded = body["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(encoded) == {"project": "旧"}
    assert body["messages"][2]["tool_call_id"] == "call_0"
    assert result.content == ""
    assert result.tool_calls[0].arguments == {"project": "水光", "budget": "3000"}
    assert result.prompt_tokens == 6
    assert result.completion_tokens == 2
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_omits_authorization_when_key_is_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "你好"}}]},
        )

    provider = _provider(handler, api_key="", base_url="http://localhost:11434")
    result = await provider.chat([Message(role="user", content="hi")])
    assert result.content == "你好"
    assert provider.chat_url == "http://localhost:11434/v1/chat/completions"
    assert provider.num_ctx == 8192
    assert provider.keep_alive == "30m"


@pytest.mark.asyncio
async def test_error_payloads() -> None:
    cases = [
        (httpx.Response(200, json={"choices": []}), True, None),
        (httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]}), True, None),
        (httpx.Response(200, text="nope"), True, None),
        (httpx.Response(200, json=[1]), True, None),
        (httpx.Response(200, json={"choices": [{"message": "x"}]}), True, None),
        (httpx.Response(401, text="unauthorized"), False, 401),
        (httpx.Response(429, text="slow"), True, 429),
    ]
    for response, retryable, status in cases:
        provider = _provider(lambda _request, response=response: response)
        with pytest.raises(LLMError) as caught:
            await provider.chat([Message(role="user", content="hi")])
        assert caught.value.retryable is retryable
        assert caught.value.status_code == status


@pytest.mark.asyncio
async def test_stream_assembles_tool_arguments() -> None:
    events = [
        'data: {"model":"glm-4","choices":[{"delta":{"content":"已"}}]}',
        'data: {"choices":[{"delta":{"content":"记录"}}]}',
        (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_9",
                                        "function": {"name": "record_intent", "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        ),
        (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"project":"'}}
                                ]
                            }
                        }
                    ]
                }
            )
        ),
        (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '水光"}'}}]
                            }
                        }
                    ]
                }
            )
        ),
        "data: [DONE]",
        ": keep-alive",
    ]
    provider = _provider(lambda _request: httpx.Response(200, text="\n".join(events)))
    chunks = await provider.chat([Message(role="user", content="记录")], stream=True)
    result = await collect_stream(chunks)
    assert result.content == "已记录"
    assert result.model == "glm-4"
    assert result.tool_calls[0].id == "call_9"
    assert result.tool_calls[0].arguments == {"project": "水光"}


@pytest.mark.asyncio
async def test_stream_failures() -> None:
    empty = _provider(lambda _request: httpx.Response(200, text="data: [DONE]\n"))
    chunks = await empty.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert caught.value.retryable is True

    bad = _provider(lambda _request: httpx.Response(200, text="data: {bad}\n"))
    chunks = await bad.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert "无法解析" in str(caught.value)

    array_event = _provider(lambda _request: httpx.Response(200, text="data: [1]\n"))
    chunks = await array_event.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError):
        await collect_stream(chunks)

    denied = _provider(lambda _request: httpx.Response(401, text="no"))
    chunks = await denied.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert caught.value.retryable is False

    def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(refused)
    chunks = await provider.chat([Message(role="user", content="hi")], stream=True)
    with pytest.raises(LLMError) as caught:
        await collect_stream(chunks)
    assert "连接失败" in str(caught.value)


@pytest.mark.asyncio
async def test_stream_with_tools_is_not_implemented() -> None:
    provider = _provider(lambda _request: httpx.Response(500, text="不应发出请求"))
    with pytest.raises(NotImplementedError, match="流式输出暂不支持同时传入工具"):
        await provider.chat([Message(role="user", content="hi")], tools=[TOOL], stream=True)


def test_rejects_empty_model() -> None:
    with pytest.raises(ValueError):
        OpenAICompatProvider(
            model="",
            base_url="https://example.test/v1",
            api_key="",
            num_ctx=1,
            temperature=0,
            keep_alive="",
            timeout=1,
        )


@pytest.mark.asyncio
async def test_owned_client_is_closed() -> None:
    provider = OpenAICompatProvider(
        model="glm-4",
        base_url="https://example.test/v1",
        api_key="",
        num_ctx=1,
        temperature=0,
        keep_alive="",
        timeout=1,
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})
        )
    )
    provider._owns_client = True
    result = await provider.chat([Message(role="user", content="hi")])
    assert result.content == "好"
    await provider.aclose()
    assert provider._client is None
