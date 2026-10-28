"""工具参数解析与流式拼接。"""

from __future__ import annotations

import httpx
import pytest
from app.llm.base import (
    ChatChunk,
    ChatResult,
    LLMError,
    LLMProvider,
    Message,
    ThinkStripper,
    ToolCall,
    close_iterator,
    collect_stream,
    error_from_httpx,
    error_from_status,
    iter_visible_chunks,
    normalize_content,
    optional_int,
    parse_tool_arguments,
    parse_tool_calls,
    strip_think,
)


def test_parse_tool_arguments_accepts_dict_string_and_garbage() -> None:
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments({"phone": "138"}) == {"phone": "138"}
    assert parse_tool_arguments(' {"phone": "138"} ') == {"phone": "138"}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments("not-json") == {"_raw": "not-json"}
    assert parse_tool_arguments("[1]") == {"_raw": "[1]"}
    assert parse_tool_arguments(5) == {"_raw": "5"}


def test_parse_tool_calls_skips_incomplete_entries() -> None:
    calls = parse_tool_calls(
        [
            "bad",
            {"function": {"name": ""}},
            {"name": "bare_tool", "arguments": {"x": 1}},
            {
                "id": "call_1",
                "function": {"name": "query_appointment", "arguments": '{"phone":"138"}'},
            },
            {"function": {"name": "no_args"}, "arguments": {"from": "raw"}},
        ]
    )
    assert calls[2].name == "no_args"
    assert calls[2].arguments == {"from": "raw"}
    assert [call.name for call in calls] == ["bare_tool", "query_appointment", "no_args"]
    assert calls[0].arguments == {"x": 1}
    assert calls[1].id == "call_1"
    assert calls[1].arguments == {"phone": "138"}
    assert parse_tool_calls({"nope": True}) == []


def test_generated_tool_call_id_is_present() -> None:
    calls = parse_tool_calls([{"function": {"name": "record_intent", "arguments": {}}}])
    assert calls[0].id.startswith("call_")


def test_strip_think_removes_closed_and_unclosed_blocks() -> None:
    assert strip_think("你好") == "你好"
    assert strip_think("先<think>隐藏</think>后") == "先后"
    assert strip_think("甲<think>1</think>乙<think>2</think>丙") == "甲乙丙"
    assert strip_think("<think>只有思考") == ""
    assert strip_think("<think>abc</thi") == ""
    assert strip_think("尾部<thi") == "尾部<thi"


def test_strip_think_holds_partial_tag_across_chunks() -> None:
    stripper = ThinkStripper()
    assert stripper.feed("<thi") == ""
    assert stripper.feed("nk>隐藏</thi") == ""
    assert stripper.feed("nk>可见") == "可见"
    assert stripper.finish() == ""


def test_normalize_and_optional_int() -> None:
    assert normalize_content(None) == ""
    assert normalize_content("你好") == "你好"
    assert normalize_content(1) == "1"
    assert optional_int(True) is None
    assert optional_int(None) is None
    assert optional_int(3) == 3
    assert optional_int("3") is None


def test_status_errors_mark_retryable() -> None:
    denied = error_from_status(400, b'{"error":"bad request"}')
    assert denied.retryable is False
    assert denied.status_code == 400
    limited = error_from_status(429, "slow down")
    assert limited.retryable is True
    unavailable = error_from_status(503, "x" * 500)
    assert unavailable.retryable is True
    assert len(str(unavailable)) < 400


def test_httpx_errors_are_retryable() -> None:
    timeout = error_from_httpx(httpx.ReadTimeout("slow"))
    assert isinstance(timeout, LLMError)
    assert timeout.retryable is True
    assert "超时" in str(timeout)
    refused = error_from_httpx(httpx.ConnectError("refused"))
    assert "连接失败" in str(refused)

    class OddError(httpx.HTTPError):
        pass

    other = error_from_httpx(OddError("odd"))
    assert "请求失败" in str(other)


async def test_collect_stream_joins_text_and_tools() -> None:
    async def chunks():
        yield ChatChunk(content="你", model="m")
        yield ChatChunk(content="好", model="m")
        yield ChatChunk(
            done=True,
            completion_tokens=7,
            tool_calls=[ToolCall(id="call_1", name="query_appointment", arguments={})],
        )

    result = await collect_stream(chunks())
    assert result.content == "你好"
    assert result.model == "m"
    assert result.completion_tokens == 7
    assert result.tool_calls[0].name == "query_appointment"


async def test_iter_visible_chunks_keeps_completion_tokens() -> None:
    async def chunks():
        yield ChatChunk(content="<think>隐藏</think>你", model="m")
        yield ChatChunk(content="好", done=True, model="m", completion_tokens=7)

    seen = [chunk async for chunk in iter_visible_chunks(chunks())]
    assert "".join(chunk.content for chunk in seen) == "你好"
    assert seen[-1].done is True
    assert seen[-1].completion_tokens == 7


async def test_close_iterator_ignores_plain_objects_and_closed_generators() -> None:
    await close_iterator(iter([1]))

    async def broken():
        raise RuntimeError("已经结束")
        yield ChatChunk()

    stream = broken()
    with pytest.raises(RuntimeError):
        await stream.__anext__()
    await close_iterator(stream)


class _Quiet(LLMProvider):
    name = "quiet"

    async def chat(
        self,
        messages: list[Message],
        tools=None,
        stream: bool = False,
    ) -> ChatResult:
        return ChatResult(content="ok")


async def test_default_close_is_noop() -> None:
    await _Quiet().aclose()


async def test_close_iterator_swallows_runtime_error() -> None:
    class Boom:
        async def aclose(self) -> None:
            raise RuntimeError("生成器已在运行")

    await close_iterator(Boom())
