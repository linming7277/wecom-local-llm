"""模型网关的公共类型。

stream=False 时，chat 返回 ChatResult。
stream=True 时，chat 返回异步迭代器，调用方先 await，再 async for。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx


class LLMError(Exception):
    """模型调用失败。retryable 为真时，路由可以切换到备用模型。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class ChatChunk:
    content: str = ""
    done: bool = False
    model: str = ""
    tool_calls: list[ToolCall] | None = None
    completion_tokens: int | None = None


class LLMProvider(ABC):
    """一个可调用的对话模型。"""

    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        """发送一轮对话。

        tools 是 JSON Schema 形式的工具列表。模型不支持工具时可以忽略。
        """

    async def aclose(self) -> None:
        """释放连接。默认没有可关闭的资源，持有 HTTP 客户端的子类覆盖它。"""
        return None


def new_tool_call_id() -> str:
    return f"call_{uuid4().hex[:16]}"


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """把工具参数统一成字典。无法解析时保留原文，避免丢信息。"""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}
        if isinstance(value, dict):
            return value
        return {"_raw": text}
    return {"_raw": str(raw)}


def parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    if not isinstance(raw_calls, list):
        return []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        source = function if isinstance(function, dict) else raw
        name = str(source.get("name") or "")
        if not name:
            continue
        arguments = source.get("arguments")
        if arguments is None:
            arguments = raw.get("arguments")
        calls.append(
            ToolCall(
                id=str(raw.get("id") or new_tool_call_id()),
                name=name,
                arguments=parse_tool_arguments(arguments),
            )
        )
    return calls


def tool_spec_payload(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def error_from_status(status_code: int, body: bytes | str) -> LLMError:
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = body
    compact = " ".join(text.split())[:300]
    retryable = status_code == 429 or status_code >= 500
    return LLMError(
        f"模型接口返回 {status_code}：{compact}",
        status_code=status_code,
        retryable=retryable,
    )


def error_from_httpx(exc: httpx.HTTPError) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMError(f"模型接口超时：{exc}", retryable=True)
    if isinstance(exc, httpx.TransportError):
        return LLMError(f"模型接口连接失败：{exc}", retryable=True)
    return LLMError(f"模型接口请求失败：{exc}", retryable=True)


def ensure_model_name(model: str) -> str:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError("模型名不能为空")
    return cleaned


async def collect_stream(chunks: AsyncIterator[ChatChunk]) -> ChatResult:
    """把流式片段拼成完整结果。压测和测试用它取首段之后的全文。"""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    model = ""
    completion_tokens: int | None = None
    async for chunk in chunks:
        if chunk.model:
            model = chunk.model
        if chunk.content:
            texts.append(chunk.content)
        if chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
        if chunk.completion_tokens is not None:
            completion_tokens = chunk.completion_tokens
    return ChatResult(
        content="".join(texts),
        tool_calls=tool_calls,
        model=model,
        completion_tokens=completion_tokens,
    )


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def reject_stream_tools(tools: list[ToolSpec] | None, stream: bool) -> None:
    """流式输出还没有工具调用增量解析，调用方应改用非流式。"""
    if stream and tools:
        raise NotImplementedError("流式输出暂不支持同时传入工具")


def _tag_prefix_suffix(text: str, tag: str) -> str:
    """保留末尾可能是标签前缀的那一段，避免标签被拆到两个片段后漏掉。"""
    limit = min(len(text), len(tag) - 1)
    for size in range(limit, 0, -1):
        if tag.startswith(text[-size:]):
            return text[-size:]
    return ""


class ThinkStripper:
    """去掉思考标签。未闭合的 <think> 会一直吞到结束。"""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        visible: list[str] = []
        while True:
            if self._inside:
                close_at = self._buf.find(_THINK_CLOSE)
                if close_at == -1:
                    if final:
                        self._buf = ""
                        self._inside = False
                    else:
                        self._buf = _tag_prefix_suffix(self._buf, _THINK_CLOSE)
                    break
                self._buf = self._buf[close_at + len(_THINK_CLOSE) :]
                self._inside = False
                continue
            open_at = self._buf.find(_THINK_OPEN)
            if open_at == -1:
                if final:
                    visible.append(self._buf)
                    self._buf = ""
                else:
                    keep = _tag_prefix_suffix(self._buf, _THINK_OPEN)
                    visible.append(self._buf[: len(self._buf) - len(keep)])
                    self._buf = keep
                break
            visible.append(self._buf[:open_at])
            self._buf = self._buf[open_at + len(_THINK_OPEN) :]
            self._inside = True
        return "".join(visible)


def strip_think(text: str) -> str:
    """去掉完整思考块，以及一直到结尾都没有闭合的思考块。"""
    stripper = ThinkStripper()
    return stripper.feed(text) + stripper.finish()


async def iter_visible_chunks(stream: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
    """流式片段先去掉思考标签，再交给调用方。全程没有可见内容时视为空响应。"""
    stripper = ThinkStripper()
    emitted = False
    model = ""
    completion_tokens: int | None = None
    async for chunk in stream:
        if chunk.model:
            model = chunk.model
        if chunk.completion_tokens is not None:
            completion_tokens = chunk.completion_tokens
        visible = stripper.feed(chunk.content) if chunk.content else ""
        if visible:
            emitted = True
            yield ChatChunk(content=visible, done=False, model=model)
        if chunk.tool_calls:
            emitted = True
            yield ChatChunk(content="", done=False, model=model, tool_calls=chunk.tool_calls)
        if chunk.done:
            tail = stripper.finish()
            if tail:
                emitted = True
                yield ChatChunk(content=tail, done=False, model=model)
            if not emitted:
                raise LLMError("模型返回空响应", retryable=True)
            yield ChatChunk(
                content="",
                done=True,
                model=model,
                completion_tokens=completion_tokens,
            )
            return
    tail = stripper.finish()
    if tail:
        emitted = True
        yield ChatChunk(content=tail, done=False, model=model)
    if not emitted:
        raise LLMError("模型返回空响应", retryable=True)
    yield ChatChunk(content="", done=True, model=model, completion_tokens=completion_tokens)


async def close_iterator(stream: AsyncIterator[object]) -> None:
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except RuntimeError:
        # 生成器已经在抛错时结束，再关闭会碰到「已在运行」的状态。
        return
