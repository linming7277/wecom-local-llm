"""OpenAI 兼容的 /chat/completions。

标准兼容接口提交 model 和 temperature。num_ctx 与 keep_alive 是 Ollama 原生参数，
放在本提供方的实例上供网关读取，不放进 JSON，避免智谱等服务返回 400。
timeout 作用在 HTTP 客户端上。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import require_http_url
from app.llm.base import (
    ChatChunk,
    ChatResult,
    LLMError,
    LLMProvider,
    Message,
    ToolCall,
    ToolSpec,
    ensure_model_name,
    error_from_httpx,
    error_from_status,
    new_tool_call_id,
    normalize_content,
    optional_int,
    parse_tool_arguments,
    reject_stream_tools,
    tool_spec_payload,
)


def chat_completions_url(base_url: str) -> str:
    """把配置里的根地址补成 chat/completions。

    根路径会补上 /v1。已经带路径的地址保持不变，例如智谱的 /api/paas/v4。
    """
    base = require_http_url(base_url)
    parts = urlsplit(base)
    if parts.path in ("", "/"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


@dataclass
class _PartialToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


class OpenAICompatProvider(LLMProvider):
    """调用 OpenAI 兼容接口，例如智谱 GLM 或 Ollama 的 /v1。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        num_ctx: int,
        temperature: float,
        keep_alive: str,
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = "openai"
        self.model = ensure_model_name(model)
        self.base_url = require_http_url(base_url)
        self.chat_url = chat_completions_url(base_url)
        self.api_key = api_key
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive.strip()
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, trust_env=True)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(message) for message in messages],
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [tool_spec_payload(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        reject_stream_tools(tools, stream)
        if stream:
            return self._stream(messages, tools)
        return await self._complete(messages, tools)

    async def _complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
    ) -> ChatResult:
        response = await self._post(self._payload(messages, tools, stream=False))
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("模型返回了无法解析的内容", retryable=True) from exc
        if not isinstance(data, dict):
            raise LLMError("模型返回了无法解析的内容", retryable=True)
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise LLMError("模型返回空响应", retryable=True)
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise LLMError("模型返回了无法解析的内容", retryable=True)
        content = normalize_content(message.get("content"))
        tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
        if not content.strip() and not tool_calls:
            raise LLMError("模型返回空响应", retryable=True)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            model=str(data.get("model") or self.model),
            provider=self.name,
            prompt_tokens=optional_int(usage.get("prompt_tokens")),
            completion_tokens=optional_int(usage.get("completion_tokens")),
        )

    async def _stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
    ) -> AsyncIterator[ChatChunk]:
        payload = self._payload(messages, tools, stream=True)
        client = self._http()
        try:
            async with client.stream(
                "POST",
                self.chat_url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise error_from_status(response.status_code, body)
                emitted = False
                partials: dict[int, _PartialToolCall] = {}
                model_name = self.model
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None:
                        continue
                    if event == "[DONE]":
                        break
                    try:
                        data = json.loads(event)
                    except json.JSONDecodeError as exc:
                        raise LLMError("模型返回了无法解析的流式内容", retryable=True) from exc
                    if not isinstance(data, dict):
                        raise LLMError("模型返回了无法解析的流式内容", retryable=True)
                    if data.get("model"):
                        model_name = str(data["model"])
                    choices = data.get("choices") or []
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    content = normalize_content(delta.get("content"))
                    if content:
                        emitted = True
                        yield ChatChunk(content=content, done=False, model=model_name)
                    if delta.get("tool_calls"):
                        _apply_tool_delta(partials, delta["tool_calls"])
                tool_calls = _finish_partials(partials)
                if tool_calls:
                    emitted = True
                    yield ChatChunk(
                        content="",
                        done=True,
                        model=model_name,
                        tool_calls=tool_calls,
                    )
                elif emitted:
                    yield ChatChunk(content="", done=True, model=model_name)
                if not emitted:
                    raise LLMError("模型返回空响应", retryable=True)
        except LLMError:
            raise
        except httpx.HTTPError as exc:
            raise error_from_httpx(exc) from exc

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._http().post(
                self.chat_url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise error_from_httpx(exc) from exc
        if response.status_code >= 400:
            raise error_from_status(response.status_code, response.text)
        return response


def _message_to_openai(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_openai_tool_calls(raw_calls: Any) -> list[ToolCall]:
    if not isinstance(raw_calls, list):
        return []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = str(function.get("name") or "")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=str(raw.get("id") or new_tool_call_id()),
                name=name,
                arguments=parse_tool_arguments(function.get("arguments")),
            )
        )
    return calls


def _parse_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _apply_tool_delta(partials: dict[int, _PartialToolCall], deltas: Any) -> None:
    if not isinstance(deltas, list):
        return
    for item in deltas:
        if not isinstance(item, dict):
            continue
        index = item.get("index", 0)
        if not isinstance(index, int):
            continue
        current = partials.setdefault(index, _PartialToolCall())
        if item.get("id"):
            current.id = str(item["id"])
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            current.name = name if not current.name else current.name + name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            current.arguments += arguments


def _finish_partials(partials: dict[int, _PartialToolCall]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(partials):
        partial = partials[index]
        if not partial.name:
            continue
        calls.append(
            ToolCall(
                id=partial.id or new_tool_call_id(),
                name=partial.name,
                arguments=parse_tool_arguments(partial.arguments),
            )
        )
    return calls
