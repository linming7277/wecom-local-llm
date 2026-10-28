"""Ollama 原生 /api/chat。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

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
    normalize_content,
    optional_int,
    parse_tool_calls,
    reject_stream_tools,
    tool_spec_payload,
)


class OllamaProvider(LLMProvider):
    """调用本机或远程 Ollama。num_ctx、temperature、keep_alive 放进请求体。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        num_ctx: int,
        temperature: float,
        keep_alive: str,
        timeout: float,
        think: bool = False,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = "ollama"
        self.model = ensure_model_name(model)
        self.base_url = require_http_url(base_url)
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive.strip()
        self.timeout = timeout
        self.think = think
        self.api_key = api_key
        self._client = client
        self._owns_client = client is None

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # 直连配置里的地址。trust_env 打开时，本机 Ollama 可能被 HTTP 代理拦住。
            self._client = httpx.AsyncClient(timeout=self.timeout, trust_env=False)
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
            "messages": [_message_to_ollama(message) for message in messages],
            "stream": stream,
            "think": self.think,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if tools:
            payload["tools"] = [tool_spec_payload(tool) for tool in tools]
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
        if data.get("error") and "message" not in data:
            raise LLMError(f"模型接口返回错误：{data['error']}", retryable=False)
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise LLMError("模型返回了无法解析的内容", retryable=True)
        content = normalize_content(message.get("content"))
        tool_calls = parse_tool_calls(message.get("tool_calls"))
        if not content.strip() and not tool_calls:
            raise LLMError("模型返回空响应", retryable=True)
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            model=str(data.get("model") or self.model),
            provider=self.name,
            prompt_tokens=optional_int(data.get("prompt_eval_count")),
            completion_tokens=optional_int(data.get("eval_count")),
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
                pending: list[ToolCall] = []
                model_name = self.model
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError("模型返回了无法解析的流式内容", retryable=True) from exc
                    if not isinstance(data, dict):
                        raise LLMError("模型返回了无法解析的流式内容", retryable=True)
                    if data.get("error") and "message" not in data:
                        raise LLMError(
                            f"模型接口返回错误：{data['error']}",
                            retryable=False,
                        )
                    if data.get("model"):
                        model_name = str(data["model"])
                    message = data.get("message") or {}
                    content = ""
                    if isinstance(message, dict):
                        content = normalize_content(message.get("content"))
                        if message.get("tool_calls"):
                            pending = parse_tool_calls(message.get("tool_calls"))
                    if content:
                        emitted = True
                        yield ChatChunk(content=content, done=False, model=model_name)
                    if data.get("done"):
                        completion_tokens = optional_int(data.get("eval_count"))
                        if pending:
                            emitted = True
                            yield ChatChunk(
                                content="",
                                done=True,
                                model=model_name,
                                tool_calls=pending,
                                completion_tokens=completion_tokens,
                            )
                        elif emitted:
                            yield ChatChunk(
                                content="",
                                done=True,
                                model=model_name,
                                completion_tokens=completion_tokens,
                            )
                        if not emitted:
                            raise LLMError("模型返回空响应", retryable=True)
                        return
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


def _message_to_ollama(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool" and message.name:
        payload["tool_name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return payload
