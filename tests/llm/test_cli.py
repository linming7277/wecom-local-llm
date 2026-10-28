"""命令行入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from app.config import Settings
from app.llm.base import ChatChunk, ChatResult, LLMError, Message
from app.llm.cli import console_main, main


class FakeRouter:
    def __init__(self, *, content: str = "本地回答", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.closed = False
        self.stream = False
        self.messages: list[Message] | None = None

    async def chat(
        self,
        messages: list[Message],
        tools=None,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[ChatChunk]:
        self.messages = messages
        self.stream = stream
        if self.error is not None:
            raise self.error
        if stream:

            async def generate() -> AsyncIterator[ChatChunk]:
                yield ChatChunk(content="流", model="qwen3:4b")
                yield ChatChunk(content="式", done=True, model="qwen3:4b", completion_tokens=2)

            return generate()
        return ChatResult(
            content=self.content,
            model="qwen3.8:27b",
            provider="ollama",
            completion_tokens=12,
        )

    async def aclose(self) -> None:
        self.closed = True


def test_cli_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    router = FakeRouter()

    def factory(settings: Settings) -> FakeRouter:
        assert settings.llm_model == "qwen3.8:27b"
        return router

    monkeypatch.setattr("app.llm.cli.build_router", factory)
    monkeypatch.setattr("app.llm.cli.time.perf_counter", iter((10.0, 11.25)).__next__)
    assert main(["用一句话介绍你自己"]) == 0
    assert capsys.readouterr().out == (
        "本地回答\n\n模型：qwen3.8:27b\n总耗时：1.250s\n输出 token 数：12\n"
    )
    assert router.closed is True
    assert router.messages is not None
    assert router.messages[0].content == "用一句话介绍你自己"


def test_cli_model_override_and_stream(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}
    router = FakeRouter()

    def factory(settings: Settings) -> FakeRouter:
        seen["model"] = settings.llm_model
        return router

    monkeypatch.setattr("app.llm.cli.build_router", factory)
    monkeypatch.setattr("app.llm.cli.time.perf_counter", iter((10.0, 10.2, 10.5)).__next__)
    assert main(["你好", "--model", "qwen3:4b", "--stream"]) == 0
    assert seen["model"] == "qwen3:4b"
    assert router.stream is True
    assert capsys.readouterr().out == (
        "流式\n\n模型：qwen3:4b\n总耗时：0.500s\n首 token 耗时：0.200s\n输出 token 数：2\n"
    )


def test_cli_reports_model_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    router = FakeRouter(error=LLMError("模型接口连接失败：refused", retryable=True))
    monkeypatch.setattr("app.llm.cli.build_router", lambda _settings: router)
    assert main(["你好"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "模型调用失败" in captured.err
    assert router.closed is True


def test_cli_requires_prompt() -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_cli_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_interrupt(coro: object) -> int:
        close = getattr(coro, "close", None)
        if close is not None:
            close()
        raise KeyboardInterrupt

    monkeypatch.setattr("app.llm.cli.asyncio.run", raise_interrupt)
    assert main(["你好"]) == 130


def test_console_main_uses_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.cli.main", lambda: 0)
    with pytest.raises(SystemExit) as caught:
        console_main()
    assert caught.value.code == 0
