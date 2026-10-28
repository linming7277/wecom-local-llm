"""命令行调用当前配置的模型。

用法：python -m app.llm.cli "你的问题"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from app.config import get_settings
from app.llm.base import LLMError, Message
from app.llm.router import build_router
from app.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调用 .env 里配置的对话模型")
    parser.add_argument("prompt", help="发给模型的用户问题")
    parser.add_argument("--model", help="临时覆盖 LLM_MODEL，不写回 .env")
    parser.add_argument("--stream", action="store_true", help="按流式输出回答")
    return parser


def format_run_stats(
    *,
    model: str,
    elapsed_s: float,
    ttft_s: float | None,
    output_tokens: int | None,
) -> str:
    """命令行结尾的统计。没有首 token 时间时不打印这一行。"""
    lines = [
        f"模型：{model}",
        f"总耗时：{elapsed_s:.3f}s",
    ]
    if ttft_s is not None:
        lines.append(f"首 token 耗时：{ttft_s:.3f}s")
    if output_tokens is None:
        lines.append("输出 token 数：未知")
    else:
        lines.append(f"输出 token 数：{output_tokens}")
    return "\n".join(lines)


async def run_chat(prompt: str, *, model: str | None, stream: bool) -> int:
    settings = get_settings()
    if model:
        settings = settings.model_copy(update={"llm_model": model})
    setup_logging()
    router = build_router(settings)
    messages = [Message(role="user", content=prompt)]
    try:
        started = time.perf_counter()
        model_name = settings.llm_model
        output_tokens: int | None = None
        ttft_s: float | None = None
        if stream:
            chunks = await router.chat(messages, stream=True)
            async for chunk in chunks:
                if chunk.model:
                    model_name = chunk.model
                if chunk.completion_tokens is not None:
                    output_tokens = chunk.completion_tokens
                if chunk.content:
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - started
                    print(chunk.content, end="", flush=True)
            print()
        else:
            result = await router.chat(messages, stream=False)
            if result.model:
                model_name = result.model
            output_tokens = result.completion_tokens
            print(result.content)
        elapsed_s = time.perf_counter() - started
        print()
        print(
            format_run_stats(
                model=model_name,
                elapsed_s=elapsed_s,
                ttft_s=ttft_s,
                output_tokens=output_tokens,
            )
        )
        return 0
    except LLMError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        return 1
    finally:
        await router.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_chat(args.prompt, model=args.model, stream=args.stream))
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130


def console_main() -> None:
    """安装后的 wecom-llm 入口。退出码来自 main。"""
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
