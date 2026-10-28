"""本机 Ollama 压测。

用法（在仓库根目录）：

    python -m bench.bench_ollama --num-ctx 4096 --concurrency 1

脚本不会拉取模型。未安装的标签跳过并写进报告。这个入口不进 CI。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from bench.metrics import (
    PARALLEL_WARNING,
    BenchConfig,
    StreamFold,
    build_sample,
    clean_ollama_version,
    config_public,
    fold_chat_event,
    format_parallel,
    load_bench_config,
    ns_to_seconds,
    override_config,
    parse_ollama_ps,
    prefix_cache_bust,
    recommend,
    render_report,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def chat_payload(
    config: BenchConfig,
    *,
    model: str,
    prompt: str,
    num_ctx: int,
    think: bool,
) -> dict[str, Any]:
    messages = prefix_cache_bust([{"role": "user", "content": prompt}], config.bust_cache)
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": config.keep_alive,
        "options": {
            "temperature": config.temperature,
            "num_ctx": num_ctx,
            "seed": config.seed,
            "num_predict": config.num_predict,
        },
    }


async def fetch_installed_models(client: httpx.AsyncClient, base_url: str) -> set[str]:
    response = await client.get(f"{base_url}/api/tags", timeout=30)
    response.raise_for_status()
    payload = response.json()
    names: set[str] = set()
    models = payload.get("models") if isinstance(payload, dict) else None
    for item in models or []:
        if not isinstance(item, dict):
            continue
        if item.get("name"):
            names.add(str(item["name"]))
        if item.get("model"):
            names.add(str(item["model"]))
    return names


async def run_one(
    client: httpx.AsyncClient,
    config: BenchConfig,
    *,
    model: str,
    prompt_name: str,
    prompt: str,
    num_ctx: int,
    concurrency: int,
    think: bool,
    repeat_index: int,
    excluded: bool,
) -> dict[str, Any]:
    meta = {
        "model": model,
        "prompt": prompt_name,
        "num_ctx": num_ctx,
        "concurrency": concurrency,
        "think": think,
        "repeat_index": repeat_index,
        "num_predict": config.num_predict,
    }
    started = time.perf_counter()
    state = StreamFold()
    error: str | None = None
    try:
        payload = chat_payload(
            config,
            model=model,
            prompt=prompt,
            num_ctx=num_ctx,
            think=think,
        )
        async with client.stream(
            "POST",
            f"{config.base_url}/api/chat",
            json=payload,
            timeout=config.timeout,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error = f"HTTP {response.status_code}：{' '.join(body.split())[:300]}"
            else:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        error = "模型返回了无法解析的流式内容"
                        break
                    fold_chat_event(state, data, time.perf_counter() - started)
                    if state.error:
                        break
    except httpx.HTTPError as exc:
        error = f"连接失败：{exc}"
    return build_sample(
        meta=meta,
        state=state,
        total_s=time.perf_counter() - started,
        error=error or state.error,
        excluded=excluded,
    )


def command_output(args: list[str]) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return (completed.stdout or "").strip()


def read_ollama_ps() -> str:
    completed = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return completed.stderr or completed.stdout
    return completed.stdout


def chip_from_profiler(text: str) -> str:
    for line in text.splitlines():
        if "Chip:" in line or "芯片" in line:
            return line.split(":", 1)[-1].split("：", 1)[-1].strip()
    return ""


def format_memory(raw: str) -> str:
    try:
        size = int(raw)
    except ValueError:
        return raw or "未知"
    gib = size / 1024**3
    if abs(gib - round(gib)) < 0.05:
        return f"{round(gib)} GB"
    return f"{gib:.1f} GB"


def machine_info(run: Callable[[list[str]], str] | None = None) -> dict[str, str]:
    runner = run or command_output
    brand = runner(["sysctl", "-n", "machdep.cpu.brand_string"])
    chip = brand
    if not chip:
        chip = chip_from_profiler(runner(["system_profiler", "SPHardwareDataType"]))
    if not chip:
        chip = runner(["sysctl", "-n", "hw.model"]) or "未知芯片"
    return {
        "chip": chip,
        "memory": format_memory(runner(["sysctl", "-n", "hw.memsize"])),
        "macos": runner(["sw_vers", "-productVersion"]) or "未知",
        "ollama": clean_ollama_version(runner(["ollama", "--version"]) or "") or "未知",
        "arch": platform.machine(),
    }


def _remember_ps(found: dict[str, dict[str, str]], text: str) -> None:
    for row in parse_ollama_ps(text):
        found[row["name"]] = row


async def run_bench(
    config: BenchConfig,
    *,
    client: httpx.AsyncClient,
    ps_provider: Callable[[], str],
    parallel_value: str | None,
    machine: dict[str, str],
) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    list_error: str | None = None
    installed: set[str] = set()
    try:
        installed = await fetch_installed_models(client, config.base_url)
    except httpx.HTTPError as exc:
        list_error = f"无法读取本机模型列表：{exc}"
    skipped: list[str] = []
    planned: list[tuple[str, list[int], list[int]]] = []
    if list_error is None:
        for model in config.models:
            if model in installed:
                planned.append((model, list(config.num_ctx), list(config.concurrency)))
            else:
                skipped.append(model)
                print(f"跳过未拉取的模型：{model}", file=sys.stderr)
        if config.include_heavy:
            for model in config.heavy_models:
                if model in installed:
                    planned.append((model, [4096], [1]))
                else:
                    skipped.append(model)
                    print(f"跳过未拉取的模型：{model}", file=sys.stderr)
    if any(item > 1 for item in config.concurrency):
        print(
            f"提示：{PARALLEL_WARNING}。当前值：{format_parallel(parallel_value)}",
            file=sys.stderr,
        )
    warmup: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []
    ps_found: dict[str, dict[str, str]] = {}
    for model, contexts, concurrencies in planned:
        prompt_name = "short" if "short" in config.prompts else next(iter(config.prompts))
        warm = await run_one(
            client,
            config,
            model=model,
            prompt_name=prompt_name,
            prompt=config.prompts[prompt_name],
            num_ctx=contexts[0],
            concurrency=1,
            think=False,
            repeat_index=0,
            excluded=True,
        )
        warmup.append(warm)
        loads.append(
            {
                "model": model,
                "num_ctx": contexts[0],
                "load_duration_ns": warm.get("load_duration_ns"),
                "load_duration_s": ns_to_seconds(warm.get("load_duration_ns")),
                "note": "预热请求的 load_duration，不计入正式结果",
            }
        )
        for name, prompt in config.prompts.items():
            for num_ctx in contexts:
                for concurrency in concurrencies:
                    for think in config.think:
                        for repeat in range(config.repeats):
                            print(
                                (
                                    f"{model} prompt={name} num_ctx={num_ctx} "
                                    f"concurrency={concurrency} think={think} "
                                    f"repeat={repeat + 1}/{config.repeats}"
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                            batch = await asyncio.gather(
                                *[
                                    run_one(
                                        client,
                                        config,
                                        model=model,
                                        prompt_name=name,
                                        prompt=prompt,
                                        num_ctx=num_ctx,
                                        concurrency=concurrency,
                                        think=think,
                                        repeat_index=repeat * concurrency + slot,
                                        excluded=False,
                                    )
                                    for slot in range(concurrency)
                                ]
                            )
                            samples.extend(batch)
        _remember_ps(ps_found, ps_provider())
    summary = summarize(samples)
    document: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": machine,
        "ollama_num_parallel": parallel_value,
        "error": list_error,
        "config": config_public(config),
        "skipped_models": skipped,
        "loads": loads,
        "ollama_ps": list(ps_found.values()),
        "warmup": warmup,
        "samples": samples,
        "summary": summary,
    }
    document["conclusion"] = recommend(machine, summary, skipped)
    return document


def write_outputs(
    document: dict[str, Any],
    results_dir: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    json_path = results_dir / f"{stamp}.json"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(document), encoding="utf-8")
    return json_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="压测本机 Ollama 模型")
    parser.add_argument("--config", type=Path, default=ROOT / "bench" / "config.yaml")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "bench" / "prompts")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "bench" / "results")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "bench_report.md")
    parser.add_argument("--models", action="append", default=None, help="覆盖配置里的模型，可重复")
    parser.add_argument("--num-ctx", dest="num_ctx", action="append", type=int, default=None)
    parser.add_argument("--concurrency", action="append", type=int, default=None)
    parser.add_argument("--think", action="append", choices=("true", "false"), default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="同时跑 heavy_models，只使用 num_ctx=4096、并发 1",
    )
    return parser


def _split_args(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def config_from_args(args: argparse.Namespace) -> BenchConfig:
    config = load_bench_config(args.config, args.prompts_dir)
    think_values = _split_args(args.think)
    think = None if think_values is None else [item == "true" for item in think_values]
    num_ctx = None if args.num_ctx is None else [int(item) for item in args.num_ctx]
    concurrency = None if args.concurrency is None else [int(item) for item in args.concurrency]
    return override_config(
        config,
        models=_split_args(args.models),
        num_ctx=num_ctx,
        concurrency=concurrency,
        think=think,
        repeats=args.repeats,
        include_heavy=bool(getattr(args, "include_heavy", False)),
    )


async def _execute(config: BenchConfig) -> dict[str, Any]:
    async with httpx.AsyncClient(trust_env=False, timeout=config.timeout) as client:
        return await run_bench(
            config,
            client=client,
            ps_provider=read_ollama_ps,
            parallel_value=os.environ.get("OLLAMA_NUM_PARALLEL"),
            machine=machine_info(),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    document = asyncio.run(_execute(config))
    json_path, report_path = write_outputs(document, args.results_dir, args.report)
    print(json_path)
    print(report_path)
    print(document.get("conclusion") or "")
    if document.get("error"):
        return 1
    if not document.get("samples"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
