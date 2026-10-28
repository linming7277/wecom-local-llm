"""压测运行器用模拟的 Ollama 响应。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from bench.bench_ollama import (
    chip_from_profiler,
    command_output,
    config_from_args,
    format_memory,
    machine_info,
    main,
    read_ollama_ps,
    run_bench,
)
from bench.metrics import BenchConfig, config_public, render_report


def _config(**updates: object) -> BenchConfig:
    base = BenchConfig(
        models=["demo:q4_K_M"],
        num_ctx=[4096],
        concurrency=[1],
        think=[True],
        repeats=1,
        temperature=0,
        seed=42,
        base_url="http://ollama.test",
        timeout=5,
        num_predict=16,
        prompts={"short": "问题"},
        keep_alive="10m",
    )
    data = base.__dict__.copy()
    data.update(updates)
    return BenchConfig(**data)


def _ndjson(events: list[dict]) -> str:
    return "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)


def _ok_events() -> list[dict]:
    return [
        {"message": {"content": "<thi"}, "done": False},
        {"message": {"content": "nk>隐藏</think>你"}, "done": False},
        {"message": {"thinking": "字段"}, "done": False},
        {"message": {"content": "好"}, "done": False},
        {
            "model": "demo:q4_K_M",
            "message": {"content": ""},
            "done": True,
            "eval_count": 8,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 4,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 3_000_000_000,
        },
    ]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)


async def test_run_bench_excludes_warmup_and_checks_think_filter() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "demo:q4_K_M"}, "skip-me"]},
            )
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text="\n" + _ndjson(_ok_events()))

    async with _client(handler) as client:
        document = await run_bench(
            _config(),
            client=client,
            ps_provider=lambda: (
                f"{'NAME':<20}{'ID':<8}{'SIZE':<12}{'PROCESSOR':<12}\n"
                f"{'demo:q4_K_M':<20}{'abc':<8}{'3.8 GB':<12}{'100% GPU':<12}\n"
            ),
            parallel_value=None,
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )

    assert bodies[0]["think"] is False
    assert bodies[1]["think"] is True
    assert bodies[1]["options"]["temperature"] == 0
    assert bodies[1]["options"]["seed"] == 42
    assert bodies[1]["options"]["num_ctx"] == 4096
    assert bodies[1]["stream"] is True
    first = bodies[0]["messages"][0]["content"]
    second = bodies[1]["messages"][0]["content"]
    assert first.startswith("[req:")
    assert second.startswith("[req:")
    assert first != second
    assert first.endswith("\n问题")
    assert len(document["warmup"]) == 1
    assert document["warmup"][0]["excluded"] is True
    assert document["loads"][0]["load_duration_ns"] == 3_000_000_000
    assert len(document["samples"]) == 1
    sample = document["samples"][0]
    assert sample["ok"] is True
    assert sample["excluded"] is False
    assert sample["gen_tokens_per_s"] == 4
    assert sample["prefill_tokens_per_s"] == 8
    assert sample["visible_preview"] == "你好"
    assert sample["gateway_stripped"] is True
    assert sample["raw_has_thinking_field"] is True
    assert document["summary"][0]["count"] == 1
    assert document["summary"][0]["fail_rate"] == 0
    assert document["ollama_ps"][0]["size"] == "3.8 GB"
    assert document["ollama_ps"][0]["processor"] == "100% GPU"
    assert document["samples"][0]["first_thinking_ms"] is not None
    assert document["samples"][0]["num_predict"] == 16
    assert document["conclusion"] == "在 [Apple M3 Pro / 18 GB] 上，样本不足，暂不下结论。"


async def test_missing_model_is_skipped_without_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "other:latest"}]})

    async with _client(handler) as client:
        document = await run_bench(
            _config(models=["qwen3.8:27b"]),
            client=client,
            ps_provider=lambda: "",
            parallel_value="1",
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )
    assert document["skipped_models"] == ["qwen3.8:27b"]
    assert document["samples"] == []
    assert document["error"] is None


async def test_heavy_model_stays_out_until_requested(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "demo:q4_K_M"}, {"name": "qwen3.8:27b"}]},
            )
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=_ndjson(_ok_events()))

    common = {
        "models": ["demo:q4_K_M"],
        "heavy_models": ["qwen3.8:27b"],
        "num_ctx": [8192],
        "concurrency": [1],
        "think": [False],
        "repeats": 1,
        "bust_cache": False,
    }
    async with _client(handler) as client:
        quiet = await run_bench(
            _config(**common, include_heavy=False),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )
    assert {body["model"] for body in seen} == {"demo:q4_K_M"}
    assert quiet["skipped_models"] == []
    assert "本次未运行" in render_report(quiet)

    seen.clear()
    async with _client(handler) as client:
        included = await run_bench(
            _config(**common, include_heavy=True),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )
    heavy = [body for body in seen if body["model"] == "qwen3.8:27b"]
    demo = [body for body in seen if body["model"] == "demo:q4_K_M"]
    assert [body["options"]["num_ctx"] for body in heavy] == [4096, 4096]
    assert [body["options"]["num_ctx"] for body in demo] == [8192, 8192]
    assert included["skipped_models"] == []
    assert "本次已包含" in render_report(included)

    def missing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "demo:q4_K_M"}]})
        return httpx.Response(200, text=_ndjson(_ok_events()))

    async with _client(missing) as client:
        skipped = await run_bench(
            _config(**common, include_heavy=True),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )
    assert skipped["skipped_models"] == ["qwen3.8:27b"]
    assert "跳过未拉取的模型：qwen3.8:27b" in capsys.readouterr().err


async def test_failed_requests_count_toward_fail_rate(capsys: pytest.CaptureFixture[str]) -> None:
    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"model": "demo:q4_K_M"}]})
        calls["chat"] += 1
        if calls["chat"] <= 2:
            return httpx.Response(200, text=_ndjson(_ok_events()))
        return httpx.Response(500, text="overloaded")

    async with _client(handler) as client:
        document = await run_bench(
            _config(concurrency=[2], think=[False], repeats=1),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "Apple M3 Pro", "memory": "18 GB"},
        )
    assert calls["chat"] == 3
    assert document["summary"][0]["count"] == 2
    assert document["summary"][0]["fail_rate"] == 0.5
    errors = [sample.get("error") or "" for sample in document["samples"]]
    assert any("HTTP 500" in item for item in errors)
    assert "实际并行数" in capsys.readouterr().err


async def test_bad_stream_and_connect_error_become_samples() -> None:
    def bad_json(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "demo:q4_K_M"}]})
        return httpx.Response(200, text="{bad\n")

    async with _client(bad_json) as client:
        parsed = await run_bench(
            _config(think=[False], repeats=1),
            client=client,
            ps_provider=lambda: "",
            parallel_value="",
            machine={"chip": "x", "memory": "1 GB"},
        )
    assert parsed["samples"][0]["ok"] is False
    assert "无法解析" in parsed["samples"][0]["error"]

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "demo:q4_K_M"}]})
        return httpx.Response(200, text='{"error":"missing"}\n')

    async with _client(upstream) as client:
        errored = await run_bench(
            _config(think=[False], repeats=1),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "x", "memory": "1 GB"},
        )
    assert "missing" in errored["samples"][0]["error"]

    def refused(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "demo:q4_K_M"}]})
        raise httpx.ConnectError("refused")

    async with _client(refused) as client:
        failed = await run_bench(
            _config(think=[False], repeats=1),
            client=client,
            ps_provider=lambda: "",
            parallel_value="2",
            machine={"chip": "x", "memory": "1 GB"},
        )
    assert "连接失败" in failed["samples"][0]["error"]


async def test_tags_failure_is_reported() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    async with _client(handler) as client:
        document = await run_bench(
            _config(),
            client=client,
            ps_provider=lambda: "",
            parallel_value=None,
            machine={"chip": "x", "memory": "1 GB"},
        )
    assert document["error"]
    assert "无法读取本机模型列表" in document["error"]
    assert document["samples"] == []


def test_machine_info_uses_profiler_when_brand_is_empty() -> None:
    seen: list[list[str]] = []

    def run(args: list[str]) -> str:
        seen.append(args)
        if args[:2] == ["sysctl", "-n"] and args[-1] == "machdep.cpu.brand_string":
            return ""
        if args == ["system_profiler", "SPHardwareDataType"]:
            return "Chip: Apple M3 Pro\n"
        if args == ["sysctl", "-n", "hw.memsize"]:
            return "19327352832"
        if args == ["sw_vers", "-productVersion"]:
            return "14.4"
        if args == ["ollama", "--version"]:
            return "ollama version is 0.40.11"
        return ""

    info = machine_info(run)
    assert chip_from_profiler("no chip here") == ""
    assert format_memory(str(int(1.5 * 1024**3))) == "1.5 GB"

    def fallback(args: list[str]) -> str:
        if args[-1] == "machdep.cpu.brand_string":
            return ""
        if args[0] == "system_profiler":
            return "Model: only\n"
        if args[-1] == "hw.model":
            return "Mac15,7"
        if args[-1] == "hw.memsize":
            return "19327352832"
        return ""

    assert machine_info(fallback)["chip"] == "Mac15,7"
    assert info["chip"] == "Apple M3 Pro"
    assert info["memory"] == "18 GB"
    assert info["macos"] == "14.4"
    assert info["ollama"] == "0.40.11"
    assert any(command[0] == "system_profiler" for command in seen)
    assert chip_from_profiler("芯片：Apple M3 Pro") == "Apple M3 Pro"


def test_command_helpers_read_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        def __init__(self, code: int, stdout: str, stderr: str = "") -> None:
            self.returncode = code
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args: list[str], **_kwargs: object) -> Completed:
        if args == ["ollama", "ps"]:
            return Completed(1, "", "ps failed")
        return Completed(0, "value\n")

    monkeypatch.setattr("bench.bench_ollama.subprocess.run", fake_run)
    assert command_output(["sysctl", "-n", "hw.model"]) == "value"
    assert read_ollama_ps() == "ps failed"

    def running(args: list[str], **_kwargs: object) -> Completed:
        return Completed(0, "NAME ID\n")

    monkeypatch.setattr("bench.bench_ollama.subprocess.run", running)
    assert read_ollama_ps() == "NAME ID\n"


def test_main_writes_outputs_and_applies_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, BenchConfig] = {}

    async def fake_execute(config: BenchConfig) -> dict:
        seen["config"] = config
        return {
            "error": None,
            "samples": [{"ok": True}],
            "summary": [],
            "skipped_models": [],
            "loads": [],
            "ollama_ps": [],
            "machine": {
                "chip": "Apple M3 Pro",
                "memory": "18 GB",
                "macos": "14.4",
                "ollama": "ollama version is 0.40.11",
                "arch": "arm64",
            },
            "ollama_num_parallel": None,
            "config": config_public(config),
            "conclusion": "在 [Apple M3 Pro / 18 GB] 上，客服场景推荐 [demo]，理由是示例。",
        }

    monkeypatch.setattr("bench.bench_ollama._execute", fake_execute)
    report = tmp_path / "report.md"
    code = main(
        [
            "--models",
            "qwen3:4b-instruct-2507-q4_K_M",
            "--num-ctx",
            "4096",
            "--concurrency",
            "1",
            "--think",
            "false",
            "--repeats",
            "1",
            "--results-dir",
            str(tmp_path / "results"),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    config = seen["config"]
    assert config.models == ["qwen3:4b-instruct-2507-q4_K_M"]
    assert config.num_ctx == [4096]
    assert config.concurrency == [1]
    assert config.think == [False]
    assert config.repeats == 1
    assert config.temperature == 0
    assert config.num_predict == 512
    assert config.bust_cache is True
    assert config.include_heavy is False
    assert config.heavy_models == ["qwen3.8:27b"]
    assert "客服场景推荐" in report.read_text(encoding="utf-8")
    assert list((tmp_path / "results").glob("*.json"))


def test_main_returns_nonzero_when_ollama_cannot_be_listed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_execute(_config: BenchConfig) -> dict:
        return {"error": "无法读取本机模型列表", "samples": [], "conclusion": "失败"}

    monkeypatch.setattr("bench.bench_ollama._execute", fake_execute)
    code = main(
        [
            "--results-dir",
            str(tmp_path / "results"),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 1


def test_main_returns_two_when_every_model_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_execute(_config: BenchConfig) -> dict:
        return {"error": None, "samples": [], "conclusion": "没有样本"}

    monkeypatch.setattr("bench.bench_ollama._execute", fake_execute)
    code = main(
        [
            "--results-dir",
            str(tmp_path / "results"),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 2


def test_config_from_args_splits_commas() -> None:
    args = config_from_args(
        type(
            "Args",
            (),
            {
                "config": Path(__file__).resolve().parents[2] / "bench" / "config.yaml",
                "prompts_dir": Path(__file__).resolve().parents[2] / "bench" / "prompts",
                "models": ["a:q4,b:q8"],
                "num_ctx": [4096],
                "concurrency": [1, 2],
                "think": ["true,false"],
                "repeats": 2,
                "include_heavy": False,
            },
        )()
    )
    assert args.models == ["a:q4", "b:q8"]
    assert args.think == [True, False]
    assert args.concurrency == [1, 2]
    assert args.include_heavy is False


def test_include_heavy_flag_keeps_regular_models() -> None:
    root = Path(__file__).resolve().parents[2]
    args = config_from_args(
        type(
            "Args",
            (),
            {
                "config": root / "bench" / "config.yaml",
                "prompts_dir": root / "bench" / "prompts",
                "models": None,
                "num_ctx": None,
                "concurrency": None,
                "think": None,
                "repeats": None,
                "include_heavy": True,
            },
        )()
    )
    assert args.include_heavy is True
    assert args.heavy_models == ["qwen3.8:27b"]
    assert args.models == ["qwen3:4b-instruct-2507-q4_K_M", "qwen3:8b", "qwen3:14b"]
    assert args.bust_cache is True
    assert args.num_predict == 512
