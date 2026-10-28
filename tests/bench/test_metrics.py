"""压测解析、分位数和报告，不访问 Ollama。"""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.bench_ollama import format_memory
from bench.metrics import (
    PARALLEL_WARNING,
    BenchConfig,
    StreamFold,
    build_sample,
    clean_ollama_version,
    fold_chat_event,
    inspect_output,
    load_bench_config,
    median,
    model_with_quant,
    override_config,
    parse_ollama_ps,
    percentile,
    prefix_cache_bust,
    processor_offloaded,
    quantization_label,
    recommend,
    render_report,
    report_notes,
    summarize,
    tokens_per_second,
)

ROOT = Path(__file__).resolve().parents[2]

PS_TABLE = (
    "NAME                             ID              SIZE      PROCESSOR    "
    "CONTEXT    RUNNER      UNTIL               \n"
    "qwen3:4b-instruct-2507-q4_K_M    0edcdef34593    3.8 GB    100% GPU     "
    "8192       llamacpp    19 minutes from now    \n"
)


def test_parse_ollama_ps_reads_size_and_processor() -> None:
    rows = parse_ollama_ps(PS_TABLE)
    assert rows == [
        {
            "name": "qwen3:4b-instruct-2507-q4_K_M",
            "size": "3.8 GB",
            "processor": "100% GPU",
        }
    ]
    assert parse_ollama_ps("ollama 没有运行") == []
    assert parse_ollama_ps("NAME ID SIZE PROCESSOR\n") == []


def test_tokens_per_second_uses_nanoseconds() -> None:
    assert tokens_per_second(10, 2_000_000_000) == 5
    assert tokens_per_second(4, 500_000_000) == 8
    assert tokens_per_second(10, 0) is None
    assert tokens_per_second(None, 2_000_000_000) is None


def test_median_and_p95() -> None:
    assert median([]) is None
    assert median([2, 1, 3]) == 2
    assert median([1, 2, 3, 4]) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3], 95) == pytest.approx(2.9)
    assert percentile([], 95) is None
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_fold_stream_records_ttft_and_durations() -> None:
    state = StreamFold()
    fold_chat_event(state, {"message": {"content": ""}}, 0.1)
    assert state.ttft_s is None
    fold_chat_event(state, {"message": {"thinking": "先想"}}, 0.2)
    fold_chat_event(state, {"message": {"content": "你好"}, "model": "demo"}, 0.4)
    fold_chat_event(
        state,
        {
            "done": True,
            "eval_count": 8,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 4.0,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 3_000_000_000,
        },
        0.9,
    )
    assert state.ttft_s == 0.4
    assert state.first_thinking_ms == pytest.approx(200)
    assert state.model == "demo"
    assert state.done is True
    assert state.eval_count == 8
    assert state.prompt_eval_count == 4
    assert state.load_duration_ns == 3_000_000_000
    broken = StreamFold()
    fold_chat_event(broken, ["nope"], 0.1)
    assert broken.error is not None
    failed = StreamFold()
    fold_chat_event(failed, {"error": "model not found"}, 0.1)
    assert "model not found" in str(failed.error)


def test_empty_and_role_chunks_do_not_count_as_ttft() -> None:
    state = StreamFold()
    fold_chat_event(state, {"message": {"role": "assistant"}}, 0.1)
    fold_chat_event(state, {"message": {"role": "assistant", "content": ""}}, 0.2)
    fold_chat_event(state, {"message": {"thinking": ""}}, 0.3)
    assert state.ttft_s is None
    assert state.first_thinking_ms is None
    assert state.content_parts == []
    assert state.thinking_parts == []
    fold_chat_event(state, {"message": {"thinking": "先想"}}, 0.4)
    assert state.ttft_s is None
    assert state.first_thinking_ms == pytest.approx(400)
    fold_chat_event(state, {"message": {"content": "正文"}}, 0.8)
    assert state.ttft_s == pytest.approx(0.8)
    fold_chat_event(state, {"message": {"thinking": "再想", "content": "续"}}, 0.9)
    assert state.first_thinking_ms == pytest.approx(400)
    assert state.ttft_s == pytest.approx(0.8)


def test_cache_bust_prefixes_system_before_user() -> None:
    system_first = prefix_cache_bust(
        [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "问题"},
        ],
        True,
        nonce="abc",
    )
    assert system_first[0]["content"] == "[req:abc]\n规则"
    assert system_first[1]["content"] == "问题"
    user_only = prefix_cache_bust([{"role": "user", "content": "问题"}], True, nonce="abc")
    assert user_only[0]["content"] == "[req:abc]\n问题"
    assert prefix_cache_bust([{"role": "user", "content": "问题"}], False)[0]["content"] == "问题"
    first = prefix_cache_bust([{"role": "user", "content": "问题"}], True)[0]["content"]
    second = prefix_cache_bust([{"role": "user", "content": "问题"}], True)[0]["content"]
    assert first.startswith("[req:")
    assert first != second


def test_inspect_output_uses_gateway_stripper() -> None:
    split = inspect_output(["<thi", "nk>隐藏</think>你", "好"], [])
    assert split["visible"] == "你好"
    assert split["raw_has_think"] is True
    assert split["gateway_stripped"] is True
    assert split["visible_has_think_tag"] is False

    unclosed = inspect_output(["<think>一直想"], [])
    assert unclosed["visible"] == ""
    assert unclosed["gateway_stripped"] is True

    field_only = inspect_output(["你好"], ["秘密"])
    assert field_only["raw_has_thinking_field"] is True
    assert field_only["gateway_stripped"] is True
    assert field_only["visible"] == "你好"

    plain = inspect_output(["你好"], [])
    assert plain["raw_has_think"] is False
    assert plain["gateway_stripped"] is None


def test_build_sample_marks_incomplete_stream() -> None:
    state = StreamFold()
    fold_chat_event(state, {"message": {"content": "<think>隐藏</think>好"}}, 0.3)
    sample = build_sample(
        meta={"model": "demo", "prompt": "short", "num_ctx": 4096, "concurrency": 1, "think": True},
        state=state,
        total_s=1.2,
        error=None,
        excluded=False,
    )
    assert sample["ok"] is False
    assert sample["error"] == "流式响应没有正常结束"
    assert sample["ttft_s"] == pytest.approx(0.3)
    assert sample["first_thinking_ms"] is None
    assert sample["gen_tokens_per_s"] is None
    assert sample["raw_has_think"] is True
    assert sample["gateway_stripped"] is True


def test_summarize_skips_warmup_and_reports_median() -> None:
    def sample(total: float, *, ok: bool = True, excluded: bool = False) -> dict:
        return {
            "model": "demo:q4_K_M",
            "prompt": "short",
            "num_ctx": 4096,
            "concurrency": 1,
            "think": False,
            "excluded": excluded,
            "ok": ok,
            "ttft_s": total / 2 if ok else None,
            "total_s": total if ok else None,
            "gen_tokens_per_s": 10.0 if ok else None,
            "prefill_tokens_per_s": 20.0 if ok else None,
            "raw_has_think": False,
            "gateway_stripped": None,
        }

    rows = summarize(
        [
            sample(9, excluded=True),
            sample(1),
            sample(3),
            sample(2, ok=False),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["count"] == 3
    assert rows[0]["fail_rate"] == pytest.approx(1 / 3)
    assert rows[0]["total_median"] == 2
    assert rows[0]["ttft_median"] == 1
    assert rows[0]["ttft_p95"] == pytest.approx(1.45)
    assert rows[0]["truncation_rate"] == 0


def test_truncation_rate_uses_group_size() -> None:
    def sample(eval_count: int, *, ok: bool = True) -> dict:
        return {
            "model": "demo",
            "prompt": "short",
            "num_ctx": 4096,
            "concurrency": 1,
            "think": False,
            "ok": ok,
            "eval_count": eval_count,
            "num_predict": 512,
            "ttft_s": 0.2 if ok else None,
            "total_s": 1,
            "gen_tokens_per_s": 10,
            "prefill_tokens_per_s": 20,
            "raw_has_think": False,
            "gateway_stripped": None,
        }

    rows = summarize([sample(512), sample(100), sample(600, ok=False)])
    assert rows[0]["truncation_rate"] == pytest.approx(1 / 3)


def test_recommend_sentence_follows_template() -> None:
    groups = [
        {
            "model": "demo:q4_K_M",
            "prompt": "short",
            "num_ctx": 4096,
            "concurrency": 1,
            "think": False,
            "fail_rate": 0,
            "ttft_median": 0.4,
            "total_median": 1.2,
            "gen_tps_median": 30,
        },
        {
            "model": "demo:q4_K_M",
            "prompt": "short",
            "num_ctx": 4096,
            "concurrency": 1,
            "think": True,
            "fail_rate": 0,
            "ttft_median": 0.35,
            "total_median": 1.5,
            "gen_tps_median": 28,
        },
    ]
    other = {
        "model": "plain-model",
        "prompt": "short",
        "num_ctx": 4096,
        "concurrency": 1,
        "think": False,
        "fail_rate": 0,
        "ttft_median": 0.9,
        "total_median": 2.0,
        "gen_tps_median": 10,
    }
    single = recommend({"chip": "Apple M3 Pro", "memory": "18 GB"}, groups, ["qwen3.8:27b"])
    assert single == "在 [Apple M3 Pro / 18 GB] 上，样本不足，暂不下结论。"
    text = recommend(
        {"chip": "Apple M3 Pro", "memory": "18 GB"},
        [*groups, other],
        ["qwen3.8:27b"],
    )
    assert text == (
        "在 [Apple M3 Pro / 18 GB] 上，客服场景推荐 "
        "[demo:q4_K_M + num_ctx=4096 + think=false]，"
        "理由是提示词 short 的首 token 中位数为 0.400s，总耗时中位数为 1.200s，"
        "生成速度中位数为 30.0 tokens/s，失败率为 0%，客服回复沿用关闭思考的这一组，"
        "未拉取因而跳过的模型：qwen3.8:27b。"
    )
    assert "q4_K_M + q4_K_M" not in text
    assert "没有出现在本次成功样本里" not in text
    faster = [dict(group) for group in groups]
    faster[1]["ttft_median"] = 0.1
    opened = recommend({"chip": "Apple M3 Pro", "memory": "18 GB"}, [*faster, other], [])
    assert "think=true" in opened
    assert "打开思考后首 token 明显更低" in opened
    only_open = recommend({"chip": "Apple M3 Pro", "memory": "18 GB"}, [groups[1]], [])
    assert only_open.endswith("样本不足，暂不下结论。")
    failed = dict(other)
    failed["fail_rate"] = 1
    assert "样本不足" in recommend(
        {"chip": "Apple M3 Pro", "memory": "18 GB"},
        [*groups, failed],
        [],
    )
    assert quantization_label("plain-model") == "未标注量化"
    assert model_with_quant("demo:q4_K_M") == "demo:q4_K_M"
    assert model_with_quant("plain-model") == "plain-model"
    empty = recommend({}, [], [])
    assert empty == "在 [未知机型 / 未知内存] 上，样本不足，暂不下结论。"


def test_render_report_includes_machine_and_parallel_warning() -> None:
    document = {
        "machine": {
            "chip": "Apple M3 Pro",
            "memory": "18 GB",
            "macos": "14.4",
            "arch": "arm64",
            "ollama": "ollama version is 0.40.11",
        },
        "ollama_num_parallel": None,
        "config": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 256,
            "models": ["demo:q4_K_M"],
            "num_ctx": [4096],
            "concurrency": [1],
            "think": [False],
            "repeats": 3,
        },
        "skipped_models": ["qwen3.8:27b"],
        "loads": [{"model": "demo", "num_ctx": 4096, "load_duration_s": 1.5}],
        "ollama_ps": [{"name": "demo", "size": "3.8 GB", "processor": "100% GPU"}],
        "summary": [
            {
                "model": "demo:q4_K_M",
                "prompt": "short",
                "num_ctx": 4096,
                "concurrency": 1,
                "think": False,
                "count": 3,
                "fail_rate": 0,
                "truncation_rate": 1 / 3,
                "ttft_median": 0.4,
                "ttft_p95": 0.5,
                "total_median": 1.2,
                "total_p95": 1.4,
                "gen_tps_median": 30,
                "gen_tps_p95": 32,
                "prefill_tps_median": 100,
                "prefill_tps_p95": 110,
                "raw_think_yes": 0,
                "raw_think_observed": 3,
                "gateway_stripped_yes": 0,
                "gateway_stripped_observed": 0,
            }
        ],
        "conclusion": "在 [Apple M3 Pro / 18 GB] 上，客服场景推荐 [demo]，理由是示例。",
    }
    report = render_report(document)
    assert "Apple M3 Pro" in report
    assert "18 GB" in report
    assert "14.4" in report
    assert "| Ollama | 0.40.11 |" in report
    assert "ollama version is" not in report
    assert clean_ollama_version("Ollama version is 0.40.11") == "0.40.11"
    assert PARALLEL_WARNING in report
    assert "[req:uuid]" in report
    assert "截断率" in report
    assert "33.3%" in report
    assert "未设置" in report
    assert "qwen3.8:27b" in report
    assert "3.8 GB" in report
    assert "100% GPU" in report
    assert "1.500s" in report
    assert "0.400" in report
    assert "0/3" in report
    assert "0.0%" in report
    assert "本次模型：demo:q4_K_M" in report
    assert report.index("temperature=") < report.index("本次模型") < report.index("每组重复")


def test_report_notes_mentions_missing_think_cap_and_fast_repeats() -> None:
    notes = report_notes(
        {
            "config": {"num_predict": 256},
            "summary": [{"think": True, "raw_think_yes": 0, "gateway_stripped_yes": 0}],
            "samples": [
                {
                    "ok": True,
                    "prompt": "short",
                    "model": "m",
                    "num_ctx": 4096,
                    "think": True,
                    "eval_count": 256,
                    "prefill_tokens_per_s": 10,
                },
                {
                    "ok": True,
                    "prompt": "short",
                    "model": "m",
                    "num_ctx": 4096,
                    "think": True,
                    "eval_count": 256,
                    "prefill_tokens_per_s": 200,
                },
            ],
        }
    )
    text = "\n".join(notes)
    assert "没有 `<think>`" in text
    assert "num_predict=256" in text
    assert "prompt_eval_duration" in text
    stripped = report_notes(
        {
            "summary": [{"think": True, "raw_think_yes": 1, "gateway_stripped_yes": 1}],
            "samples": [],
        }
    )
    assert any("不再包含" in note for note in stripped)
    mixed = report_notes(
        {
            "summary": [],
            "samples": [],
            "ollama_ps": [
                {"processor": "100% GPU"},
                {"processor": "30%/70% CPU/GPU"},
            ],
        }
    )
    assert any("发生 CPU 卸载" in note for note in mixed)
    assert "30%/70% CPU/GPU" in "\n".join(mixed)
    gpu_only = report_notes(
        {"summary": [], "samples": [], "ollama_ps": [{"processor": "100% GPU"}]}
    )
    assert all("CPU 卸载" not in note for note in gpu_only)
    assert processor_offloaded("100% CPU") is False


def test_parse_ollama_ps_keeps_mixed_processor_text() -> None:
    text = "NAME  ID  SIZE  PROCESSOR         CONTEXT\nbig   abc 16GB  30%/70% CPU/GPU   4096\n"
    rows = parse_ollama_ps(text)
    assert rows[0]["processor"] == "30%/70% CPU/GPU"
    assert rows[0]["size"] == "16GB"


def test_repo_config_matches_the_fixed_matrix() -> None:
    config = load_bench_config(ROOT / "bench" / "config.yaml", ROOT / "bench" / "prompts")
    assert config.models == [
        "qwen3:4b-instruct-2507-q4_K_M",
        "qwen3:8b",
        "qwen3:14b",
    ]
    assert config.heavy_models == ["qwen3.8:27b"]
    assert config.include_heavy is False
    assert config.bust_cache is True
    assert config.num_predict == 512
    assert config.num_ctx == [4096, 8192, 16384]
    assert config.concurrency == [1, 2, 4]
    assert config.think == [False, True]
    assert config.repeats == 3
    assert config.temperature == 0
    assert config.seed == 42
    short = config.prompts["short"].strip()
    assert 45 <= len(short) <= 55
    material, separator, question = config.prompts["long"].partition("问题：")
    assert separator == "问题："
    assert 2900 <= len(material.strip()) <= 3300
    assert question.strip()
    narrowed = override_config(
        config,
        models=["qwen3:4b-instruct-2507-q4_K_M"],
        num_ctx=[4096],
        concurrency=[1],
        repeats=1,
    )
    assert narrowed.models == ["qwen3:4b-instruct-2507-q4_K_M"]
    assert narrowed.num_ctx == [4096]
    assert narrowed.concurrency == [1]
    assert narrowed.repeats == 1
    assert narrowed.think == [False, True]
    assert narrowed.heavy_models == ["qwen3.8:27b"]
    assert narrowed.include_heavy is False
    assert narrowed.bust_cache is True
    assert narrowed.num_predict == 512
    with pytest.raises(ValueError):
        override_config(config, repeats=0)


def test_missing_prompt_file_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models: [demo]\nnum_ctx: [4096]\nconcurrency: [1]\nthink: [false]\n"
        "repeats: 1\nprompts: [missing]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="缺少测试提示词"):
        load_bench_config(config_path, tmp_path)


def test_format_memory_rounds_gib() -> None:
    assert format_memory("19327352832") == "18 GB"
    assert format_memory("nope") == "nope"
    assert format_memory("") == "未知"


def test_ci_does_not_launch_the_bench() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "bench_ollama" not in workflow
    assert "ollama" not in workflow.lower()


def test_sample_config_type_is_explicit() -> None:
    config = BenchConfig(
        models=["demo"],
        num_ctx=[4096],
        concurrency=[1],
        think=[False],
        repeats=1,
        temperature=0,
        seed=42,
        base_url="http://127.0.0.1:11434",
        timeout=5,
        num_predict=16,
        prompts={"short": "问题"},
    )
    assert config.seed == 42
