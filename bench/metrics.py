"""压测的解析和统计。这些函数不访问网络。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from app.llm.base import ThinkStripper

PARALLEL_WARNING = "Ollama 的实际并行数受 OLLAMA_NUM_PARALLEL 环境变量影响"
_QUANT = re.compile(r"(q\d+_k_[a-z0-9]+|q\d+_[a-z0-9]+|q\d+|fp16|f16|iq\d+)", re.IGNORECASE)
_OLLAMA_VERSION_PREFIX = "ollama version is "


@dataclass
class BenchConfig:
    models: list[str]
    num_ctx: list[int]
    concurrency: list[int]
    think: list[bool]
    repeats: int
    temperature: float
    seed: int
    base_url: str
    timeout: float
    num_predict: int
    prompts: dict[str, str]
    keep_alive: str = "10m"
    bust_cache: bool = True
    heavy_models: list[str] = field(default_factory=list)
    include_heavy: bool = False


@dataclass
class StreamFold:
    """把 Ollama 的流式 JSON 收成一条样本需要的字段。"""

    content_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    ttft_s: float | None = None
    first_thinking_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    load_duration_ns: int | None = None
    model: str = ""
    error: str | None = None
    done: bool = False


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def ns_to_seconds(duration_ns: int | None) -> float | None:
    if duration_ns is None:
        return None
    return duration_ns / 1_000_000_000


def tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    """Ollama 的时长是纳秒。时长为 0 时不计算速度。"""
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return count / (duration_ns / 1_000_000_000)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2


def percentile(values: list[float], percent: float) -> float | None:
    """线性插值。位置是 (n - 1) * percent / 100。"""
    if not values:
        return None
    if percent < 0 or percent > 100:
        raise ValueError("分位数必须在 0 到 100 之间")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (percent / 100) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low]) * (1 - weight) + float(ordered[high]) * weight


def parse_ollama_ps(text: str) -> list[dict[str, str]]:
    """按表头列起点切 SIZE 和 PROCESSOR。模型名本身不含空格，单元格可以含空格。"""
    lines = text.splitlines()
    header = ""
    header_index = -1
    for index, line in enumerate(lines):
        if "NAME" in line and "SIZE" in line and "PROCESSOR" in line:
            header = line
            header_index = index
            break
    if header_index < 0:
        return []
    columns = [(match.group(), match.start()) for match in re.finditer(r"\S+", header)]
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        cells: dict[str, str] = {}
        for index, (name, start) in enumerate(columns):
            end = columns[index + 1][1] if index + 1 < len(columns) else len(line)
            cells[name] = line[start:end].strip() if start < len(line) else ""
        if not cells.get("NAME"):
            continue
        rows.append(
            {
                "name": cells.get("NAME", ""),
                "size": cells.get("SIZE", ""),
                "processor": cells.get("PROCESSOR", ""),
            }
        )
    return rows


def fold_chat_event(state: StreamFold, data: Any, elapsed_s: float) -> None:
    if not isinstance(data, dict):
        state.error = "模型返回了无法解析的流式内容"
        return
    if data.get("error") and "message" not in data:
        state.error = f"模型接口返回错误：{data.get('error')}"
        return
    if data.get("model"):
        state.model = str(data["model"])
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    _take_content(state, message.get("content"), elapsed_s)
    _take_thinking(state, message.get("thinking"), elapsed_s)
    for key, attr in (
        ("eval_count", "eval_count"),
        ("eval_duration", "eval_duration_ns"),
        ("prompt_eval_count", "prompt_eval_count"),
        ("prompt_eval_duration", "prompt_eval_duration_ns"),
        ("load_duration", "load_duration_ns"),
    ):
        if key in data and data[key] is not None:
            setattr(state, attr, as_int(data[key]))
    if data.get("done"):
        state.done = True


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _take_content(state: StreamFold, value: Any, elapsed_s: float) -> None:
    """TTFT 只认第一个非空 message.content。空块和只有 role 的块没有正文。"""
    text = _nonempty_text(value)
    if text is None:
        return
    if state.ttft_s is None:
        state.ttft_s = elapsed_s
    state.content_parts.append(text)


def _take_thinking(state: StreamFold, value: Any, elapsed_s: float) -> None:
    text = _nonempty_text(value)
    if text is None:
        return
    if state.first_thinking_ms is None:
        state.first_thinking_ms = elapsed_s * 1000
    state.thinking_parts.append(text)


def prefix_cache_bust(
    messages: list[dict[str, Any]],
    enabled: bool,
    nonce: str | None = None,
) -> list[dict[str, Any]]:
    """在 system 或 user 消息前加一行随机 nonce，避免复用 KV 缓存。"""
    if not enabled:
        return messages
    token = nonce if nonce else str(uuid.uuid4())
    prefix = f"[req:{token}]\n"
    updated = [dict(message) for message in messages]
    for role in ("system", "user"):
        for message in updated:
            if message.get("role") == role and isinstance(message.get("content"), str):
                message["content"] = prefix + str(message["content"])
                return updated
    return updated


def inspect_output(content_chunks: list[str], thinking_chunks: list[str]) -> dict[str, Any]:
    """用网关同一个 ThinkStripper 判断思考段有没有从可见正文里去掉。"""
    stripper = ThinkStripper()
    visible = "".join(stripper.feed(chunk) for chunk in content_chunks) + stripper.finish()
    raw = "".join(content_chunks)
    thinking = "".join(thinking_chunks)
    raw_has_think_tag = "<think>" in raw or "</think>" in raw
    visible_has_think_tag = "<think>" in visible or "</think>" in visible
    raw_has_thinking_field = bool(thinking.strip())
    if raw_has_think_tag:
        gateway_stripped: bool | None = not visible_has_think_tag
    elif raw_has_thinking_field:
        gateway_stripped = thinking.strip() not in visible
    else:
        gateway_stripped = None
    return {
        "raw_has_think": raw_has_think_tag or raw_has_thinking_field,
        "raw_has_think_tag": raw_has_think_tag,
        "raw_has_thinking_field": raw_has_thinking_field,
        "visible_has_think_tag": visible_has_think_tag,
        "gateway_stripped": gateway_stripped,
        "visible": visible,
    }


def build_sample(
    *,
    meta: dict[str, Any],
    state: StreamFold,
    total_s: float,
    error: str | None,
    excluded: bool,
) -> dict[str, Any]:
    inspected = inspect_output(state.content_parts, state.thinking_parts)
    visible = str(inspected.pop("visible"))
    if error:
        ok = False
        message: str | None = error
    elif state.error:
        ok = False
        message = state.error
    elif not state.done:
        ok = False
        message = "流式响应没有正常结束"
    else:
        ok = True
        message = None
    return {
        **meta,
        "excluded": excluded,
        "ok": ok,
        "error": message,
        "ttft_s": state.ttft_s,
        "first_thinking_ms": state.first_thinking_ms,
        "total_s": total_s,
        "eval_count": state.eval_count,
        "eval_duration_ns": state.eval_duration_ns,
        "prompt_eval_count": state.prompt_eval_count,
        "prompt_eval_duration_ns": state.prompt_eval_duration_ns,
        "load_duration_ns": state.load_duration_ns,
        "gen_tokens_per_s": tokens_per_second(state.eval_count, state.eval_duration_ns),
        "prefill_tokens_per_s": tokens_per_second(
            state.prompt_eval_count,
            state.prompt_eval_duration_ns,
        ),
        "visible_preview": visible[:200],
        **inspected,
    }


def _numbers(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for sample in samples:
        if sample.get("excluded"):
            continue
        key = (
            sample.get("model"),
            sample.get("prompt"),
            sample.get("num_ctx"),
            sample.get("concurrency"),
            bool(sample.get("think")),
        )
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(sample)
    return [_summarize_group(key, groups[key]) for key in order]


def _summarize_group(key: tuple[Any, ...], group: list[dict[str, Any]]) -> dict[str, Any]:
    ok_samples = [sample for sample in group if sample.get("ok")]
    observed = [sample for sample in ok_samples if sample.get("raw_has_think") is not None]
    stripped = [sample for sample in ok_samples if sample.get("gateway_stripped") is not None]
    model, prompt, num_ctx, concurrency, think = key
    return {
        "model": model,
        "prompt": prompt,
        "num_ctx": num_ctx,
        "concurrency": concurrency,
        "think": think,
        "count": len(group),
        "fail_rate": (len(group) - len(ok_samples)) / len(group) if group else None,
        "truncation_rate": _truncation_rate(group),
        "ttft_median": median(_numbers(ok_samples, "ttft_s")),
        "ttft_p95": percentile(_numbers(ok_samples, "ttft_s"), 95),
        "total_median": median(_numbers(ok_samples, "total_s")),
        "total_p95": percentile(_numbers(ok_samples, "total_s"), 95),
        "gen_tps_median": median(_numbers(ok_samples, "gen_tokens_per_s")),
        "gen_tps_p95": percentile(_numbers(ok_samples, "gen_tokens_per_s"), 95),
        "prefill_tps_median": median(_numbers(ok_samples, "prefill_tokens_per_s")),
        "prefill_tps_p95": percentile(_numbers(ok_samples, "prefill_tokens_per_s"), 95),
        "raw_think_yes": sum(1 for sample in observed if sample.get("raw_has_think") is True),
        "raw_think_observed": len(observed),
        "gateway_stripped_yes": sum(
            1 for sample in stripped if sample.get("gateway_stripped") is True
        ),
        "gateway_stripped_observed": len(stripped),
    }


def _truncation_rate(group: list[dict[str, Any]]) -> float | None:
    """eval_count 达到或超过 num_predict 的比例。失败样本算未截断，分母是整组。"""
    if not group:
        return None
    truncated = 0
    for sample in group:
        if not sample.get("ok"):
            continue
        cap = sample.get("num_predict")
        count = sample.get("eval_count")
        if _is_number(cap) and _is_number(count) and float(count) >= float(cap):
            truncated += 1
    return truncated / len(group)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def quantization_label(model: str) -> str:
    found = _QUANT.search(model)
    if found is None:
        return "未标注量化"
    return found.group(1)


def model_with_quant(model: str) -> str:
    """量化标签已经写在模型名里时不再拼接一遍。"""
    label = quantization_label(model)
    if label == "未标注量化" or label.lower() in model.lower():
        return model
    return f"{model} + {label}"


def _successful_models(groups: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for group in groups:
        fail_rate = group.get("fail_rate")
        model = group.get("model")
        if not model or not _is_number(fail_rate):
            continue
        if float(fail_rate) < 1:
            found.add(str(model))
    return found


def recommend(
    machine: dict[str, str],
    groups: list[dict[str, Any]],
    skipped: list[str],
) -> str:
    host = f"{machine.get('chip') or '未知机型'} / {machine.get('memory') or '未知内存'}"
    insufficient = f"在 [{host}] 上，样本不足，暂不下结论。"
    if len(_successful_models(groups)) < 2:
        return insufficient
    usable = [
        group
        for group in groups
        if group.get("prompt") == "short"
        and group.get("fail_rate") == 0
        and _is_number(group.get("ttft_median"))
    ]
    if not usable:
        usable = [
            group
            for group in groups
            if group.get("fail_rate") == 0 and _is_number(group.get("ttft_median"))
        ]
    if not usable:
        return insufficient
    chosen = _choose_group(usable)
    think = "true" if chosen["think"] else "false"
    setting = (
        f"{model_with_quant(str(chosen['model']))} + num_ctx={chosen['num_ctx']} + think={think}"
    )
    reason = _reason(chosen, groups, skipped)
    return f"在 [{host}] 上，客服场景推荐 [{setting}]，理由是{reason}"


def _choose_group(groups: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [group for group in groups if not group.get("think")]
    opened = [group for group in groups if group.get("think")]
    if not closed or not opened:
        return min(groups, key=lambda group: float(group["ttft_median"]))
    best_closed = min(closed, key=lambda group: float(group["ttft_median"]))
    best_opened = min(opened, key=lambda group: float(group["ttft_median"]))
    if float(best_opened["ttft_median"]) < float(best_closed["ttft_median"]) * 0.8:
        return best_opened
    return best_closed


def _sibling_group(chosen: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    for group in groups:
        same_setup = (
            group.get("model") == chosen.get("model")
            and group.get("prompt") == chosen.get("prompt")
            and group.get("num_ctx") == chosen.get("num_ctx")
            and group.get("concurrency") == chosen.get("concurrency")
            and bool(group.get("think")) != bool(chosen.get("think"))
            and group.get("fail_rate") == 0
        )
        if same_setup:
            return group
    return None


def _reason(chosen: dict[str, Any], groups: list[dict[str, Any]], skipped: list[str]) -> str:
    pieces = [
        f"提示词 {chosen['prompt']} 的首 token 中位数为 {float(chosen['ttft_median']):.3f}s",
        f"总耗时中位数为 {float(chosen['total_median']):.3f}s",
        f"生成速度中位数为 {_plain_number(chosen.get('gen_tps_median'))} tokens/s",
        f"失败率为 {float(chosen['fail_rate']):.0%}",
    ]
    sibling = _sibling_group(chosen, groups)
    if chosen.get("think") and sibling is not None:
        pieces.append("打开思考后首 token 明显更低，所以取这一组")
    elif chosen.get("think"):
        pieces.append("本次成功样本使用打开思考")
    else:
        pieces.append("客服回复沿用关闭思考的这一组")
    if skipped:
        pieces.append("未拉取因而跳过的模型：" + "、".join(skipped))
    return "，".join(pieces) + "。"


def _plain_number(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "未知"
    return f"{float(value):.1f}"


def clean_ollama_version(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith(_OLLAMA_VERSION_PREFIX):
        return cleaned[len(_OLLAMA_VERSION_PREFIX) :].strip()
    return cleaned


def processor_offloaded(text: str) -> bool:
    """PROCESSOR 同时出现 CPU 和 GPU 时，表示发生了 CPU 卸载。"""
    folded = text.upper()
    return "CPU" in folded and "GPU" in folded


def format_parallel(value: str | None) -> str:
    if value is None or value.strip() == "":
        return "未设置"
    return value


def format_metric(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "—"
    return f"{float(value):.{digits}f}"


def format_ratio(yes: int, observed: int) -> str:
    if observed <= 0:
        return "—"
    return f"{yes}/{observed}"


def report_notes(document: dict[str, Any]) -> list[str]:
    """补充表里看不出来的情况：没有思考段、生成被截断、重复请求变快。"""
    notes: list[str] = []
    summary = document.get("summary") or []
    think_rows = [row for row in summary if row.get("think")]
    if think_rows and all(row.get("raw_think_yes") == 0 for row in think_rows):
        notes.append(
            "think=true 的样本里，原始输出没有 `<think>`，也没有 Ollama 的 thinking 字段。"
            "网关没有需要剥离的思考段。"
        )
    elif any(row.get("gateway_stripped_yes") for row in think_rows):
        notes.append("原始输出里出现过思考段。网关过滤后的可见文本不再包含思考标签。")
    cap = (document.get("config") or {}).get("num_predict")
    if _is_number(cap):
        capped = [
            sample
            for sample in document.get("samples") or []
            if sample.get("ok")
            and _is_number(sample.get("eval_count"))
            and float(sample["eval_count"]) >= float(cap)
        ]
        if capped:
            names = sorted({str(sample.get("prompt")) for sample in capped})
            notes.append(
                "提示词 " + "、".join(names) + f" 的 eval_count 达到 num_predict={cap}，"
                "这些回答在生成上限处停下。"
            )
    speeds: dict[tuple[Any, ...], list[float]] = {}
    for sample in document.get("samples") or []:
        speed = sample.get("prefill_tokens_per_s")
        if not sample.get("ok") or not isinstance(speed, int | float):
            continue
        key = (
            sample.get("model"),
            sample.get("prompt"),
            sample.get("num_ctx"),
            sample.get("think"),
        )
        speeds.setdefault(key, []).append(float(speed))
    spread = any(
        len(values) >= 2 and min(values) > 0 and max(values) > min(values) * 10
        for values in speeds.values()
    )
    if spread:
        notes.append(
            "同一提示词第一次的 prompt_eval_duration 明显更长，后面的重复短很多。"
            "预填充速度的中位数会偏向后面这些更快的重复。"
        )
    offloaded = [
        str(row.get("processor"))
        for row in document.get("ollama_ps") or []
        if isinstance(row, dict) and processor_offloaded(str(row.get("processor") or ""))
    ]
    if offloaded:
        notes.append("PROCESSOR 为 " + "、".join(offloaded) + "，发生 CPU 卸载。")
    return notes


def _bust_cache_line(config: dict[str, Any]) -> str:
    if config.get("bust_cache", True):
        return (
            "- bust_cache=true：每次请求在 system 或 user 消息前加一行随机 nonce"
            "（`[req:uuid]`），避免 Ollama 复用 KV 缓存。"
        )
    return "- bust_cache=false：请求正文不加随机 nonce。"


def _heavy_line(config: dict[str, Any]) -> str | None:
    names = [str(item) for item in config.get("heavy_models") or []]
    if not names:
        return None
    listed = "、".join(names)
    if config.get("include_heavy"):
        return f"- 重模型：{listed}。本次已包含，只使用 num_ctx=4096、并发 1。"
    return (
        f"- 重模型：{listed}。本次未运行。"
        "传入 --include-heavy 后才会执行，并且只使用 num_ctx=4096、并发 1。"
    )


def _run_matrix_line(config: dict[str, Any]) -> str:
    models = "、".join(str(item) for item in config.get("models") or [])
    num_ctx = "、".join(str(item) for item in config.get("num_ctx") or [])
    concurrency = "、".join(str(item) for item in config.get("concurrency") or [])
    think = "、".join("true" if item else "false" for item in config.get("think") or [])
    repeats = config.get("repeats", "")
    return (
        f"- 本次模型：{models}；num_ctx：{num_ctx}；并发：{concurrency}；"
        f"think：{think}；重复：{repeats}。"
    )


def render_report(document: dict[str, Any]) -> str:
    machine = document.get("machine") or {}
    config = document.get("config") or {}
    ollama = clean_ollama_version(str(machine.get("ollama") or ""))
    lines = [
        "# Ollama 压测报告",
        "",
        "## 机器",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 芯片 | {machine.get('chip') or '未知'} |",
        f"| 内存 | {machine.get('memory') or '未知'} |",
        f"| macOS | {machine.get('macos') or '未知'} |",
        f"| 架构 | {machine.get('arch') or '未知'} |",
        f"| Ollama | {ollama or '未知'} |",
        "",
        "## 运行条件",
        "",
        (
            f"- temperature={config.get('temperature', 0)}，seed={config.get('seed', 42)}，"
            f"num_predict={config.get('num_predict', 512)}。"
        ),
        "- TTFT 是第一个非空 message.content 到达的时间。thinking 字段另记 first_thinking_ms。",
        "- 截断率是 eval_count 达到或超过 num_predict 的样本占该组样本的比例。",
        "- 每组重复多次后取中位数和 P95。P95 用线性插值，位置是 (n - 1) * 0.95。",
        "- 每个模型先预热 1 次。预热请求和它的 load_duration 不计入结果表。",
        "- 生成速度用 eval_count / eval_duration。",
        "- 预填充速度用 prompt_eval_count / prompt_eval_duration。",
        _bust_cache_line(config),
        (f"- {PARALLEL_WARNING}。当前值：{format_parallel(document.get('ollama_num_parallel'))}。"),
        "",
    ]
    if config.get("models"):
        lines.insert(lines.index("## 运行条件") + 3, _run_matrix_line(config))
    heavy = _heavy_line(config)
    if heavy:
        anchor = lines.index("## 运行条件")
        offset = 4 if config.get("models") else 3
        lines.insert(anchor + offset, heavy)
    if document.get("error"):
        lines.extend(["## 错误", "", str(document["error"]), ""])
    lines.extend(["## 跳过的模型", ""])
    skipped = document.get("skipped_models") or []
    if skipped:
        lines.extend(f"- {name}" for name in skipped)
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 首次加载",
            "",
            "| 模型 | num_ctx | load_duration |",
            "| --- | --- | --- |",
        ]
    )
    loads = document.get("loads") or []
    if not loads:
        lines.append("| — | — | — |")
    for item in loads:
        lines.append(
            "| {model} | {num_ctx} | {duration} |".format(
                model=item.get("model") or "—",
                num_ctx=item.get("num_ctx") if item.get("num_ctx") is not None else "—",
                duration=_load_text(item.get("load_duration_s")),
            )
        )
    lines.extend(
        [
            "",
            "上表只记录每个模型第一次预热读到的 load_duration，不计入正式结果。",
            "",
            "## 显存 / 内存",
            "",
            "来自 `ollama ps` 的 SIZE 和 PROCESSOR。",
            "",
            "| 模型 | SIZE | PROCESSOR |",
            "| --- | --- | --- |",
        ]
    )
    ps_rows = document.get("ollama_ps") or []
    if not ps_rows:
        lines.append("| — | — | — |")
    for row in ps_rows:
        name = row.get("name") or "—"
        size = row.get("size") or "—"
        processor = row.get("processor") or "—"
        lines.append(f"| {name} | {size} | {processor} |")
    result_headers = [
        "模型",
        "提示词",
        "num_ctx",
        "并发",
        "think",
        "样本",
        "失败率",
        "截断率",
        "TTFT 中位",
        "TTFT P95",
        "总耗时中位",
        "总耗时 P95",
        "生成 tok/s 中位",
        "生成 tok/s P95",
        "预填充 tok/s 中位",
        "预填充 tok/s P95",
        "原始含思考",
        "网关已剥离",
    ]
    lines.extend(["", "## 结果", ""])
    lines.append("| " + " | ".join(result_headers) + " |")
    lines.append("| " + " | ".join("---" for _ in result_headers) + " |")
    summary = document.get("summary") or []
    if not summary:
        lines.append("| " + " | ".join("—" for _ in result_headers) + " |")
    for row in summary:
        lines.append(_result_row(row))
    notes = report_notes(document)
    if notes:
        lines.extend(["", "## 观察", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## 结论", "", str(document.get("conclusion") or ""), ""])
    return "\n".join(lines)


def _load_text(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "—"
    return f"{float(value):.3f}s"


def format_percent(value: Any) -> str:
    if not _is_number(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def _result_row(row: dict[str, Any]) -> str:
    fail_text = format_percent(row.get("fail_rate"))
    cells = [
        str(row.get("model") or "—"),
        str(row.get("prompt") or "—"),
        str(row.get("num_ctx") if row.get("num_ctx") is not None else "—"),
        str(row.get("concurrency") if row.get("concurrency") is not None else "—"),
        "true" if row.get("think") else "false",
        str(row.get("count") if row.get("count") is not None else "—"),
        fail_text,
        format_percent(row.get("truncation_rate")),
        format_metric(row.get("ttft_median")),
        format_metric(row.get("ttft_p95")),
        format_metric(row.get("total_median")),
        format_metric(row.get("total_p95")),
        format_metric(row.get("gen_tps_median"), 1),
        format_metric(row.get("gen_tps_p95"), 1),
        format_metric(row.get("prefill_tps_median"), 1),
        format_metric(row.get("prefill_tps_p95"), 1),
        format_ratio(int(row.get("raw_think_yes") or 0), int(row.get("raw_think_observed") or 0)),
        format_ratio(
            int(row.get("gateway_stripped_yes") or 0),
            int(row.get("gateway_stripped_observed") or 0),
        ),
    ]
    return "| " + " | ".join(cells) + " |"


def validate_config(config: BenchConfig) -> None:
    if not config.models:
        raise ValueError("models 不能为空")
    if not config.num_ctx or not config.concurrency or not config.think or not config.prompts:
        raise ValueError("num_ctx、concurrency、think、prompts 都不能为空")
    if config.repeats < 1:
        raise ValueError("repeats 至少为 1")
    if any(item < 1 for item in config.concurrency):
        raise ValueError("concurrency 至少为 1")
    if any(item < 1 for item in config.num_ctx):
        raise ValueError("num_ctx 至少为 1")


def load_bench_config(path: Path, prompts_dir: Path) -> BenchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("压测配置必须是映射")
    prompt_names = raw.get("prompts") or ["short", "long"]
    prompts: dict[str, str] = {}
    for name in prompt_names:
        prompt_path = prompts_dir / f"{name}.txt"
        if not prompt_path.is_file():
            raise ValueError(f"缺少测试提示词：{prompt_path}")
        prompts[str(name)] = prompt_path.read_text(encoding="utf-8")
    config = BenchConfig(
        models=[str(item) for item in raw.get("models") or []],
        num_ctx=[int(item) for item in raw.get("num_ctx") or []],
        concurrency=[int(item) for item in raw.get("concurrency") or []],
        think=[bool(item) for item in raw.get("think") or []],
        repeats=int(raw.get("repeats", 3)),
        temperature=float(raw.get("temperature", 0)),
        seed=int(raw.get("seed", 42)),
        base_url=str(raw.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
        timeout=float(raw.get("timeout_seconds", 240)),
        num_predict=int(raw.get("num_predict", 512)),
        prompts=prompts,
        keep_alive=str(raw.get("keep_alive", "10m")),
        bust_cache=_config_bool(raw.get("bust_cache"), default=True),
        heavy_models=[str(item) for item in raw.get("heavy_models") or []],
        include_heavy=_config_bool(raw.get("include_heavy"), default=False),
    )
    validate_config(config)
    return config


def _config_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return bool(value)


def override_config(
    config: BenchConfig,
    *,
    models: list[str] | None = None,
    num_ctx: list[int] | None = None,
    concurrency: list[int] | None = None,
    think: list[bool] | None = None,
    repeats: int | None = None,
    include_heavy: bool | None = None,
) -> BenchConfig:
    updated = BenchConfig(
        models=list(models) if models is not None else list(config.models),
        num_ctx=list(num_ctx) if num_ctx is not None else list(config.num_ctx),
        concurrency=list(concurrency) if concurrency is not None else list(config.concurrency),
        think=list(think) if think is not None else list(config.think),
        repeats=config.repeats if repeats is None else repeats,
        temperature=config.temperature,
        seed=config.seed,
        base_url=config.base_url,
        timeout=config.timeout,
        num_predict=config.num_predict,
        prompts=dict(config.prompts),
        keep_alive=config.keep_alive,
        bust_cache=config.bust_cache,
        heavy_models=list(config.heavy_models),
        include_heavy=config.include_heavy if include_heavy is None else include_heavy,
    )
    validate_config(updated)
    return updated


def config_public(config: BenchConfig) -> dict[str, Any]:
    return {
        "models": list(config.models),
        "num_ctx": list(config.num_ctx),
        "concurrency": list(config.concurrency),
        "think": list(config.think),
        "repeats": config.repeats,
        "temperature": config.temperature,
        "seed": config.seed,
        "num_predict": config.num_predict,
        "timeout_seconds": config.timeout,
        "keep_alive": config.keep_alive,
        "base_url": config.base_url,
        "bust_cache": config.bust_cache,
        "heavy_models": list(config.heavy_models),
        "include_heavy": config.include_heavy,
        "prompts": [
            {"name": name, "chars": len(text.strip())} for name, text in config.prompts.items()
        ],
    }
