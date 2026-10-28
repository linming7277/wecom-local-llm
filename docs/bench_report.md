# Ollama 压测报告

## 机器

| 项目 | 值 |
| --- | --- |
| 芯片 | Apple M3 Pro |
| 内存 | 18 GB |
| macOS | 14.4 |
| 架构 | arm64 |
| Ollama | 0.40.11 |

## 运行条件

- temperature=0.0，seed=42，num_predict=512。
- 本次模型：qwen3:4b-instruct-2507-q4_K_M；num_ctx：4096；并发：1；think：false、true；重复：3。
- 重模型：qwen3.8:27b。本次未运行。传入 --include-heavy 后才会执行，并且只使用 num_ctx=4096、并发 1。
- TTFT 是第一个非空 message.content 到达的时间。thinking 字段另记 first_thinking_ms。
- 截断率是 eval_count 达到或超过 num_predict 的样本占该组样本的比例。
- 每组重复多次后取中位数和 P95。P95 用线性插值，位置是 (n - 1) * 0.95。
- 每个模型先预热 1 次。预热请求和它的 load_duration 不计入结果表。
- 生成速度用 eval_count / eval_duration。
- 预填充速度用 prompt_eval_count / prompt_eval_duration。
- bust_cache=true：每次请求在 system 或 user 消息前加一行随机 nonce（`[req:uuid]`），避免 Ollama 复用 KV 缓存。
- Ollama 的实际并行数受 OLLAMA_NUM_PARALLEL 环境变量影响。当前值：未设置。

## 跳过的模型

无。

## 首次加载

| 模型 | num_ctx | load_duration |
| --- | --- | --- |
| qwen3:4b-instruct-2507-q4_K_M | 4096 | 5.347s |

上表只记录每个模型第一次预热读到的 load_duration，不计入正式结果。

## 显存 / 内存

来自 `ollama ps` 的 SIZE 和 PROCESSOR。

| 模型 | SIZE | PROCESSOR |
| --- | --- | --- |
| qwen3:4b-instruct-2507-q4_K_M | 3.2 GB | 100% GPU |

## 结果

| 模型 | 提示词 | num_ctx | 并发 | think | 样本 | 失败率 | 截断率 | TTFT 中位 | TTFT P95 | 总耗时中位 | 总耗时 P95 | 生成 tok/s 中位 | 生成 tok/s P95 | 预填充 tok/s 中位 | 预填充 tok/s P95 | 原始含思考 | 网关已剥离 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3:4b-instruct-2507-q4_K_M | short | 4096 | 1 | false | 3 | 0.0% | 100.0% | 0.224 | 0.228 | 13.248 | 13.554 | 39.3 | 39.7 | 463.2 | 463.6 | 0/3 | — |
| qwen3:4b-instruct-2507-q4_K_M | short | 4096 | 1 | true | 3 | 0.0% | 100.0% | 0.212 | 0.218 | 12.263 | 14.460 | 42.5 | 42.6 | 465.3 | 476.7 | 0/3 | — |
| qwen3:4b-instruct-2507-q4_K_M | long | 4096 | 1 | false | 3 | 0.0% | 0.0% | 3.904 | 3.936 | 4.844 | 4.929 | 37.4 | 39.8 | 585.5 | 596.4 | 0/3 | — |
| qwen3:4b-instruct-2507-q4_K_M | long | 4096 | 1 | true | 3 | 0.0% | 0.0% | 3.968 | 4.009 | 4.870 | 4.903 | 35.8 | 38.0 | 577.6 | 579.7 | 0/3 | — |

## 观察

- think=true 的样本里，原始输出没有 `<think>`，也没有 Ollama 的 thinking 字段。网关没有需要剥离的思考段。
- 提示词 short 的 eval_count 达到 num_predict=512，这些回答在生成上限处停下。

## 结论

在 [Apple M3 Pro / 18 GB] 上，样本不足，暂不下结论。
