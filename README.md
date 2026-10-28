# wecom-local-llm

基于本地大模型（Ollama）的企业微信智能助手。当前仓库完成了 M1：服务骨架和模型网关。群机器人、企微回调、知识库、工具调用和模拟器在后续里程碑实现。

## 环境

- Python 3.11+
- 可选：本机 [Ollama](https://ollama.com/)（默认 `http://localhost:11434`）

## 安装

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

模型名只放在 `.env` 的 `LLM_MODEL`。默认值是 `qwen3.8:27b`。本机没有这个标签时，改成已经拉取的模型，例如 `qwen3:4b`。

密钥只从 `.env` 读取。`.gitignore` 排除了 `.env` 和数据库文件。

## 启动

`APP_PORT` 默认 `8100`。直接启动时使用 `.env` 里的 `APP_HOST` 和 `APP_PORT`：

```bash
python -m app.main
```

也可以把端口写在命令里：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8100
```

健康检查：

```bash
curl http://127.0.0.1:8100/health
```

`/health` 只返回配置摘要，不会请求模型。

## 命令行提问

```bash
python -m app.llm.cli "用一句话介绍你自己"
python -m app.llm.cli --stream "你好"
python -m app.llm.cli --model qwen3:4b "你好"
```

`--model` 只影响这一次运行，不写回 `.env`。安装后也可以用 `wecom-llm "你好"`。

输出正文之后会再打印模型名、总耗时、输出 token 数。流式调用能测到首 token 时，一并打印首 token 耗时。

## 模型网关

`LLM_PROVIDER` 取 `ollama` 或 `openai`。

- `ollama` 调用 `{LLM_BASE_URL}/api/chat`，并提交 `num_ctx`、`temperature`、`keep_alive` 和 `think`。`LLM_THINK` 默认 `false`。
- 网关会去掉回答里的 `<think>...</think>`。标签没有闭合时，从 `<think>` 一直到结尾都去掉。
- `openai` 调用 OpenAI 兼容的 `/chat/completions`。根地址会补上 `/v1`；已经带路径的地址保持不变，例如智谱的 `/api/paas/v4`。请求里提交 `model` 和 `temperature`。`num_ctx`、`keep_alive`、`think` 不放进兼容接口的 JSON。
- `LLM_FALLBACK_ENABLED` 默认 `false`，这时即使填了备用提供方也不会切换。设为 `true` 后，主模型遇到连接失败、超时、HTTP 429、HTTP 5xx 或空响应，会切换一次备用模型，并写一条带原因的 WARN 日志。其它 4xx 不切换。
- `stream=True` 且同时传入工具时，网关抛出 `NotImplementedError`。工具调用走非流式。

备用模型的名字或地址留空时，沿用主模型的对应值。API Key 不沿用。

流式调用的用法是先 `await chat(..., stream=True)`，再 `async for`。已经开始输出之后失败，不再切换备用模型。

## 压测

`bench/bench_ollama.py` 只在本机手动运行，不进 CI。参数来自 `bench/config.yaml`，命令行可以覆盖。未拉取的模型会跳过，脚本不会执行 `ollama pull`。

```bash
python -m bench.bench_ollama \
  --models qwen3:4b-instruct-2507-q4_K_M \
  --num-ctx 4096 \
  --concurrency 1
```

原始结果写到 `bench/results/`（已忽略）。汇总写到 `docs/bench_report.md`。

默认模型是 `qwen3:4b-instruct-2507-q4_K_M`、`qwen3:8b`、`qwen3:14b`。`qwen3.8:27b` 写在 `heavy_models`，加上 `--include-heavy` 才会跑，并且固定 `num_ctx=4096`、并发 1。`num_predict` 默认 512。`bust_cache` 默认打开，每次请求在 system 或 user 消息前加一行 `[req:uuid]`。TTFT 取第一个非空正文；thinking 字段另记，不计入 TTFT。结果表有截断率，等于 `eval_count` 达到 `num_predict` 的比例。只有至少两个模型有成功样本时才写推荐，否则结论是「样本不足，暂不下结论」。

## 测试

```bash
ruff check .
ruff format --check .
pytest
```

单测使用模拟的 HTTP 响应，不访问本机 Ollama。覆盖率门槛是 70%。压测的解析和统计也在这批单测里，用模拟的 Ollama 响应。

## 许可证

MIT。见 [LICENSE](LICENSE)。

当前版本只提供模型调用，不提供医美建议。后续示例数据会标明为虚构内容。
