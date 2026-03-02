# ANA (AI Native Agent)

ANA is an **AI Native Agent** CLI built around a minimal ReAct loop, policy-guarded tools, and eval-driven self-improvement.

## What it does

- Runs interactive or one-shot agent chats
- Supports `mock` and `litellm` providers
- Provides controlled tools (`read_file`, `write_file`, `shell`, `web_search`, etc.)
- Supports IM channels (Telegram / Discord)
- Includes offline eval and regression gates
- Supports a proposal-based evolution workflow (analyze -> propose -> validate -> deploy)

## Quickstart

```bash
uv sync
uv run ana chat --provider mock --prompt "hello"
```

## Configure a real model

Create `~/.ana/config.json`:

```json
{
  "provider": "litellm",
  "model": "openai/your-model-name",
  "endpoint": "https://your-llm-endpoint.example.com/v1",
  "api_key_env": "ANA_LLM_API_KEY"
}
```

Set key and check connectivity:

```bash
export ANA_LLM_API_KEY="your-api-key"
uv run ana doctor --ping
```

Start chat:

```bash
uv run ana
```

## IM runtime (Telegram / Discord)

```bash
uv sync --extra im
uv run ana serve
# alias:
uv run ana im
```

Channel config is under `~/.ana/config.json` in the `channels` field.

UX and implicit-steer design:
- `IM_UX_STEERING_PROPOSAL.md`

## Core commands

```bash
# Chat
uv run ana chat

# Diagnostics
uv run ana doctor --ping

# Evals
uv run ana eval-export --output memory/eval.dataset.jsonl
uv run ana eval-metrics --output memory/eval.metrics.json
uv run ana eval-run --provider mock --tasks benchmarks/core.json

# Evolution workflow
uv run ana evolve-analyze --output memory/failure_reports.json
uv run ana evolve-propose --failures memory/failure_reports.json --skill-name planner
uv run ana evolve-validate --proposal-id <id> --tasks benchmarks/core.json --provider mock
uv run ana evolve-deploy --proposal-id <id>
uv run ana evolve-monitor --baseline memory/eval.metrics.baseline.json
uv run ana evolve-rollback --skill-name planner
```

## Runtime logs

- `logs/ana.log`: runtime events in normal mode
- `logs/ana.debug.log`: runtime events in debug mode
- `memory/TRACE.jsonl`: trace/audit stream

Enable debug logs:

```bash
uv run ana --debug
```
