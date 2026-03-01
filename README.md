# aha (MVP)

Minimal ReAct agent implementation with policy-guarded tools and LiteLLM provider support.

## Quickstart

```bash
uv sync
uv run aha chat --provider mock --prompt "hello"
```

## Real Model (LiteLLM)

Create `~/.aha/config.json`:

```json
{
  "provider": "litellm",
  "model": "openai/your-model-name",
  "endpoint": "https://your-llm-endpoint.example.com/v1",
  "api_key_env": "AHA_LLM_API_KEY",
  "request_timeout_seconds": 60
}
```

Set env var and run doctor:

```bash
export AHA_LLM_API_KEY="your-api-key"
uv run aha doctor --ping
```

Start chat:

```bash
uv run aha
```

## Runtime Logs

Runtime logs are separate from `memory/TRACE.jsonl`:
- `logs/aha.log`: runtime events for normal mode (`INFO` by default)
- `logs/aha.debug.log`: runtime events for debug mode (`DEBUG` by default)
- `memory/TRACE.jsonl`: agent trace/audit stream

Enable debug runtime logging:

```bash
uv run aha --debug
# or
uv run aha chat --debug
```

Override runtime logging:

```bash
uv run aha --log-level DEBUG --log-dir ./logs
uv run aha --no-runtime-log
```

View logs:

```bash
tail -n 200 logs/aha.log
tail -f logs/aha.debug.log
```

## Repo Hygiene

This repo keeps only source code and tests in Git by default. The following are local runtime artifacts and are ignored:
- `doc/` (local design/notes)
- `logs/`
- `memory/`
- `sessions/`
- `skills_local/`

Key resolution priority for `litellm`:
- `api_key` (direct in config) > `api_key_env` > no key
- No key is allowed for local/anonymous endpoints.

OpenRouter example:

```json
{
  "provider": "litellm",
  "model": "openrouter/anthropic/claude-3.5-sonnet",
  "api_key_env": "OPENROUTER_API_KEY",
  "endpoint": "https://openrouter.ai/api/v1"
}
```

## MVP Notes

- `ToolResult` now uses `data/redactions` as primary fields.
- Compatibility fields `output/meta` are still returned for MVP stability.
- `/new` slash command is supported inside `aha chat`.
- `/dry-run` was explicitly deferred and is not implemented in this MVP.

## Without uv (fallback)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
aha
```
