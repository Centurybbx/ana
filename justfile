# Default recipe - show help
default:
    @just --list

# Install/update dependencies (uv-managed .venv)
install:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync

# Start interactive chat (defaults to config in ~/.aha/config.json)
chat:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run aha

# Doctor ping (connectivity)
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run aha doctor --ping

# Run tests
test:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run pytest -q
