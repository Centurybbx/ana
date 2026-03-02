# Default recipe - show help
default:
    @just --list

# Install/update dependencies (uv-managed .venv)
install:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync

# Start interactive chat (defaults to config in ~/.ana/config.json)
chat:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run ana

# Doctor ping (connectivity)
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run ana doctor --ping

# Run tests
test:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run pytest -q
