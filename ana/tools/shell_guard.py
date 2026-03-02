from __future__ import annotations

import shlex
from dataclasses import dataclass

ALLOWED_BINARIES = {"ls", "pwd", "cat", "echo", "rg", "find", "git", "pytest", "python", "python3"}
BLOCKED_BINARIES = {"rm", "dd", "mkfs", "chmod", "chown", "shutdown", "reboot", "killall"}
BLOCKED_OPERATORS = ("|", ">", "<", "$(", "`", "&&", "||", ";")
BLOCKED_REMOTE_FRAGMENTS = ("http://", "https://", "curl ", "wget ")


@dataclass
class ShellValidationResult:
    allowed: bool
    reason: str
    parts: list[str]


def validate_shell_command(cmd: str) -> ShellValidationResult:
    normalized = cmd.strip()
    if not normalized:
        return ShellValidationResult(False, "missing cmd", [])

    if any(token in normalized for token in BLOCKED_OPERATORS):
        return ShellValidationResult(False, "shell operators are blocked in MVP policy", [])

    try:
        parts = shlex.split(normalized)
    except ValueError:
        return ShellValidationResult(False, "invalid shell syntax", [])

    if not parts:
        return ShellValidationResult(False, "empty command", [])

    binary = parts[0]
    if binary in BLOCKED_BINARIES:
        return ShellValidationResult(False, f"blocked command: {binary}", parts)
    if binary not in ALLOWED_BINARIES:
        return ShellValidationResult(False, f"not in allowlist: {binary}", parts)
    if binary in {"python", "python3"} and "-c" in parts:
        return ShellValidationResult(False, "python -c is blocked in MVP policy", parts)
    if any(fragment in normalized.lower() for fragment in BLOCKED_REMOTE_FRAGMENTS):
        return ShellValidationResult(False, "shell commands containing remote fetch snippets are blocked", parts)

    return ShellValidationResult(True, "ok", parts)
