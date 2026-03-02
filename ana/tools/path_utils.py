from __future__ import annotations

from pathlib import Path


def resolve_candidate(path_str: str, workspace: Path) -> Path:
    base = workspace.resolve()
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_within_any(path: Path, roots: list[Path]) -> bool:
    return any(is_within(path, root) for root in roots)
