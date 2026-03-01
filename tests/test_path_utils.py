from __future__ import annotations

from aha.tools.path_utils import is_within, is_within_any, resolve_candidate


def test_resolve_candidate_relative_and_absolute(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    relative = resolve_candidate("a/b.txt", workspace)
    absolute = resolve_candidate(str(tmp_path / "outside.txt"), workspace)

    assert str(relative).endswith("/workspace/a/b.txt")
    assert absolute == (tmp_path / "outside.txt").resolve()


def test_is_within_and_is_within_any(tmp_path):
    workspace = tmp_path / "workspace"
    skills_local = workspace / "skills_local"
    workspace.mkdir()
    skills_local.mkdir(parents=True)

    inside_workspace = (workspace / "main.py").resolve()
    inside_skills = (skills_local / "demo/SKILL.md").resolve()
    outside = (tmp_path / "outside.txt").resolve()

    assert is_within(inside_workspace, workspace)
    assert not is_within(outside, workspace)
    assert is_within_any(inside_skills, [workspace, skills_local])
    assert not is_within_any(outside, [workspace, skills_local])
