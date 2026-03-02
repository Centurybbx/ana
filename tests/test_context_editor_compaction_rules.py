from __future__ import annotations

from ana.context_manager.editor import ContextEditor


def test_editor_compacts_long_tool_outputs_with_traceable_op():
    editor = ContextEditor()
    messages = [
        {"role": "user", "content": "start"},
        {"role": "tool", "name": "web_fetch", "content": "L" * 3000},
        {"role": "assistant", "content": "thanks"},
        {"role": "user", "content": "final"},
    ]
    result = editor.compact(messages=messages, soft_budget_tokens=120, hard_budget_tokens=160)
    assert result.messages
    compacted_tool = [item for item in result.messages if item.get("role") == "tool"][0]
    assert len(str(compacted_tool.get("content", ""))) < 600
    assert any(op.kind == "long_tool_summary" for op in result.operations)

