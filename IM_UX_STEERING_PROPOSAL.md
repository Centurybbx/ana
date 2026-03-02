# Telegram IM UX And Implicit Steer Proposal (Implemented)

Status: implemented. This document captures the product/architecture decisions that are now reflected in runtime behavior.

## Goals

- Improve Telegram waiting experience so users can see progress while a turn is running.
- Support steer without control commands: new user messages during execution should naturally redirect the agent.
- Preserve context integrity during steer: no history loss and no stale output overriding latest output.

## Non-goals (Kept)

- No `/stop` or `/steer` command design.
- No change to core ReAct semantics or ToolPolicy semantics.
- No additional implementation in this document; this serves as an implemented design record.

## Telegram UX Capabilities

1. Use `sendChatAction("typing")` while processing.
2. Use placeholder message + `editMessageText` updates for streaming-like perception.
3. Throttle edit frequency and degrade to `send_message` on edit failures/rate limits.
4. Keep existing chunking for long text, while preferring one editable "main" message.

## Implicit Steer Model (IM-native)

When a new message arrives for the same chat/thread while a turn is in-flight:

1. Treat it as steer automatically (no command).
2. Cancel current turn cooperatively.
3. Mark prior in-flight output as superseded by newer user input.
4. Start a new turn from latest input (`latest-wins`), preserving session continuity.

## Context Integrity Requirements

1. Full user history is retained in `session.messages`, including messages that triggered interruption.
2. Every turn gets a `generation_id`; only latest generation may publish user-visible output.
3. Canceled turn stores a lightweight snapshot (completed steps/tools + cancel reason) for continuity.
4. Outbound path is idempotent (`generation_id + message_ref`) to avoid duplicate or stale delivery.

## Product Entry Strategy

- `ana serve` / `ana im` is the primary collaborative IM entrypoint.
- `ana chat` moves to a deprecation path (compatibility period allowed).

## Suggested Observability Events

- `im_typing_start`, `im_typing_stop`
- `im_stream_placeholder_sent`, `im_stream_edit_tick`
- `im_implicit_steer`, `im_turn_cancelled_by_new_input`
- `im_generation_drop_stale_output`
- `im_context_snapshot_saved`
