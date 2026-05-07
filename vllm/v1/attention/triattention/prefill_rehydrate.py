"""Tier 3 prefill hook for the V3 + longctx rescue stack.

Two responsibilities, both gated on `LONGCTX_ENDPOINT` being set:

1. Stash the prompt's token IDs into the V3 engine state via
   `set_prompt_token_ids`, so the eviction callback (Tier 2 in
   `backend_helpers.py`) can decode evicted positions back to text on
   each subsequent eviction round.

2. Retrieve top-K evicted spans relevant to the user's most recent
   message via `retrieve_evicted_for_query`, and prepend them as a
   leading system message into the request's `messages` array. The
   model sees recovered context BEFORE prefill runs.

Both fire from `OpenAIServingChat.create_chat_completion`, just before
`render_chat_request` tokenises the prompt. No-op when V3 isn't enabled
or longctx isn't configured — the hook is safe to leave installed
permanently.

Design lives in obsidian:
  TriAttention V3 — 3-Tier Eviction Rescue Architecture (Tier 3)
"""
from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


_TOP_K = int(os.environ.get("VLLM_TRIATT_RESCUE_TOPK", "8"))
_SCORE_FLOOR = float(os.environ.get("VLLM_TRIATT_RESCUE_FLOOR", "0.20"))
_MAX_INJECTED_CHARS = int(
    os.environ.get("VLLM_TRIATT_RESCUE_MAX_CHARS", "8000")
)


def _last_user_text(messages: list[dict] | list[Any]) -> str | None:
    """Walk the messages array end-to-start; return the first user
    message's text. None when there's no user message."""
    if not messages:
        return None
    for msg in reversed(list(messages)):
        # Messages can be Pydantic models or plain dicts depending on
        # where this hook fires from. Support both.
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str):
            return content
        # Multi-modal content: list of {type: text, text: ...} parts.
        if isinstance(content, list):
            parts = []
            for c in content:
                t = getattr(c, "type", None) or (
                    c.get("type") if isinstance(c, dict) else None
                )
                if t == "text":
                    txt = getattr(c, "text", None) or (
                        c.get("text") if isinstance(c, dict) else None
                    )
                    if txt:
                        parts.append(txt)
            if parts:
                return "\n".join(parts)
        return None
    return None


def _format_rehydrate_system_message(chunks: list[dict]) -> str:
    """Render retrieved evicted chunks as a single system-message string.

    Caps total characters to avoid blowing up prompt size on agentic
    workloads. Layer / token-range metadata is included as inline
    comments — useful for debugging, ignored by the model.
    """
    header = (
        "[Recovered context from earlier in this session "
        "(evicted from KV cache, restored via longctx)]:\n"
    )
    body_parts: list[str] = []
    used = len(header)
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            continue
        tr = c.get("token_range") or (0, 0)
        layer = c.get("layer", -1)
        block = (
            f"\n--- evicted span (tokens {tr[0]}..{tr[1]}, "
            f"layer {layer}) ---\n{text}\n"
        )
        if used + len(block) > _MAX_INJECTED_CHARS:
            break
        body_parts.append(block)
        used += len(block)
    if not body_parts:
        return ""
    return header + "".join(body_parts)


def maybe_rehydrate_messages(
    messages: list[dict] | list[Any],
) -> tuple[list, int]:
    """If longctx is configured AND the session has evicted spans,
    prepend a system message with rescued context. Returns the
    (possibly-mutated) messages list + the number of chunks injected.

    Safe no-op when:
      * LONGCTX_ENDPOINT unset
      * No user message in the conversation yet
      * Session has no evictions
      * Retrieval call fails (logged + swallowed)
    """
    if not os.environ.get("LONGCTX_ENDPOINT"):
        return list(messages), 0
    user_text = _last_user_text(messages)
    if not user_text:
        return list(messages), 0
    # Lazy-import to avoid pulling backend_helpers at module load
    from vllm.v1.attention.triattention.backend_helpers import (
        retrieve_evicted_for_query,
    )
    try:
        chunks = retrieve_evicted_for_query(
            user_text, top_k=_TOP_K, score_floor=_SCORE_FLOOR,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TriAttention V3 rescue: /evict/retrieve failed: %s — falling "
            "back to no-rehydrate prefill.", exc,
        )
        return list(messages), 0
    if not chunks:
        return list(messages), 0
    sys_text = _format_rehydrate_system_message(chunks)
    if not sys_text:
        return list(messages), 0
    out = [{"role": "system", "content": sys_text}, *list(messages)]
    logger.info(
        "TriAttention V3 rescue: injected %d evicted chunks (%d chars) "
        "into prefill as a leading system message.",
        len(chunks), len(sys_text),
    )
    return out, len(chunks)


def stash_prompt_token_ids(prompt_token_ids: list[int] | None) -> None:
    """Capture the tokenised prompt for the V3 eviction callback so it
    can decode evicted positions back to text. Phase A is single-batch
    (seq_id=0); future multi-batch will require request-id plumbing."""
    if not prompt_token_ids:
        return
    if not os.environ.get("LONGCTX_ENDPOINT"):
        return
    # Avoid pulling backend_helpers at module load; lazy-import here.
    from vllm.v1.attention.triattention.backend_helpers import (
        set_prompt_token_ids,
    )
    set_prompt_token_ids(0, list(prompt_token_ids))
