"""Backend integration helpers for TriAttention V3.

These helpers live here (rather than inside `turboquant_attn.py`) so the
backend file stays focused on TQ kernels and the V3 surface area lives
in one module. Three entry points:

  - `accumulate_prefill_k(layer_name, k_cached, cached_len)`: pushes one
    attention layer's dequant'd cached K into the engine. Called from
    inside `_continuation_prefill` after the K cache has been dequant'd.
    Auto-detects pass boundaries by tracking which layer indices have
    already been seen this round, so the policy fires on the first layer
    of the next pass without needing an explicit signal from the runtime.

  - `maybe_finalize_evict(layer_name)`: explicitly trigger eviction at
    end-of-pass. The duplicate-layer detection in `accumulate_prefill_k`
    only fires `finalize_evict_round` when a SECOND pass starts. For
    single-pass prefill (NIAH-style: one big forward, then decode kernel
    takes over) the second pass never comes, so eviction never fires.
    Call this after the last attention layer of prefill to force the
    policy to run on the accumulated scores.

  - `build_valid_mask(common_attn_metadata)`: returns the uint8 [B, S]
    validity tensor that the TurboQuant decode kernels consume via the
    VALID_MASK constexpr. Returns None when V3 is not engaged.

Phase A scope is single-sequence batches (seq_id=0). Multi-batch needs
request-id plumbing through `CommonAttentionMetadata` and is deferred.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.triattention.hooks import get_engine

logger = init_logger(__name__)


_SINGLE_SEQ_ID = 0


# ---------------------------------------------------------------------------
# Tier 2 + Tier 3: longctx-svc evict-to-vector / rehydrate hooks
# Design: TriAttention V3 — 3-Tier Eviction Rescue Architecture (obsidian).
# ---------------------------------------------------------------------------


# Per-session prompt token IDs (set by the prefill hook so the eviction
# callback can decode evicted positions back to text). Replaced with each
# new prefill of the same session id.
_PROMPT_TOKEN_IDS: dict[int, list[int]] = {}
# Per-session tokenizer reference (lazy-bound on first prefill capture).
_TOKENIZER = None
# Per-session id ↔ longctx session id mapping. Phase A is single-batch
# (all V3 work happens at seq_id=0), so we store a single string here.
_LONGCTX_SESSION_ID: str = "v3-single-session"
# Tier 2 endpoint config. Read once from env at first use.
_LONGCTX_BASE_URL: Optional[str] = None
_LONGCTX_HTTP = None  # requests.Session, lazy-built


def set_prompt_token_ids(seq_id: int, token_ids: list[int]) -> None:
    """Stash the prompt's token IDs for a session so the eviction
    callback can decode evicted positions back to text. Called from
    the prefill hook with the freshly-tokenized prompt.
    """
    _PROMPT_TOKEN_IDS[int(seq_id)] = list(token_ids)


def set_tokenizer(tok) -> None:
    """Bind the model's tokenizer once at engine init. Used by the
    eviction callback to decode evicted positions back to text."""
    global _TOKENIZER
    _TOKENIZER = tok


def set_longctx_session_id(session_id: str) -> None:
    """Set the longctx-svc session id used by /evict/write + /retrieve.
    Default is `v3-single-session`; per-request batches will need
    request-id plumbing in a future phase.
    """
    global _LONGCTX_SESSION_ID
    _LONGCTX_SESSION_ID = str(session_id)


def _get_longctx_base_url() -> Optional[str]:
    global _LONGCTX_BASE_URL
    if _LONGCTX_BASE_URL is None:
        _LONGCTX_BASE_URL = os.environ.get("LONGCTX_ENDPOINT")
    return _LONGCTX_BASE_URL


def _get_http_session():
    global _LONGCTX_HTTP
    if _LONGCTX_HTTP is None:
        import requests
        _LONGCTX_HTTP = requests.Session()
    return _LONGCTX_HTTP


def _evict_to_longctx_callback(
    seq_id: int, evict_pos: torch.Tensor, n_evicted: int,
) -> None:
    """V3 engine's eviction callback. Decodes evicted positions back to
    text spans (via the bound tokenizer + the stashed prompt token IDs)
    and POSTs to longctx-svc /evict/write.

    Span grouping: contiguous runs of evicted positions are merged into
    single spans (avoids one-chunk-per-token). Spans are decoded with a
    ±32 token bleed for context (so the chunk doesn't start mid-sentence
    and the embedder sees coherent text).

    No-op when LONGCTX_ENDPOINT is unset, when the tokenizer hasn't been
    bound yet, or when no token IDs are stashed for this session.
    """
    base = _get_longctx_base_url()
    if not base:
        return
    if _TOKENIZER is None:
        logger.warning(
            "TriAttention V3 eviction callback: tokenizer not bound; "
            "cannot decode evicted positions. Call set_tokenizer() at "
            "engine init."
        )
        return
    token_ids = _PROMPT_TOKEN_IDS.get(int(seq_id))
    if not token_ids:
        # Common case: short prefill, no token IDs stashed yet (e.g.
        # tool-call probes). Skip silently.
        return

    # Sort + group contiguous positions
    pos_cpu = sorted(p for p in evict_pos.detach().cpu().tolist()
                     if 0 <= p < len(token_ids))
    if not pos_cpu:
        return
    spans: list[tuple[int, int]] = []
    cur_start = pos_cpu[0]
    cur_end = pos_cpu[0]
    for p in pos_cpu[1:]:
        if p == cur_end + 1:
            cur_end = p
        else:
            spans.append((cur_start, cur_end))
            cur_start = p
            cur_end = p
    spans.append((cur_start, cur_end))

    # Expand each span ±32 tokens for context bleed; clamp to bounds.
    BLEED = 32
    chunks_payload: list[dict] = []
    for (s, e) in spans:
        ws = max(0, s - BLEED)
        we = min(len(token_ids), e + 1 + BLEED)
        try:
            text = _TOKENIZER.decode(
                token_ids[ws:we], skip_special_tokens=True,
            )
        except Exception:  # noqa: BLE001
            continue
        if not text.strip():
            continue
        chunks_payload.append({
            "text": text,
            "token_range": (int(ws), int(we)),
            "layer": -1,  # multi-layer eviction; we don't track layer here
            "score": 0.0,  # score not surfaced through callback yet
        })

    if not chunks_payload:
        return

    url = base.rstrip("/") + "/evict/write"
    body = {
        "session_id": _LONGCTX_SESSION_ID,
        "chunks": chunks_payload,
    }
    try:
        # Synchronous POST is fine because finalize_evict_round is itself
        # off the hot decode path (only fires when cache pressure crosses
        # budget). Latency dominated by MiniLM embed on the longctx side
        # (~tens of ms for a few chunks). Async / fire-and-forget is a
        # follow-up optimization.
        sess = _get_http_session()
        sess.post(url, json=body, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TriAttention V3 evict-to-vector POST to %s failed: %s",
            url, exc,
        )


def install_eviction_to_longctx() -> bool:
    """Register the Tier 2 callback on the global V3 engine. Idempotent.
    Returns True when wired, False when prerequisites are missing
    (engine not yet initialized, or LONGCTX_ENDPOINT unset).
    """
    eng = get_engine()
    if eng is None:
        return False
    if not _get_longctx_base_url():
        return False
    eng.set_eviction_callback(_evict_to_longctx_callback)
    logger.info(
        "TriAttention V3 evict-to-vector enabled — POSTing evicted spans "
        "to %s/evict/write (session_id=%s)",
        _LONGCTX_BASE_URL, _LONGCTX_SESSION_ID,
    )
    return True


def retrieve_evicted_for_query(
    query: str, top_k: int = 8, score_floor: float = 0.0,
) -> list[dict]:
    """Tier 3: retrieve evicted spans relevant to the user's current
    query. Called by a prefill hook before each new turn. Returns a
    list of {text, token_range, layer, score} dicts; empty list when
    longctx is unconfigured or has no evictions for this session.
    """
    base = _get_longctx_base_url()
    if not base:
        return []
    url = base.rstrip("/") + "/evict/retrieve"
    body = {
        "session_id": _LONGCTX_SESSION_ID,
        "query": query,
        "top_k": int(top_k),
        "score_floor": float(score_floor),
    }
    try:
        sess = _get_http_session()
        r = sess.post(url, json=body, timeout=10.0)
        if r.status_code != 200:
            return []
        data = r.json()
        return list(data.get("chunks") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TriAttention V3 evict-retrieve from %s failed: %s", url, exc,
        )
        return []


def accumulate_prefill_k(
    layer_name: str,
    k_cached: torch.Tensor,
    cached_len: int,
) -> None:
    """Push one layer's dequant'd K to the V3 engine. No-op when V3 off.

    Pass boundaries are detected by watching for a layer index that's
    already been accumulated this round. That naturally tolerates the
    boundary layers that bypass the TQ backend (FP16-stored first/last
    attention layers don't fire this hook, and the engine just doesn't
    score them).
    """
    eng = get_engine()
    if eng is None or not eng.calibrated or cached_len == 0:
        return

    layer_il = extract_layer_index(layer_name)
    seq_id = _SINGLE_SEQ_ID
    seq_len = cached_len

    st = eng._seq_state.get(seq_id)
    valid = eng.get_valid_mask(seq_id, seq_len, k_cached.device)

    pending_layers = st.get("pending_layers", set()) if st else set()
    if (
        st is not None
        and "pending_scores" in st
        and layer_il in pending_layers
    ):
        eng.finalize_evict_round(seq_id)
        st = eng._seq_state.get(seq_id)
        pending_layers = set()

    needs_open = (
        st is None
        or "pending_scores" not in st
        or st["pending_scores"].shape[0] != seq_len
    )
    if needs_open:
        eng.begin_score_round(seq_id, seq_len, k_cached.device)

    positions = torch.arange(seq_len, dtype=torch.int32, device=valid.device)
    live_pos = positions[valid]
    if live_pos.numel() == 0:
        return
    max_pos = int(live_pos.max().item())
    window_thr = max_pos - eng.cfg.window_size + 1

    eng.accumulate_layer_score(seq_id, layer_il, k_cached, max_pos, window_thr)
    st = eng._seq_state[seq_id]
    if "pending_layers" in st:
        st["pending_layers"].add(layer_il)


def maybe_finalize_evict(layer_name: str | None = None) -> int:
    """Explicit end-of-pass trigger for V3's policy.

    Returns the number of positions evicted (0 when V3 is off, not
    calibrated, no pending scores, or cache size below the budget gate).

    Designed to be called once per prefill / continuation chunk after
    the last attention layer's K has been pushed via
    `accumulate_prefill_k`. The duplicate-layer detection inside
    `accumulate_prefill_k` is fine for chunked / multi-pass prefill but
    misses the single-pass case (typical for NIAH-style benches: one
    forward over the whole prompt, then small decode chunks via a
    different kernel). Calling this after each pass is closes that gap.

    Idempotent: safe to call from multiple sites; it only finalizes when
    there are pending scores AND `should_evict` says budget is exceeded.
    """
    eng = get_engine()
    if eng is None or not eng.calibrated:
        return 0
    seq_id = _SINGLE_SEQ_ID
    st = eng._seq_state.get(seq_id)
    if st is None or "pending_scores" not in st:
        return 0
    seq_len = int(st.get("pending_seq_len", 0))
    if seq_len <= 0:
        return 0
    if not eng.should_evict(seq_id, seq_len):
        # Cache below budget — accumulated scores stay open for the next
        # pass; no point firing the policy with nothing to evict.
        return 0
    # Only fire when ALL expected attention layers have contributed to
    # the pending score buffer. Otherwise we'd run the policy on a
    # partial sum (layers 1..N missing) and the per-segment quota
    # would behave erratically. `n_layers - boundary_skip` is the count
    # of layers expected to fire `accumulate_layer_score`; the boundary-
    # skip layers are the ones the engine refuses to score by config.
    expected_layers = max(1, eng.n_layers - eng.cfg.boundary_skip)
    seen_layers = len(st.get("pending_layers", set()))
    if seen_layers < expected_layers:
        return 0
    return eng.finalize_evict_round(seq_id)


def build_valid_mask(common_attn_metadata) -> torch.Tensor | None:
    """Build the per-(batch, position) validity mask for the TQ kernels.

    Returns a uint8 [B, max_seq_len] tensor where 1=live, 0=evicted, or
    None if V3 is not engaged or the batch shape isn't supported yet.
    """
    eng = get_engine()
    if eng is None or not eng.calibrated:
        return None
    seq_lens = common_attn_metadata.seq_lens
    if int(seq_lens.shape[0]) != 1:
        # V3 multi-batch path not yet implemented; safe fallback (all live).
        return None
    seq_len = int(common_attn_metadata.max_seq_len)
    device = seq_lens.device
    valid_bool = eng.get_valid_mask(
        seq_id=_SINGLE_SEQ_ID, seq_len=seq_len, device=device
    )
    return valid_bool.to(torch.uint8).contiguous().unsqueeze(0)
