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

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.triattention.hooks import get_engine


_SINGLE_SEQ_ID = 0


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
