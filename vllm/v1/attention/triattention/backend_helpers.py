"""Backend integration helpers for TriAttention V3.

These helpers live here (rather than inside `turboquant_attn.py`) so the
backend file stays focused on TQ kernels and the V3 surface area lives
in one module. Two entry points:

  - `accumulate_prefill_k(layer_name, k_cached, cached_len)`: pushes one
    attention layer's dequant'd cached K into the engine. Called from
    inside `_continuation_prefill` after the K cache has been dequant'd.
    Auto-detects pass boundaries by tracking which layer indices have
    already been seen this round, so the policy fires on the first layer
    of the next pass without needing an explicit signal from the runtime.

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
