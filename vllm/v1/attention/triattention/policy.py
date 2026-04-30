"""V3 selection: protected window + prefix + per-segment quota + cleanup.

Mirrors the eviction-selection block of `llama_triattention::evict()`. The
score input is "high score = evict me first" (the trig formula measures
orthogonality, not alignment).

V1 (mode=0): global sort, evict highest-scoring N cells (excluding window).
V2 (mode=1): per-segment quota (no prefix protection).
V3 (mode=2): per-segment quota + first prefix_protect tokens are protected.
"""
from __future__ import annotations

import torch


def select_v3_evictions(
    scores: torch.Tensor,    # [seq_len] fp32, higher = evict first
    valid: torch.Tensor,     # [seq_len] bool
    n_to_evict: int,
    window_thr: int,
    prefix_lo: int,
    n_segments: int,
    mode: int = 2,
) -> torch.Tensor:
    """Return int64 tensor of positions to evict (length <= n_to_evict)."""
    if n_to_evict <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    seq_len = scores.shape[0]
    device = scores.device
    positions = torch.arange(seq_len, dtype=torch.long, device=device)

    # Candidates: valid AND not in protected window AND (V3) past prefix.
    in_window = positions >= window_thr
    in_prefix = positions < prefix_lo
    candidates = valid & ~in_window
    if mode == 2:
        candidates = candidates & ~in_prefix

    candidate_pos = positions[candidates]
    if candidate_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if mode == 0:
        # V1: global sort, take top n_to_evict highest-scoring candidates.
        cand_scores = scores[candidate_pos]
        n_take = min(int(n_to_evict), int(candidate_pos.numel()))
        topk = torch.topk(cand_scores, k=n_take, largest=True)
        return candidate_pos[topk.indices]

    # V2/V3: per-segment quota.
    k = max(1, int(n_segments))
    pos_hi = max(int(window_thr), int(prefix_lo) + 1)
    seg_width = max(1.0, float(pos_hi - prefix_lo) / float(k))

    cand_scores = scores[candidate_pos]               # [N]
    seg_idx = ((candidate_pos.float() - float(prefix_lo)) / seg_width).long()
    seg_idx = seg_idx.clamp_(0, k - 1)

    total_eligible = int(candidate_pos.numel())
    target_frac = float(n_to_evict) / float(total_eligible) if total_eligible > 0 else 0.0

    chosen: list[torch.Tensor] = []
    n_chosen = 0
    for s in range(k):
        bucket_mask = seg_idx == s
        bucket_pos = candidate_pos[bucket_mask]
        bucket_score = cand_scores[bucket_mask]
        if bucket_pos.numel() == 0:
            continue
        bucket_target = int(float(bucket_pos.numel()) * target_frac)
        n_take = min(bucket_target, int(bucket_pos.numel()), n_to_evict - n_chosen)
        if n_take <= 0:
            continue
        topk = torch.topk(bucket_score, k=n_take, largest=True)
        chosen.append(bucket_pos[topk.indices])
        n_chosen += n_take

    # Cleanup: if rounding left a deficit, fill from remaining candidates.
    if n_chosen < n_to_evict:
        if chosen:
            already = torch.cat(chosen)
            already_set = torch.zeros(seq_len, dtype=torch.bool, device=device)
            already_set[already] = True
            remaining_mask = candidates & ~already_set
        else:
            remaining_mask = candidates
        remaining_pos = positions[remaining_mask]
        if remaining_pos.numel() > 0:
            remaining_scores = scores[remaining_pos]
            n_take = min(n_to_evict - n_chosen, int(remaining_pos.numel()))
            topk = torch.topk(remaining_scores, k=n_take, largest=True)
            chosen.append(remaining_pos[topk.indices])

    if not chosen:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(chosen)
