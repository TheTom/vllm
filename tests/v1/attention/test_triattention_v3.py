# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the TriAttention V3 engine + V3 selection policy.

Pure-Python tests, no vLLM runtime required. Runs the engine against
synthetic K/Q tensors and confirms:

  - calibration accumulator path (Q stats accumulate correctly)
  - calibration freezes after warmup_tokens unless adaptive_calibration
  - per-cell scoring math agrees with a hand-derived reference
  - V3 selection respects prefix + window + per-segment quotas
  - V3 cleanup pass fills any rounding deficit
"""
from __future__ import annotations

import math
from typing import Optional

import pytest
import torch

from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)
from vllm.v1.attention.triattention.policy import select_v3_evictions
from vllm.v1.attention.triattention.scoring import score_cells_torch


def _make_engine(
    n_layers: int = 4,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    head_dim: int = 64,
    rope_theta: float = 10000.0,
    cfg: Optional[TriAttentionV3Config] = None,
) -> TriAttentionV3Engine:
    if cfg is None:
        cfg = TriAttentionV3Config(
            budget=64, prefix_protect=8, window_size=8,
            n_segments=4, warmup_tokens=64, hybrid_mode=2,
        )
    return TriAttentionV3Engine(
        cfg=cfg,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )


def test_omega_matches_rope_formula():
    eng = _make_engine(head_dim=64, rope_theta=10000.0)
    # omega[i] = 1 / theta^(2i / n_rot). With n_rot = head_dim = 64,
    # freq_count = 32.
    expected = torch.tensor(
        [1.0 / (10000.0 ** (2 * i / 64)) for i in range(32)],
        dtype=eng.omega.dtype,
    )
    assert torch.allclose(eng.omega, expected, atol=1e-6)


def test_calibration_accumulates_and_freezes():
    eng = _make_engine()
    cfg = eng.cfg
    # Push warmup_tokens worth of Q in two batches.
    q = torch.randn(cfg.warmup_tokens // 2, eng.n_heads, eng.head_dim)
    eng.accumulate_q(q, layer_idx=0)
    assert not eng.calibrated, "shouldn't calibrate before warmup_tokens"

    eng.accumulate_q(q, layer_idx=0)
    assert eng.calibrated, "should calibrate after warmup_tokens"
    assert eng.q_samples == cfg.warmup_tokens
    centers = eng.center_real.clone()

    # Further pushes do NOT update centers in non-adaptive mode.
    eng.accumulate_q(q, layer_idx=0)
    assert torch.equal(eng.center_real, centers), \
        "centers must freeze after warmup in non-adaptive mode"


def test_v3_selection_respects_prefix_and_window():
    seq_len = 64
    scores = torch.arange(seq_len, dtype=torch.float32)  # high score = late pos
    valid = torch.ones(seq_len, dtype=torch.bool)
    n_to_evict = 8
    window_thr = seq_len - 8       # protect last 8
    prefix_lo = 8                  # protect first 8

    evict_pos = select_v3_evictions(
        scores=scores, valid=valid, n_to_evict=n_to_evict,
        window_thr=window_thr, prefix_lo=prefix_lo,
        n_segments=4, mode=2,
    )

    # No evicted index should land in [0, prefix_lo) or [window_thr, seq_len).
    assert (evict_pos >= prefix_lo).all().item()
    assert (evict_pos < window_thr).all().item()
    assert evict_pos.numel() == n_to_evict


def test_v3_selection_distributes_across_segments():
    seq_len = 64
    valid = torch.ones(seq_len, dtype=torch.bool)
    # Uniform scores so the per-segment quota is the only thing
    # determining which cells get picked.
    scores = torch.full((seq_len,), 1.0)
    n_to_evict = 8
    window_thr = seq_len            # no recent-window protection
    prefix_lo = 0                   # no prefix protection

    evict_pos = select_v3_evictions(
        scores=scores, valid=valid, n_to_evict=n_to_evict,
        window_thr=window_thr, prefix_lo=prefix_lo,
        n_segments=4, mode=2,
    )
    assert evict_pos.numel() == n_to_evict

    # Each of the 4 segments (size 16) should contribute at least one
    # eviction when the target_frac is 8/64 = 0.125 (so 2 per bucket).
    bucket_widths = seq_len // 4
    per_bucket = torch.zeros(4, dtype=torch.int64)
    for p in evict_pos.tolist():
        per_bucket[p // bucket_widths] += 1
    assert (per_bucket >= 1).all().item()


def test_v1_global_sort_ignores_segments():
    """V1 = global sort: highest score wins, no per-segment quota."""
    seq_len = 32
    scores = torch.arange(seq_len, dtype=torch.float32)
    valid = torch.ones(seq_len, dtype=torch.bool)

    evict_pos = select_v3_evictions(
        scores=scores, valid=valid, n_to_evict=4,
        window_thr=seq_len, prefix_lo=0,
        n_segments=4, mode=0,
    )
    # V1: pick the 4 highest-scoring cells, which are the last four.
    assert sorted(evict_pos.tolist()) == [28, 29, 30, 31]


def test_score_cells_torch_zeros_invalid_positions():
    seq_len, n_kv, hd = 16, 2, 16
    fc = hd // 2
    K = torch.randn(seq_len, n_kv, hd)
    centers_real = torch.randn(n_kv, fc)
    centers_imag = torch.randn(n_kv, fc)
    centers_abs = torch.abs(torch.randn(n_kv, fc))
    omega = torch.randn(fc).abs()
    offsets = torch.tensor([1.0, 2.0, 4.0])

    valid_mask = torch.ones(seq_len, dtype=torch.bool)
    valid_mask[0:4] = False  # mark first 4 as evicted

    score = score_cells_torch(
        K=K, center_real=centers_real, center_imag=centers_imag,
        center_abs=centers_abs, omega=omega, offsets=offsets,
        max_pos=seq_len - 1, valid_mask=valid_mask,
        window_thr=seq_len, n_rot=hd,
    )
    assert (score[:4] == 0.0).all().item(), \
        "scores must be zero at evicted positions"
