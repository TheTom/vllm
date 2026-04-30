"""Triton kernel for V3 cell scoring.

Per-cell math collapses (after the algebraic simplification documented in
docs/papers/triattention-v3.md §4.7) to a pair of dot products:

  acc = sum_f (A_f * cos_sum[f] - B_f * sin_sum[f])
  ext = sum_f cb_delta[f] * k_abs[f]

Where for cell `c` and head `h`:
  A_f = c_r[h, f] * k_r[c, h, f] + c_i[h, f] * k_i[c, h, f]
  B_f = c_i[h, f] * k_r[c, h, f] - c_r[h, f] * k_i[c, h, f]
  k_abs[f] = sqrt(k_r[c, h, f]^2 + k_i[c, h, f]^2 + 1e-8)
  cb_delta[f] = c_abs[h, f] - sqrt(c_r[h, f]^2 + c_i[h, f]^2 + 1e-8)

This is a sum over `fc` (frequency bins) for each (cell, head) pair. The kernel
tiles `[M_CELLS, FC]` so the inner f-axis is FC-sized contiguous, and across
heads we sum sequentially (heads are small, ≤8 in tested models).

On gfx942 the FC dim (typically 64 for Qwen2.5-7B head_dim=128) matches an
MFMA-friendly tile, and M_CELLS≥16 unlocks tl.dot via FMA fallback or MFMA
once we extend tile shapes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _v3_score_kernel(
    K_ptr,         # [seq_len, n_kv_heads, head_dim] float32 (or BF16, cast inside)
    Cr_ptr,        # [n_kv_heads, fc] float32
    Ci_ptr,        # [n_kv_heads, fc] float32
    CbDelta_ptr,   # [n_kv_heads, fc] float32 (precomputed = c_abs - |c_complex|)
    CosSum_ptr,    # [fc] float32 (precomputed once per evict)
    SinSum_ptr,    # [fc] float32
    Scores_ptr,    # [seq_len] float32 (accumulated; kernel adds)
    Valid_ptr,     # [seq_len] uint8 (1 if cell is live, 0 if evicted)
    seq_len,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    fc: tl.constexpr,
    BLOCK_M: tl.constexpr,    # cells per program
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < seq_len

    # Read per-cell validity (0/1 → 1 if live)
    valid = tl.load(Valid_ptr + m_offs, mask=m_mask, other=0).to(tl.int32)
    live = valid > 0

    f_offs = tl.arange(0, fc)

    # Per-cell, per-head accumulators
    acc_total = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate kv-heads sequentially (small count)
    for h in tl.static_range(0, n_kv_heads):
        # Centers [fc]
        c_r = tl.load(Cr_ptr + h * fc + f_offs)
        c_i = tl.load(Ci_ptr + h * fc + f_offs)
        cb_delta = tl.load(CbDelta_ptr + h * fc + f_offs)

        # K[m, h, :n_rot] split into k_r [BLOCK_M, fc], k_i [BLOCK_M, fc]
        # K layout: [seq_len, n_kv_heads, head_dim]
        k_base = m_offs[:, None] * (n_kv_heads * head_dim) + h * head_dim
        k_r = tl.load(K_ptr + k_base + f_offs[None, :], mask=m_mask[:, None], other=0.0)
        k_i = tl.load(
            K_ptr + k_base + (fc + f_offs[None, :]),
            mask=m_mask[:, None],
            other=0.0,
        )

        # A, B, k_abs per (m, f)
        A = c_r[None, :] * k_r + c_i[None, :] * k_i
        B = c_i[None, :] * k_r - c_r[None, :] * k_i
        k_abs = tl.sqrt(k_r * k_r + k_i * k_i + 1e-8)

        # Aggregate over offsets (already collapsed into cos_sum / sin_sum)
        cos_sum = tl.load(CosSum_ptr + f_offs)
        sin_sum = tl.load(SinSum_ptr + f_offs)

        acc = tl.sum(A * cos_sum[None, :] - B * sin_sum[None, :], axis=1)
        ext = tl.sum(cb_delta[None, :] * k_abs, axis=1)
        acc_total += acc + ext

    # Mask out non-live cells
    acc_total = tl.where(live, acc_total, 0.0)

    # Atomic-add into the scores buffer is not needed because each m_offs is
    # unique per program; just accumulate via a non-atomic add on the
    # accumulator the caller already zeroed.
    existing = tl.load(Scores_ptr + m_offs, mask=m_mask, other=0.0)
    tl.store(Scores_ptr + m_offs, existing + acc_total, mask=m_mask)


def _score_cells_triton(
    K: torch.Tensor,            # [seq_len, n_kv_heads, head_dim] float32
    center_real: torch.Tensor,  # [n_kv_heads, fc]
    center_imag: torch.Tensor,
    center_abs: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    max_pos: int,
    valid_mask: torch.Tensor,   # [seq_len] bool
    window_thr: int,
    n_rot: int,
    scores_inout: torch.Tensor | None = None,
    block_m: int = 32,
) -> torch.Tensor:
    """Run the Triton kernel; return per-cell score tensor [seq_len].

    Accumulates into `scores_inout` if provided, else allocates a fresh tensor.
    Caller is responsible for accumulating across attention layers.
    """
    seq_len, n_kv, head_dim = K.shape
    fc = n_rot // 2
    device = K.device
    dtype = torch.float32

    K = K.to(dtype).contiguous()
    center_real = center_real.to(dtype).contiguous()
    center_imag = center_imag.to(dtype).contiguous()
    center_abs = center_abs.to(dtype).contiguous()

    # Precompute cb_delta = c_abs - |c_complex| once outside the kernel
    c_mag = torch.sqrt(center_real * center_real + center_imag * center_imag + 1e-8)
    cb_delta = (center_abs - c_mag).contiguous()

    # cos_sum / sin_sum averaged over offsets at max_pos
    t_vals = (float(max_pos) + offsets).to(dtype)
    phase = t_vals.unsqueeze(0) * omega.to(dtype).unsqueeze(1)  # [fc, n_off]
    cos_sum = torch.cos(phase).mean(dim=1).contiguous()
    sin_sum = torch.sin(phase).mean(dim=1).contiguous()

    if scores_inout is None:
        scores = torch.zeros(seq_len, dtype=dtype, device=device)
    else:
        scores = scores_inout

    # Apply window mask up front (kernel sees the post-window valid mask)
    full_valid = valid_mask.to(torch.uint8).contiguous()
    if window_thr > 0:
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)
        full_valid = (full_valid & (positions < window_thr).to(torch.uint8)).contiguous()

    grid = ((seq_len + block_m - 1) // block_m,)
    _v3_score_kernel[grid](
        K, center_real, center_imag, cb_delta, cos_sum, sin_sum, scores, full_valid,
        seq_len,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        fc=fc,
        BLOCK_M=block_m,
    )
    return scores
