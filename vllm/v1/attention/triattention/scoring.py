"""TriAttention V3 scoring.

Two implementations:
  - score_cells_torch: PyTorch reference (Phase A)
  - score_cells_triton: Triton kernel for AMD MI300X (Phase B), MFMA-friendly
    via [n_cells, freq_count] tile shape on `tl.dot`.

Math (per cell, per kv-head block):
  Let c_r, c_i, c_abs be the calibration centers for this (layer, kv_head).
  Let k be the K vector for this cell, split half-layout: k_r = k[:fc],
  k_i = k[fc:n_rot] (real/imag pair).

  A_f = c_r[f] * k_r[f] + c_i[f] * k_i[f]   # Re(center · conj(K))
  B_f = c_i[f] * k_r[f] - c_r[f] * k_i[f]   # Im(center · conj(K))
  k_abs[f] = sqrt(k_r[f]**2 + k_i[f]**2 + 1e-8)
  c_mag[f] = sqrt(c_r[f]**2 + c_i[f]**2 + 1e-8)
  cb_delta[f] = c_abs[f] - c_mag[f]   # MLR additive scale

  cos_sum[f] = mean over offsets o of cos((max_pos + o) * omega[f])
  sin_sum[f] = mean over offsets o of sin((max_pos + o) * omega[f])

  acc_cell = sum_f (A_f * cos_sum[f] - B_f * sin_sum[f])
  ext_cell = sum_f cb_delta[f] * k_abs[f]
  score[cell] += acc_cell + ext_cell    (accumulated over all (layer, kv_head))

The trig formula scores HIGH for tokens to *evict* (it measures orthogonality
to the calibrated center, not alignment).
"""
from __future__ import annotations

from typing import Optional

import torch


def _precompute_offset_sums(
    omega: torch.Tensor,    # [fc]
    offsets: torch.Tensor,  # [n_off]
    max_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos_sum, sin_sum, each shape [fc], averaged over offsets."""
    # phase[f, o] = (max_pos + offsets[o]) * omega[f]
    t_vals = (float(max_pos) + offsets).to(torch.float32)         # [n_off]
    omega_f = omega.to(torch.float32)                              # [fc]
    phase = t_vals.unsqueeze(0) * omega_f.unsqueeze(1)             # [fc, n_off]
    cos_sum = torch.cos(phase).mean(dim=1)                         # [fc]
    sin_sum = torch.sin(phase).mean(dim=1)                         # [fc]
    return cos_sum, sin_sum


def score_cells_torch(
    K: torch.Tensor,            # [seq_len, n_kv_heads, head_dim] float32
    center_real: torch.Tensor,  # [n_kv_heads, fc] float32
    center_imag: torch.Tensor,  # [n_kv_heads, fc] float32
    center_abs: torch.Tensor,   # [n_kv_heads, fc] float32
    omega: torch.Tensor,        # [fc]
    offsets: torch.Tensor,      # [n_off]
    max_pos: int,
    valid_mask: torch.Tensor,   # [seq_len] bool
    window_thr: int,
    n_rot: int,
) -> torch.Tensor:
    """Compute V3 scores for one layer, summed across kv-heads.

    Returns: [seq_len] float32 scores. Tokens beyond `valid_mask=False` get
    score 0; tokens at pos >= window_thr (recent window) also get 0 (they are
    skipped in the policy stage, so the 0 doesn't matter there).
    """
    seq_len, n_kv, hd = K.shape
    fc = n_rot // 2
    device = K.device

    # Take rotated half of K and split real/imag
    k_r = K[:, :, :fc]                  # [T, H, fc]
    k_i = K[:, :, fc:n_rot]             # [T, H, fc]
    k_abs = torch.sqrt(k_r * k_r + k_i * k_i + 1e-8)  # [T, H, fc]

    # Complex product center * conj(k):  (cr + i ci)(kr - i ki) = (cr*kr + ci*ki) + i (ci*kr - cr*ki)
    # Broadcast: center [H, fc] -> [1, H, fc]
    cr = center_real.unsqueeze(0)       # [1, H, fc]
    ci = center_imag.unsqueeze(0)
    A = cr * k_r + ci * k_i             # [T, H, fc]
    B = ci * k_r - cr * k_i             # [T, H, fc]

    # cb_delta = c_abs - |center_complex|
    c_mag = torch.sqrt(center_real * center_real + center_imag * center_imag + 1e-8)
    cb_delta = (center_abs - c_mag).unsqueeze(0)  # [1, H, fc]

    cos_sum, sin_sum = _precompute_offset_sums(omega, offsets, max_pos)
    cos_sum = cos_sum.unsqueeze(0).unsqueeze(0)   # [1, 1, fc]
    sin_sum = sin_sum.unsqueeze(0).unsqueeze(0)

    # Per-cell-per-head: acc, ext
    acc = (A * cos_sum - B * sin_sum).sum(dim=-1)           # [T, H]
    ext = (cb_delta * k_abs).sum(dim=-1)                    # [T, H]
    per_head = acc + ext                                     # [T, H]

    # Sum across kv-heads
    score = per_head.sum(dim=-1)                             # [T]

    # Zero out invalid + window-protected
    if valid_mask is not None:
        score = torch.where(valid_mask, score, torch.zeros_like(score))
    if window_thr > 0:
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)
        score = torch.where(positions >= window_thr, torch.zeros_like(score), score)

    return score


# ---------------------------------------------------------------------------
# Triton kernel (Phase B)
# ---------------------------------------------------------------------------

def score_cells_triton(*args, **kwargs):
    """Triton implementation lives in scoring_kernel.py; lazy-imported."""
    from vllm.v1.attention.triattention.scoring_kernel import _score_cells_triton
    return _score_cells_triton(*args, **kwargs)
