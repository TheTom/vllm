# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

_FP8_E4B15: dict[int, int] = {}


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0.
    On non-CUDA platforms (e.g. XPU), always returns 0 (use e4nv format).
    """
    if device not in _FP8_E4B15:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TriAttention V3 valid mask (uint8 [B, max_seq_len], 1=live, 0=evicted).
    # Pointer is unused when VALID_MASK=0 (constexpr); ignored at runtime.
    Valid_mask_ptr,  # [B, max_seq_len] uint8 or null
    stride_vm_b,     # bytes per row of the valid_mask (== max_seq_len)
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
    VALUE_CENTROID: tl.constexpr = 0,  # 1 = V is centroid-quantized, not uniform
    SPARSE_V: tl.constexpr = 0,  # 1 = skip V load+accum for tiles with max_p < threshold
    SPARSE_V_THRESHOLD: tl.constexpr = 0.001,
    VALID_MASK: tl.constexpr = 0,  # 1 = read Valid_mask_ptr to mask evicted positions
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # TriAttention V3: drop scores at evicted positions to -inf so the
        # softmax sees them as zero contribution. Mask is per (batch, position).
        if VALID_MASK:
            vm_addr = bid * stride_vm_b + kv_offs
            vm = tl.load(Valid_mask_ptr + vm_addr, mask=kv_mask, other=0).to(tl.int32)
            scores = tl.where(vm > 0, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # P3.1 sparse V: skip the V load + dequant for tiles whose
        # softmax probability is entirely below threshold. The V load,
        # dequant, and accumulator update are wrapped in a single
        # if/else so Triton emits actual branch (skipping the loads
        # on the taken-skip side) rather than predicated execution.
        # softmax normalisation (l_prev, m_prev) is updated either way
        # so the running totals stay exact.
        skip_v_tile = False
        if SPARSE_V:
            skip_v_tile = tl.max(p) < SPARSE_V_THRESHOLD

        if skip_v_tile:
            # Nothing from this tile to accumulate. Just decay acc.
            acc = acc * re_scale
        else:
            # ========================================================
            # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
            # ========================================================
            val_bases = slot_bases + KPS

            # Step 1: extract VQB-bit indices.
            if VQB == 3:
                val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
                val_raw0 = tl.load(
                    KV_cache_ptr + val_addrs0,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                val_raw1 = tl.load(
                    KV_cache_ptr + val_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx_int = (
                    ((val_raw0 | (val_raw1 << 8)) >> val_bit_shift[None, :]) & 0x7
                )
            elif VQB == 4:
                vb_idx = d_offs // 2
                vb_shift = (d_offs % 2) * 4
                val_addrs = val_bases[:, None] + vb_idx[None, :]
                val_raw = tl.load(
                    KV_cache_ptr + val_addrs,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx_int = (val_raw >> vb_shift[None, :]) & 0xF
            else:  # VQB == 2 — 4 indices per byte
                vb_idx = d_offs // 4
                vb_shift = (d_offs % 4) * 2
                val_addrs = val_bases[:, None] + vb_idx[None, :]
                val_raw = tl.load(
                    KV_cache_ptr + val_addrs,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx_int = (val_raw >> vb_shift[None, :]) & 0x3

            # Step 2: dequantize.
            if VALUE_CENTROID:
                v_centroids = tl.load(
                    Centroids_ptr + v_idx_int,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                n_bases = val_bases + VAL_DATA_BYTES
                n_lo_v = tl.load(KV_cache_ptr + n_bases, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                n_hi_v = tl.load(KV_cache_ptr + n_bases + 1, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                v_norms = (
                    (n_lo_v | (n_hi_v << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                values = v_centroids * v_norms[:, None]
            else:
                v_idx = v_idx_int.to(tl.float32)
                sc_bases = val_bases + VAL_DATA_BYTES
                sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                v_scales = (
                    (sc_lo | (sc_hi << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                v_zeros = (
                    (zr_lo | (zr_hi << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                values = v_idx * v_scales[:, None] + v_zeros[:, None]

            # Weighted accumulation [BLOCK_D]
            acc = acc * re_scale + tl.sum(p[:, None] * values, 0)

        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
    VALUE_CENTROID: tl.constexpr = 0,
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    # Step 1: extract VQB-bit indices (same packing for centroid + uniform)
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx_int = (val_raw >> vb_shift) & 0xF
    elif VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        v_idx_int = ((val_raw0 | (val_raw1 << 8)) >> val_bit_shift) & 0x7
    elif VQB == 2:
        vb_idx = d_offs // 4
        vb_shift = (d_offs % 4) * 2
        val_raw = tl.load(
            KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0
        ).to(tl.int32)
        v_idx_int = (val_raw >> vb_shift) & 0x3
    else:
        v_idx_int = tl.zeros([BLOCK_D], dtype=tl.int32)

    # Step 2: dequantize. Centroid (gather + norm) or uniform (scale + zero).
    if VQB == 2 or VQB == 3 or VQB == 4:
        if VALUE_CENTROID:
            v_centroid = tl.load(Centroids_ptr + v_idx_int, mask=d_mask, other=0.0)
            n_base = val_base + VAL_DATA_BYTES
            n_lo_v = tl.load(KV_cache_ptr + n_base).to(tl.uint16)
            n_hi_v = tl.load(KV_cache_ptr + n_base + 1).to(tl.uint16)
            v_norm_f = (
                (n_lo_v | (n_hi_v << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            v_vals = v_centroid * v_norm_f
        else:
            v_idx = v_idx_int.to(tl.float32)
            sc_base = val_base + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
            v_scale = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
            v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D_wht, D_wht] float32 (may be padded dim)
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D_wht, D_wht] pre-computed
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
    rotate_values: bool = False,
    original_head_dim: int = 0,
    value_centroid: bool = False,
    sparse_v: bool = False,
    sparse_v_threshold: float = 0.001,
    valid_mask: torch.Tensor | None = None,  # uint8 [B, max_seq_len], 1=live
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    `valid_mask` is the TriAttention V3 per-position validity mask. When
    provided, evicted positions (mask == 0) score -inf at attention time.

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    D_orig = original_head_dim if original_head_dim > 0 else D
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    # For padded head dims, query is padded before rotation.
    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        D_wht = PiT.shape[0]
        if D_wht > D:
            q_float = torch.nn.functional.pad(q_float, (0, D_wht - D))
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1: split-KV tiled attention scoring + value accumulation
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_KV = 4
    grid = (B, Hq, NUM_KV_SPLITS)
    if valid_mask is not None:
        vm_tensor = valid_mask
        vm_stride_b = valid_mask.stride(0)
        valid_mask_flag = 1
    else:
        vm_tensor = q_rot  # dummy; constexpr gate skips reads
        vm_stride_b = 0
        valid_mask_flag = 0

    _tq_decode_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        vm_tensor,
        vm_stride_b,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        VALUE_CENTROID=1 if value_centroid else 0,
        SPARSE_V=1 if sparse_v else 0,
        SPARSE_V_THRESHOLD=sparse_v_threshold,
        VALID_MASK=valid_mask_flag,
        num_warps=1,
        num_stages=1,
    )

    # Stage 2: Reduce across KV splits
    # Output in query dtype — eliminates float16_copy kernel after stage2
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    # TQ+: inverse WHT on accumulated values. WHT is linear so
    # H·Σ(w_i·v_i) = Σ(w_i·H·v_i) — one GEMM undoes the V rotation.
    # output is in query.dtype after stage2; cast Pi to match for the GEMM.
    if rotate_values:
        B_out, Hq_out, D_out = output.shape
        Pi_q = Pi.to(output.dtype)
        output = (output.reshape(-1, D_out) @ Pi_q).reshape(B_out, Hq_out, D_out)

    # Slice back to original head_dim if padded for WHT
    if D_orig < output.shape[-1]:
        output = output[..., :D_orig].contiguous()

    return output  # already in query dtype


# ---------------------------------------------------------------------------
# Stage 1 — grouped variant (P3.4 / P3.2 foundation)
#
# Processes M_GRP queries per program, sharing one kv_head's K/V loads
# across the group. Score reduction becomes a tl.dot which engages MFMA
# when M_GRP × HEAD_DIM × BLOCK_KV align with gfx942 tile sizes (≥16 each).
# At M_GRP=KV_GROUP_SIZE this naturally batches all queries that map to
# the same kv_head via GQA.
#
# Minimum-viable scope (this version):
#   - MSE K only (no FP8 K)
#   - VQB == 4 only (no 2/3-bit)
# Supported: NORM_CORRECTION, VALUE_CENTROID, SPARSE_V.
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1_grouped(
    Q_rot_ptr,              # [B, Hq, D] float32 (post-rotation)
    KV_cache_ptr,           # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,        # [B, max_blocks] int32
    Seq_lens_ptr,           # [B] int32
    Centroids_ptr,          # [n_centroids] float32
    Mid_o_ptr,              # [B, Hq, NUM_KV_SPLITS, D+1] float32
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    M_GRP: tl.constexpr,    # queries per program
    NORM_CORRECTION: tl.constexpr = 0,
    VALUE_CENTROID: tl.constexpr = 0,
    SPARSE_V: tl.constexpr = 0,
    SPARSE_V_THRESHOLD: tl.constexpr = 0.1,
):
    bid = tl.program_id(0)         # batch
    grp_id = tl.program_id(1)      # which kv-head-group (== kv_head when M_GRP=KV_GROUP_SIZE)
    sid = tl.program_id(2)         # kv split

    # M_GRP queries map to one kv_head. q_head_start is the first q_head idx.
    kv_head = grp_id
    q_head_start = grp_id * M_GRP

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    m_range = tl.arange(0, M_GRP)

    # Load Q tile [M_GRP, BLOCK_D]
    q_rot = tl.load(
        Q_rot_ptr
        + bid * stride_qb
        + (q_head_start + m_range[:, None]) * stride_qh
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # Zero out padding columns so tl.dot doesn't see junk
    q_rot = tl.where(d_mask[None, :], q_rot, 0.0)

    # MSE bit/byte indices
    mse_bit_off = d_offs * MSE_BITS
    mse_byte_idx = mse_bit_off // 8
    mse_bit_shift = mse_bit_off % 8
    mse_mask = (1 << MSE_BITS) - 1

    # Online softmax state — per row (M_GRP)
    m_prev = tl.full([M_GRP], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([M_GRP], dtype=tl.float32)
    acc = tl.zeros([M_GRP, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ---------------- K MSE unpack [BLOCK_KV, BLOCK_D] ----------------
        mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
        mse_raw0 = tl.load(
            KV_cache_ptr + mse_addrs0,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + mse_addrs0 + 1,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        raw16 = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
        c_vals = tl.load(
            Centroids_ptr + mse_idx,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(
                tl.where(d_mask[None, :], c_vals * c_vals, 0.0), axis=1
            )
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            c_vals = c_vals * c_inv_norm[:, None]
        c_vals = tl.where(d_mask[None, :], c_vals, 0.0)

        # Per-vector norms [BLOCK_KV]
        norm_bases = slot_bases + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        # Q · C^T → [M_GRP, BLOCK_KV], scale + multiply by per-key norm
        # tl.dot expects f16/bf16 inputs for MFMA; we keep f32 and accept FMA.
        scores = tl.dot(q_rot, tl.trans(c_vals)) * vec_norms[None, :] * ATTN_SCALE
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        # ---------------- Online softmax (per row) ----------------
        score_max = tl.max(scores, axis=1)               # [M_GRP]
        n_e_max = tl.maximum(score_max, m_prev)          # [M_GRP]
        re_scale = tl.exp(m_prev - n_e_max)              # [M_GRP]
        p = tl.exp(scores - n_e_max[:, None])            # [M_GRP, BLOCK_KV]
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

        # Sparse V tile-skip: if max softmax mass in this tile (across all
        # M_GRP queries and BLOCK_KV positions) is below threshold, the V
        # contribution from this tile is negligible. Skip the V load +
        # dequant + accumulator update; just decay acc by re_scale.
        skip_v_tile = False
        if SPARSE_V:
            skip_v_tile = tl.max(p) < SPARSE_V_THRESHOLD

        if skip_v_tile:
            acc = acc * re_scale[:, None]
        else:
            # ---------------- V load (4-bit indices) ----------------
            val_bases = slot_bases + KPS
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx_int = (val_raw >> vb_shift[None, :]) & 0xF

            if VALUE_CENTROID:
                v_centroids = tl.load(
                    Centroids_ptr + v_idx_int,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                n_bases = val_bases + VAL_DATA_BYTES
                n_lo_v = tl.load(KV_cache_ptr + n_bases, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                n_hi_v = tl.load(
                    KV_cache_ptr + n_bases + 1, mask=kv_mask, other=0
                ).to(tl.uint16)
                v_norms = (
                    (n_lo_v | (n_hi_v << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                values = v_centroids * v_norms[:, None]
            else:
                v_idx = v_idx_int.to(tl.float32)
                sc_bases = val_bases + VAL_DATA_BYTES
                sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                    tl.uint16
                )
                sc_hi = tl.load(
                    KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0
                ).to(tl.uint16)
                v_scales = (
                    (sc_lo | (sc_hi << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                zr_lo = tl.load(
                    KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0
                ).to(tl.uint16)
                zr_hi = tl.load(
                    KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0
                ).to(tl.uint16)
                v_zeros = (
                    (zr_lo | (zr_hi << 8))
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                values = v_idx * v_scales[:, None] + v_zeros[:, None]
            values = tl.where(d_mask[None, :], values, 0.0)

            # acc [M_GRP, D] += p [M_GRP, BLOCK_KV] @ V [BLOCK_KV, D]
            acc = acc * re_scale[:, None] + tl.dot(p, values)

    # Output: M_GRP rows
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l[:, None]
    out_base = (
        bid * stride_mid_b
        + (q_head_start + m_range[:, None]) * stride_mid_h
        + sid * stride_mid_s
    )
    tl.store(Mid_o_ptr + out_base + d_offs[None, :], out, mask=d_mask[None, :])
    lse = m_prev + tl.log(safe_l)
    out_lse_base = (
        bid * stride_mid_b
        + (q_head_start + m_range) * stride_mid_h
        + sid * stride_mid_s
        + HEAD_DIM
    )
    tl.store(Mid_o_ptr + out_lse_base, lse)


def triton_turboquant_decode_attention_grouped(
    query: torch.Tensor,             # [B, Hq, D]
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    norm_correction: bool = False,
    value_centroid: bool = False,
    PiT: torch.Tensor | None = None,
    max_num_kv_splits: int = 32,
    rotate_values: bool = False,
    original_head_dim: int = 0,
    m_grp: int | None = None,
    sparse_v: bool = False,
    sparse_v_threshold: float = 0.1,
) -> torch.Tensor:
    """Batched-Q variant of triton_turboquant_decode_attention.

    Each kernel program processes M_GRP queries that share a kv_head
    (M_GRP defaults to KV_GROUP_SIZE so all queries within a program
    map to the same kv head via GQA — letting one K/V load serve the
    whole group).

    Restricted scope:
      - MSE K only (no FP8 K)
      - VQB == 4 only (no 2-bit, no 3-bit)
      - M_GRP == KV_GROUP_SIZE
    Supports NORM_CORRECTION, VALUE_CENTROID, and SPARSE_V.
    """
    assert mse_bits in (3, 4) and value_quant_bits == 4, (
        f"grouped path only supports MSE K + 4-bit V (got mse_bits={mse_bits}, "
        f"value_quant_bits={value_quant_bits})"
    )
    B, Hq, D = query.shape
    D_orig = original_head_dim if original_head_dim > 0 else D
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    if m_grp is None:
        m_grp = kv_group_size
    assert m_grp == kv_group_size, (
        "current grouped impl requires M_GRP == KV_GROUP_SIZE so a program's "
        f"queries all share one kv_head (got M_GRP={m_grp}, KV_GROUP_SIZE={kv_group_size})"
    )
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Q rotation: same as the non-grouped path.
    q_float = query.float()
    if PiT is None:
        PiT = Pi.T.contiguous()
    D_wht = PiT.shape[0]
    if D_wht > D:
        q_float = torch.nn.functional.pad(q_float, (0, D_wht - D))
    q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits
    mid_o = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device)

    # Grid: (B, num_kv_heads, NUM_KV_SPLITS) — each program handles M_GRP q heads
    grid = (B, Hk, NUM_KV_SPLITS)
    BLOCK_KV = 16  # MFMA-aligned (M_GRP=8 × BLOCK_KV=16 × HEAD_DIM=128 — 2 of 3 dims are 16+)
    _tq_decode_stage1_grouped[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        M_GRP=m_grp,
        NORM_CORRECTION=1 if norm_correction else 0,
        VALUE_CENTROID=1 if value_centroid else 0,
        SPARSE_V=1 if sparse_v else 0,
        SPARSE_V_THRESHOLD=sparse_v_threshold,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2: reuse existing reduction kernel (same mid_o layout)
    out_dtype = query.dtype
    output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    if rotate_values:
        B_out, Hq_out, D_out = output.shape
        Pi_q = Pi.to(output.dtype)
        output = (output.reshape(-1, D_out) @ Pi_q).reshape(B_out, Hq_out, D_out)
    if D_orig < output.shape[-1]:
        output = output[..., :D_orig].contiguous()
    return output
