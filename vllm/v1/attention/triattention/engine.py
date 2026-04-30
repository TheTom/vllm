"""TriAttention V3 engine: calibration + scoring + eviction state.

Direct port of `llama_triattention` (src/llama-triattention.cpp on the
`experiment/triattention-integration` branch of the llama.cpp fork). The math
is identical; only the surrounding plumbing changes.

State layout (per attention head group, indexed [layer * n_kv_heads + kv_h]):
  q_sum_real / q_sum_imag / q_sum_abs : accumulators over Q samples
  center_real / center_imag / center_abs: snapshot means after warmup
  q_samples                             : tokens accumulated so far

Per-sequence runtime state:
  valid_mask[seq_id] -> bool tensor [max_seq_len], True if position is live
  evicted_count[seq_id] -> int (running tally, used by should_evict)
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from vllm.v1.attention.triattention.policy import select_v3_evictions
from vllm.v1.attention.triattention.scoring import score_cells_torch


@dataclass
class TriAttentionV3Config:
    budget: int = 2048
    divide_length: int = 128
    window_size: int = 128
    prefix_protect: int = 128
    n_segments: int = 8
    warmup_tokens: int = 1024
    ema_alpha: float = 0.1  # only used when adaptive_calibration=True
    adaptive_calibration: bool = False
    hybrid_mode: int = 2  # 0=V1 global sort, 1=V2 quota only, 2=V3 prefix+quota

    # Boundary skip: kept for parity with llama.cpp impl; default 0 (proven
    # negative result on standard transformers, see docs §4.8).
    boundary_skip: int = 0

    @classmethod
    def from_env(cls) -> "TriAttentionV3Config":
        cfg = cls()
        if v := os.environ.get("VLLM_TRIATT_BUDGET"):
            cfg.budget = int(v)
        if v := os.environ.get("VLLM_TRIATT_HYBRID"):
            cfg.hybrid_mode = int(v)
        if v := os.environ.get("VLLM_TRIATT_PREFIX"):
            cfg.prefix_protect = int(v)
        if v := os.environ.get("VLLM_TRIATT_WINDOW"):
            cfg.window_size = int(v)
        if v := os.environ.get("VLLM_TRIATT_SEGMENTS"):
            cfg.n_segments = int(v)
        if v := os.environ.get("VLLM_TRIATT_WARMUP"):
            cfg.warmup_tokens = int(v)
        if v := os.environ.get("VLLM_TRIATT_ADAPTIVE"):
            cfg.adaptive_calibration = bool(int(v))
        return cfg


class TriAttentionV3Engine:
    """Calibration + scoring + per-sequence eviction state for V3.

    One engine instance per LLM (i.e. process). Calibration is global across
    all sequences; per-sequence state lives in `_seq_state`.
    """

    def __init__(
        self,
        cfg: TriAttentionV3Config,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        n_rot: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rot = head_dim if n_rot is None or n_rot == 0 else n_rot
        assert self.n_rot % 2 == 0, "n_rot must be even (real/imag halves)"
        self.freq_count = self.n_rot // 2
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype

        # RoPE angular frequencies omega[i] = 1 / theta^(2i/n_rot)
        i = torch.arange(self.freq_count, dtype=torch.float32, device=device)
        self.omega = (1.0 / (rope_theta ** (2 * i / self.n_rot))).to(dtype)

        # Geometric offsets [1, 2, 4, ..., 65536]
        offs: List[int] = []
        v = 1
        while v <= 65536:
            offs.append(v)
            v *= 2
        self.offsets = torch.tensor(offs, dtype=dtype, device=device)
        self.n_off = len(offs)

        # Per-(layer, kv_head) accumulators and centers, shape [n_total, freq_count]
        n_total = n_layers * n_kv_heads
        zero = torch.zeros(n_total, self.freq_count, dtype=dtype, device=device)
        self.q_sum_real = zero.clone()
        self.q_sum_imag = zero.clone()
        self.q_sum_abs = zero.clone()
        self.q_prev_sum_real = zero.clone()
        self.q_prev_sum_imag = zero.clone()
        self.q_prev_sum_abs = zero.clone()
        self.center_real: Optional[torch.Tensor] = None
        self.center_imag: Optional[torch.Tensor] = None
        self.center_abs: Optional[torch.Tensor] = None

        self.q_samples: int = 0
        self.q_samples_at_last_update: int = 0
        self.calibrated: bool = False
        self.first_attn_layer: int = -1
        self.pending_uninstall: bool = False

        # Per-sequence runtime state
        # seq_id -> {valid_mask: BoolTensor [max_len], n_evicted: int, max_pos: int}
        self._seq_state: Dict[int, Dict] = {}
        self._lock = threading.Lock()

        # Total eviction rounds across all sequences (for telemetry).
        self.total_evict_rounds: int = 0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def accumulate_q(self, q_pre_rope: torch.Tensor, layer_idx: int) -> None:
        """Accumulate Q stats for one attention layer.

        q_pre_rope: [n_tokens, n_heads, head_dim] float (pre-RoPE).
        """
        if self.calibrated and not self.cfg.adaptive_calibration:
            return
        if not (0 <= layer_idx < self.n_layers):
            return

        # Take the first n_rot dims (split half/half real/imag like RoPE half-layout)
        fc = self.freq_count
        # q_pre_rope -> [T, H, n_rot] -> split real, imag
        q = q_pre_rope[..., : self.n_rot].to(torch.float32)
        q_real = q[..., :fc]                 # [T, H, fc]
        q_imag = q[..., fc:self.n_rot]       # [T, H, fc]
        q_abs = torch.sqrt(q_real * q_real + q_imag * q_imag + 1e-8)

        # Group H queries into n_kv_heads groups of (heads_per_kv) and average.
        heads_per_kv = self.n_heads // self.n_kv_heads
        # Reshape -> [T, n_kv_heads, heads_per_kv, fc]
        q_real = q_real.view(-1, self.n_kv_heads, heads_per_kv, fc).mean(dim=2)
        q_imag = q_imag.view(-1, self.n_kv_heads, heads_per_kv, fc).mean(dim=2)
        q_abs = q_abs.view(-1, self.n_kv_heads, heads_per_kv, fc).mean(dim=2)
        # -> [T, n_kv_heads, fc], sum over T to add to accumulator
        sum_real = q_real.sum(dim=0)  # [n_kv_heads, fc]
        sum_imag = q_imag.sum(dim=0)
        sum_abs = q_abs.sum(dim=0)

        with self._lock:
            base = layer_idx * self.n_kv_heads
            self.q_sum_real[base : base + self.n_kv_heads] += sum_real.to(self.dtype)
            self.q_sum_imag[base : base + self.n_kv_heads] += sum_imag.to(self.dtype)
            self.q_sum_abs[base : base + self.n_kv_heads] += sum_abs.to(self.dtype)

            # Token counter: per-pass (count once on first attention layer seen)
            if self.first_attn_layer < 0 or layer_idx < self.first_attn_layer:
                self.first_attn_layer = layer_idx
            if layer_idx == self.first_attn_layer:
                self.q_samples += int(q_pre_rope.shape[0])
                if (
                    not self.calibrated
                    and self.q_samples >= self.cfg.warmup_tokens
                ):
                    self.update_calibration_locked()

    def update_calibration(self) -> None:
        with self._lock:
            self.update_calibration_locked()

    def update_calibration_locked(self) -> None:
        if self.q_samples <= 0:
            return
        if not self.calibrated:
            inv_n = 1.0 / float(self.q_samples)
            self.center_real = self.q_sum_real * inv_n
            self.center_imag = self.q_sum_imag * inv_n
            self.center_abs = self.q_sum_abs * inv_n
            self.calibrated = True
            print(
                f"[TriAttention V3] calibrated from {self.q_samples} Q samples "
                f"({self.n_layers} layers × {self.n_kv_heads} kv-heads)",
                flush=True,
            )
            if not self.cfg.adaptive_calibration:
                self.pending_uninstall = True
        else:
            alpha = self.cfg.ema_alpha
            new_samples = self.q_samples - self.q_samples_at_last_update
            if new_samples <= 0:
                return
            inv_n = 1.0 / float(new_samples)
            new_r = (self.q_sum_real - self.q_prev_sum_real) * inv_n
            new_i = (self.q_sum_imag - self.q_prev_sum_imag) * inv_n
            new_a = (self.q_sum_abs - self.q_prev_sum_abs) * inv_n
            self.center_real = (1 - alpha) * self.center_real + alpha * new_r
            self.center_imag = (1 - alpha) * self.center_imag + alpha * new_i
            self.center_abs = (1 - alpha) * self.center_abs + alpha * new_a

        self.q_prev_sum_real = self.q_sum_real.clone()
        self.q_prev_sum_imag = self.q_sum_imag.clone()
        self.q_prev_sum_abs = self.q_sum_abs.clone()
        self.q_samples_at_last_update = self.q_samples

    # ------------------------------------------------------------------
    # Per-sequence valid-mask state
    # ------------------------------------------------------------------

    def get_valid_mask(self, seq_id: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a bool tensor [seq_len] marking positions still in the cache."""
        st = self._seq_state.setdefault(
            seq_id,
            {"valid_mask": None, "n_evicted": 0, "max_pos": -1, "evict_rounds": 0},
        )
        m = st["valid_mask"]
        if m is None or m.shape[0] < seq_len:
            new = torch.ones(seq_len, dtype=torch.bool, device=device)
            if m is not None:
                new[: m.shape[0]] = m.to(device)
            st["valid_mask"] = new
            return new
        return m[:seq_len]

    def n_used(self, seq_id: int, seq_len: int) -> int:
        st = self._seq_state.get(seq_id)
        if st is None or st["valid_mask"] is None:
            return seq_len
        m = st["valid_mask"][:seq_len]
        return int(m.sum().item())

    def should_evict(self, seq_id: int, seq_len: int) -> bool:
        if not self.calibrated:
            return False
        used = self.n_used(seq_id, seq_len)
        eff_budget = max(self.cfg.budget, self.cfg.window_size + 1)
        return used > eff_budget + self.cfg.divide_length

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-layer score accumulation (incremental path used by the runtime)
    # ------------------------------------------------------------------

    def begin_score_round(
        self, seq_id: int, seq_len: int, device: torch.device
    ) -> None:
        """Initialise per-layer score accumulators for `seq_id`.

        Call once per scheduler step before per-layer K is pushed via
        `accumulate_layer_score`. After all attention layers contribute,
        `finalize_evict_round` runs the V3 policy and updates the valid mask.
        """
        st = self._seq_state.setdefault(
            seq_id,
            {"valid_mask": None, "n_evicted": 0, "max_pos": -1, "evict_rounds": 0},
        )
        st["pending_scores"] = torch.zeros(
            seq_len, dtype=torch.float32, device=device
        )
        st["pending_n_blocks"] = 0
        st["pending_seq_len"] = seq_len
        st["pending_layers"] = set()

    def accumulate_layer_score(
        self,
        seq_id: int,
        layer_il: int,
        K: torch.Tensor,
        max_pos: int,
        window_thr: int,
    ) -> None:
        """Add one layer's contribution to the pending score buffer.

        K: [seq_len, n_kv_heads, head_dim] float (BF16 / FP32 OK).
        """
        if not self.calibrated:
            return
        if layer_il < self.cfg.boundary_skip:
            return
        st = self._seq_state.get(seq_id)
        if st is None or "pending_scores" not in st:
            return
        # Guard against shape drift: vLLM's compile / capture / profile flow
        # can fire continuation prefill with a K shape that doesn't match
        # the open accumulator. Skip those layers — partial scores would
        # corrupt the next legitimate pass. The next correctly-shaped layer
        # will trigger a fresh begin_score_round via _v3_accumulate_prefill_k.
        if st["pending_scores"].shape[0] != int(K.shape[0]):
            return
        scores = st["pending_scores"]
        valid = self.get_valid_mask(seq_id, scores.shape[0], scores.device)
        cb = layer_il * self.n_kv_heads
        c_r = self.center_real[cb : cb + self.n_kv_heads].to(scores.device)
        c_i = self.center_imag[cb : cb + self.n_kv_heads].to(scores.device)
        c_abs = self.center_abs[cb : cb + self.n_kv_heads].to(scores.device)
        omega_d = self.omega.to(scores.device)
        offsets_d = self.offsets.to(scores.device)
        scores += score_cells_torch(
            K=K.to(torch.float32),
            center_real=c_r.to(torch.float32),
            center_imag=c_i.to(torch.float32),
            center_abs=c_abs.to(torch.float32),
            omega=omega_d.to(torch.float32),
            offsets=offsets_d.to(torch.float32),
            max_pos=max_pos,
            valid_mask=valid,
            window_thr=window_thr,
            n_rot=self.n_rot,
        )
        st["pending_n_blocks"] += self.n_kv_heads

    def finalize_evict_round(self, seq_id: int) -> int:
        """Run V3 policy on accumulated scores. Returns n_evicted."""
        st = self._seq_state.get(seq_id)
        if st is None or "pending_scores" not in st:
            return 0
        scores = st.pop("pending_scores")
        n_blocks = st.pop("pending_n_blocks")
        seq_len = st.pop("pending_seq_len")
        if n_blocks > 0:
            scores = scores / float(n_blocks)

        valid = self.get_valid_mask(seq_id, seq_len, scores.device)
        used = int(valid.sum().item())
        n_to_evict = used - self.cfg.budget
        if n_to_evict <= 0:
            return 0

        positions = torch.arange(seq_len, dtype=torch.int32, device=valid.device)
        live_pos = positions[valid]
        if live_pos.numel() == 0:
            return 0
        max_pos = int(live_pos.max().item())
        window_thr = max_pos - self.cfg.window_size + 1
        prefix_lo = self.cfg.prefix_protect if self.cfg.hybrid_mode == 2 else 0

        evict_pos = select_v3_evictions(
            scores=scores,
            valid=valid,
            n_to_evict=n_to_evict,
            window_thr=window_thr,
            prefix_lo=prefix_lo,
            n_segments=self.cfg.n_segments,
            mode=self.cfg.hybrid_mode,
        )
        valid[evict_pos] = False
        n_evicted = int(evict_pos.numel())
        st["n_evicted"] += n_evicted
        st["max_pos"] = max_pos
        st["evict_rounds"] += 1
        self.total_evict_rounds += 1
        if self.total_evict_rounds <= 5 or self.total_evict_rounds % 10 == 0:
            print(
                f"[TriAttention V3] evict round {self.total_evict_rounds}: "
                f"seq_len={seq_len} used={used} -> {used - n_evicted} (-{n_evicted}) "
                f"max_pos={max_pos} window=[{window_thr},{seq_len}) "
                f"prefix=[0,{prefix_lo})",
                flush=True,
            )
        return n_evicted

    # ------------------------------------------------------------------
    # One-shot eviction (used by tests + manual triggers)
    # ------------------------------------------------------------------

    def evict(
        self,
        seq_id: int,
        seq_len: int,
        k_per_layer: List[torch.Tensor],
        layer_il_map: Optional[List[int]] = None,
    ) -> int:
        """One-shot eviction: score over all layers in `k_per_layer` then
        run V3 policy. Returns n_evicted.

        Used by tests and by harnesses that already hold dequanted K for
        every attention layer of a given sequence.
        """
        if not self.calibrated or not k_per_layer:
            return 0
        device = k_per_layer[0].device
        self.begin_score_round(seq_id, seq_len, device)
        if layer_il_map is None:
            layer_il_map = list(range(len(k_per_layer)))
        valid = self.get_valid_mask(seq_id, seq_len, device)
        if int(valid.sum().item()) == 0:
            return 0
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)
        max_pos = int(positions[valid].max().item())
        window_thr = max_pos - self.cfg.window_size + 1
        for K, il in zip(k_per_layer, layer_il_map):
            self.accumulate_layer_score(seq_id, il, K, max_pos, window_thr)
        return self.finalize_evict_round(seq_id)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        return {
            "calibrated": self.calibrated,
            "q_samples": self.q_samples,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "freq_count": self.freq_count,
            "n_offsets": self.n_off,
            "rope_theta": self.rope_theta,
            "total_evict_rounds": self.total_evict_rounds,
            "n_sequences": len(self._seq_state),
            "evicted_per_seq": {
                k: v["n_evicted"] for k, v in self._seq_state.items()
            },
        }
