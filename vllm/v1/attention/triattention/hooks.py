"""Q-capture hooks for TriAttention V3.

The engine needs the *pre-RoPE* Q tensor at every attention layer. vLLM's
attention layer (`vllm.model_executor.layers.attention.Attention.forward`)
receives Q *after* RoPE, so the capture has to live earlier — inside the
model's attention block, just before its `rotary_emb(...)` call.

Mirrors the llama.cpp impl, which emits Q via the ggml scheduler's eval
callback. This module exposes a tiny explicit-emit API:

    from vllm.v1.attention.triattention.hooks import capture_q_pre_rope
    ...
    # In each model's attention forward, right before RoPE:
    capture_q_pre_rope(self.layer_idx, q_view)
    q, k = self.rotary_emb(positions, q, k)

When V3 is not installed, the call is a tight no-op (single dict lookup).
"""
from __future__ import annotations

from typing import Optional

import torch

from vllm.v1.attention.triattention.engine import TriAttentionV3Engine

# Module-level singleton engine (one per process). install_triattention sets it.
_engine: Optional[TriAttentionV3Engine] = None
_enabled: bool = False


def set_engine(engine: Optional[TriAttentionV3Engine]) -> None:
    global _engine, _enabled
    _engine = engine
    _enabled = engine is not None


def get_engine() -> Optional[TriAttentionV3Engine]:
    return _engine


def is_enabled() -> bool:
    return _enabled


def capture_q_pre_rope(layer_idx: int, q: torch.Tensor) -> None:
    """Emit pre-RoPE Q to the V3 engine. No-op when V3 is disabled.

    q: shape [n_tokens, n_heads * head_dim] OR [n_tokens, n_heads, head_dim].
    """
    eng = _engine
    if eng is None or eng.calibrated and not eng.cfg.adaptive_calibration:
        return
    # Reshape to [n_tokens, n_heads, head_dim] if flat
    if q.ndim == 2:
        n_heads = eng.n_heads
        head_dim = eng.head_dim
        if q.shape[1] != n_heads * head_dim:
            return  # not a Q tensor we can interpret
        q = q.view(-1, n_heads, head_dim)
    elif q.ndim == 3:
        if q.shape[1] != eng.n_heads or q.shape[2] != eng.head_dim:
            return
    else:
        return
    eng.accumulate_q(q.detach(), layer_idx)
