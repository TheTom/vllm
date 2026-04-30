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

vLLM V1 forks an EngineCore subprocess for model execution, so the engine
must live in the worker. `install_triattention` from the main process sets
env vars (`VLLM_TRIATT_*`); the worker lazy-initialises its engine instance
on the first Q-capture call, reading config from those env vars.

When V3 is not enabled (no env var), the call is a tight no-op.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)

ENV_ENABLED = "VLLM_TRIATT_ENABLED"
ENV_N_LAYERS = "VLLM_TRIATT_N_LAYERS"
ENV_N_HEADS = "VLLM_TRIATT_N_HEADS"
ENV_N_KV_HEADS = "VLLM_TRIATT_N_KV_HEADS"
ENV_HEAD_DIM = "VLLM_TRIATT_HEAD_DIM"
ENV_N_ROT = "VLLM_TRIATT_N_ROT"
ENV_ROPE_THETA = "VLLM_TRIATT_ROPE_THETA"

logger = init_logger(__name__)

# Module-level singleton engine (one per process).
_engine: Optional[TriAttentionV3Engine] = None
_lazy_init_attempted: bool = False


def set_engine(engine: Optional[TriAttentionV3Engine]) -> None:
    global _engine
    _engine = engine


def get_engine() -> Optional[TriAttentionV3Engine]:
    return _engine


def is_enabled() -> bool:
    return _engine is not None or os.environ.get(ENV_ENABLED, "0") == "1"


def export_config_to_env(
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_rot: int,
    rope_theta: float,
) -> None:
    """Called by install_triattention in the main process."""
    os.environ[ENV_ENABLED] = "1"
    os.environ[ENV_N_LAYERS] = str(n_layers)
    os.environ[ENV_N_HEADS] = str(n_heads)
    os.environ[ENV_N_KV_HEADS] = str(n_kv_heads)
    os.environ[ENV_HEAD_DIM] = str(head_dim)
    os.environ[ENV_N_ROT] = str(n_rot)
    os.environ[ENV_ROPE_THETA] = str(rope_theta)


def _lazy_init_from_env(device: torch.device) -> None:
    """Worker-side: build engine on first Q capture, reading env vars."""
    global _engine, _lazy_init_attempted
    _lazy_init_attempted = True
    if os.environ.get(ENV_ENABLED, "0") != "1":
        return
    cfg = TriAttentionV3Config.from_env()
    _engine = TriAttentionV3Engine(
        cfg=cfg,
        n_layers=int(os.environ[ENV_N_LAYERS]),
        n_heads=int(os.environ[ENV_N_HEADS]),
        n_kv_heads=int(os.environ[ENV_N_KV_HEADS]),
        head_dim=int(os.environ[ENV_HEAD_DIM]),
        rope_theta=float(os.environ[ENV_ROPE_THETA]),
        n_rot=int(os.environ[ENV_N_ROT]),
        device=device,
    )
    logger.info(
        "TriAttention V3 worker init: layers=%d heads=%d kv=%d head_dim=%d "
        "n_rot=%d theta=%.1f budget=%d window=%d prefix=%d warmup=%d",
        _engine.n_layers, _engine.n_heads, _engine.n_kv_heads,
        _engine.head_dim, _engine.n_rot, _engine.rope_theta,
        cfg.budget, cfg.window_size, cfg.prefix_protect, cfg.warmup_tokens,
    )


def _capture_q_impl(q: torch.Tensor, layer_idx: int) -> None:
    """Inner implementation; signature matches the registered torch op
    (Tensor first, int second — required for dispatch key inference).

    No-op when V3 is disabled. Lazy-initialises the engine in the worker
    process on first call, reading config from VLLM_TRIATT_* env vars.
    """
    if _engine is None:
        if _lazy_init_attempted:
            return
        _lazy_init_from_env(q.device)
        if _engine is None:
            return
    eng = _engine
    if eng.calibrated and not eng.cfg.adaptive_calibration:
        return
    if q.ndim == 2:
        if q.shape[1] != eng.n_heads * eng.head_dim:
            return
        q = q.view(-1, eng.n_heads, eng.head_dim)
    elif q.ndim == 3:
        if q.shape[1] != eng.n_heads or q.shape[2] != eng.head_dim:
            return
    else:
        return
    eng.accumulate_q(q.detach(), int(layer_idx))


def _capture_q_fake(q: torch.Tensor, layer_idx: int) -> None:
    """Fake/meta impl: no-op with no tensor allocation, satisfies dynamo."""
    return None


# Register as a vLLM custom op so torch.compile treats the call as opaque
# (no graph break in fullgraph mode). The op has only side-effects on a
# host-side dict; no tensor outputs.
_REGISTERED = False


def _ensure_op_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    # mutates_args=["q"] is a claim, not a fact: we read q only. Declaring
    # the mutation prevents torch.compile from DCE'ing the op as dead code
    # (it has no tensor outputs and no real mutations). The op is purely a
    # host-side side effect (Q stat accumulation) so torch never sees a
    # value flowing out of it.
    direct_register_custom_op(
        op_name="triatt_capture_q_pre_rope",
        op_func=_capture_q_impl,
        mutates_args=["q"],
        fake_impl=_capture_q_fake,
    )
    _REGISTERED = True


_ensure_op_registered()


def capture_q_pre_rope(layer_idx: int, q: torch.Tensor) -> None:
    """Emit pre-RoPE Q to the V3 engine. No-op when V3 is disabled.

    q: shape [n_tokens, n_heads * head_dim] OR [n_tokens, n_heads, head_dim].
    """
    torch.ops.vllm.triatt_capture_q_pre_rope(q, layer_idx)
