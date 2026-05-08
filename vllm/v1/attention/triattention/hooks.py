"""Q-capture hooks for TriAttention V3.

The engine needs the *pre-RoPE* Q tensor at every attention layer. vLLM's
attention layer (`vllm.model_executor.layers.attention.Attention.forward`)
receives Q *after* RoPE, so the capture has to live earlier, inside the
model's attention block, just before its `rotary_emb(...)` call.

Mirrors the llama.cpp impl, which emits Q via the ggml scheduler's eval
callback. This module exposes a tiny explicit-emit API:

    from vllm.v1.attention.triattention.hooks import capture_q_pre_rope
    ...
    # In each model's attention forward, right before RoPE:
    capture_q_pre_rope(self.layer_idx, q_view)
    q, k = self.rotary_emb(positions, q, k)

vLLM V1 forks an EngineCore subprocess for model execution, so the engine
must live in the worker. The worker lazy-initialises its engine on the
first Q-capture call. Model dims (n_layers, heads, kv_heads, head_dim,
rope_theta) are read from the active `VllmConfig` so the user doesn't
have to pass them. Tuning knobs (budget, prefix, window, etc.) come from
VLLM_TRIATT_* env vars (see `TriAttentionV3Config.from_env`).

When V3 is not enabled (`VLLM_TRIATT_ENABLED != "1"`), `capture_q_pre_rope`
is a tight no-op.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)

ENV_ENABLED = "VLLM_TRIATT_ENABLED"

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


def _resolve_dims_from_env() -> Optional[dict]:
    """Fallback: read architecture dims from VLLM_TRIATT_* env vars.

    The primary path (`get_current_vllm_config()`) raises AssertionError
    when called from inside the worker's torch-op dispatch context (the
    config is set per-thread and the dispatcher fires on a fresh thread).
    This fallback lets the user pin dims directly, bypassing the assert.
    Required vars: N_LAYERS / N_HEADS / N_KV_HEADS / HEAD_DIM. Optional:
    ROPE_THETA (default 10000), N_ROT (default = HEAD_DIM).
    """
    n_layers = os.environ.get("VLLM_TRIATT_N_LAYERS")
    n_heads = os.environ.get("VLLM_TRIATT_N_HEADS")
    if not n_layers or not n_heads:
        return None
    head_dim = int(os.environ.get("VLLM_TRIATT_HEAD_DIM", "128"))
    return {
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_kv_heads": int(
            os.environ.get("VLLM_TRIATT_N_KV_HEADS", n_heads)
        ),
        "head_dim": head_dim,
        "rope_theta": float(os.environ.get("VLLM_TRIATT_ROPE_THETA", "10000.0")),
        "n_rot": int(os.environ.get("VLLM_TRIATT_N_ROT", str(head_dim))),
    }


def _resolve_dims_from_vllm_config() -> Optional[dict]:
    """Pull architecture dims out of the active VllmConfig.

    Called from inside the worker on first Q capture. Falls back to
    `_resolve_dims_from_env` when `get_current_vllm_config()` raises
    AssertionError (the config is per-thread and the worker's torch-op
    dispatch fires on a thread without it set). Returns None when both
    paths fail; engine init then logs a warning and stays disabled.
    """
    try:
        vllm_cfg = get_current_vllm_config()
    except AssertionError:
        return _resolve_dims_from_env()
    if vllm_cfg is None:
        return _resolve_dims_from_env()
    hf = vllm_cfg.model_config.hf_text_config
    n_layers = hf.num_hidden_layers
    n_heads = hf.num_attention_heads
    n_kv_heads = getattr(hf, "num_key_value_heads", n_heads)
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // n_heads
    # Llama 4 / some Llama variants store rope_theta inside the
    # `rope_parameters` dict instead of as a top-level attribute.
    # Fall back to that shape before defaulting to 10000.0 (the latter
    # is RIGHT for early Llama / Qwen2 32K, WRONG for Llama 4 = 500000
    # and Qwen2.5 1M = 10000000). Env override wins.
    env_theta = os.environ.get("VLLM_TRIATT_ROPE_THETA")
    if env_theta is not None:
        rope_theta = float(env_theta)
    else:
        rope_theta = getattr(hf, "rope_theta", None)
        if rope_theta is None:
            rp = getattr(hf, "rope_parameters", None)
            if isinstance(rp, dict):
                rope_theta = rp.get("rope_theta")
        rope_theta = float(rope_theta) if rope_theta is not None else 10000.0
    partial = getattr(hf, "partial_rotary_factor", None)
    n_rot = head_dim if partial is None else int(round(head_dim * float(partial)))
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "n_rot": n_rot,
    }


def _lazy_init(device: torch.device) -> None:
    """Worker-side: build engine on first Q capture.

    Tuning config comes from VLLM_TRIATT_* env vars
    (see TriAttentionV3Config.from_env). Model dims come from the active
    VllmConfig.
    """
    global _engine, _lazy_init_attempted
    if os.environ.get(ENV_ENABLED, "0") != "1":
        _lazy_init_attempted = True
        return
    dims = _resolve_dims_from_vllm_config()
    if dims is None:
        # Don't latch on a transient config-not-available error. The first
        # Q-capture often fires from a thread where the per-thread VllmConfig
        # isn't set yet but a later Q-capture has it. Latching here meant
        # one early miss permanently disabled V3 for the worker. Now we
        # only latch after we've SUCCESSFULLY built the engine OR after
        # we've explicitly decided V3 is off (the early-return above).
        logger.warning(
            "TriAttention V3 enabled but VllmConfig is not available at "
            "this Q-capture; will retry on the next call. Set "
            "VLLM_TRIATT_N_LAYERS / N_HEADS / HEAD_DIM env vars as a "
            "fallback if this keeps firing."
        )
        return
    _lazy_init_attempted = True
    cfg = TriAttentionV3Config.from_env()
    _engine = TriAttentionV3Engine(
        cfg=cfg,
        n_layers=dims["n_layers"],
        n_heads=dims["n_heads"],
        n_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        rope_theta=dims["rope_theta"],
        n_rot=dims["n_rot"],
        device=device,
    )
    logger.info(
        "TriAttention V3 worker init: layers=%d heads=%d kv=%d head_dim=%d "
        "n_rot=%d theta=%.1f budget=%d window=%d prefix=%d warmup=%d",
        _engine.n_layers, _engine.n_heads, _engine.n_kv_heads,
        _engine.head_dim, _engine.n_rot, _engine.rope_theta,
        cfg.budget, cfg.window_size, cfg.prefix_protect, cfg.warmup_tokens,
    )

    # Auto-install Tier 2 callback at engine init — without this, the
    # rescue-callback registration only happens via unit tests, so the
    # eviction store stays empty in production and Tier 3 retrieval has
    # nothing to surface. Detected by the AMD E2E rescue agents on
    # 2026-05-07.
    #
    # Tokenizer binding does NOT happen here — `get_current_vllm_config()`
    # raises an AssertionError when called from a custom-op forward
    # dispatch (the per-thread vllm-config context isn't set on this
    # thread). The tokenizer load runs from the worker's add_requests
    # path instead, where `self.vllm_config` is a real attr. See
    # vllm/v1/worker/gpu/model_runner.py:add_requests.
    try:
        from vllm.v1.attention.triattention.backend_helpers import (
            install_eviction_to_longctx,
        )
        installed = install_eviction_to_longctx()
        if installed:
            logger.info(
                "TriAttention V3 Tier 2 wired: eviction callback registered "
                "and pointing at LONGCTX_ENDPOINT. Tokenizer binding will "
                "happen on first add_requests (worker path)."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TriAttention V3 Tier 2 auto-install failed: %s — V3 still works "
            "but evict-to-longctx rescue is disabled.", exc,
        )


def _capture_q_impl(q: torch.Tensor, layer_idx: int) -> None:
    """Inner implementation; signature matches the registered torch op
    (Tensor first, int second — required for dispatch key inference).

    No-op when V3 is disabled. Lazy-initialises the engine in the worker
    process on first call, reading config from VLLM_TRIATT_* env vars.
    """
    if _engine is None:
        # Retry init each call until it succeeds or VLLM_TRIATT_ENABLED=0
        # is observed (which sets the latch). The previous "latch on first
        # attempt" behaviour permanently disabled V3 if the first call
        # raced ahead of VllmConfig being available in the worker thread.
        if _lazy_init_attempted:
            return
        _lazy_init(q.device)
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
    # ROCm round-6 stream-poison fix (2026-05-07): the cumulative
    # `.to(torch.float32)` casts inside accumulate_q + _capture_query_q
    # leave the HIP runtime in an error state by chunk 8 of chunked
    # prefill, which surfaces as `SetDevice` failure inside the next
    # kernel (vLLM's `rotary_embedding`). Forcing a synchronize at the
    # hook boundary either clears the error or surfaces it deterministically
    # inside V3 for triage instead of poisoning the model's RoPE op.
    # No-op on CPU; skip to keep CPU-path tests fast.
    if os.environ.get("VLLM_TRIATT_HOOK_SYNC", "1") == "1":
        if torch.cuda.is_available() and q.device.type == "cuda":
            torch.cuda.synchronize(device=q.device)


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
