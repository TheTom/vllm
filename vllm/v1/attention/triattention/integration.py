"""Top-level entry point for installing TriAttention V3 in vLLM.

Usage:

    from vllm import LLM
    from vllm.v1.attention.triattention import (
        TriAttentionV3Config,
        install_triattention,
    )

    llm = LLM(model=..., kv_cache_dtype="auto", ...)
    install_triattention(llm, TriAttentionV3Config(budget=29491))

Two integration paths supported:

  1. **BF16 KV cache** (Phase A path): no kernel changes needed for scoring;
     V3 reads K directly from the cache via the engine's own dequant. Eviction
     applies a per-position validity mask via attention metadata. The mask is
     consumed by the TurboQuant backend (we add VALID_MASK there); on the
     stock BF16 path the mask is currently approximated by writing -inf into
     evicted slots' K-norm — Phase A defers this to a follow-up if needed.

  2. **TQ+ KV cache** (Phase C path): K is quantized but readable via the
     existing TQ+ K dequant helper. V3 reads K through that path.
"""
from __future__ import annotations

from typing import Optional

import torch

from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)
from vllm.v1.attention.triattention.hooks import set_engine


def _resolve_model_dims(llm) -> dict:
    """Pull architecture dims out of a vLLM LLM handle (V1 engine)."""
    cfg = llm.llm_engine.vllm_config
    mc = cfg.model_config
    hf = mc.hf_text_config

    n_layers = hf.num_hidden_layers
    n_heads = hf.num_attention_heads
    n_kv_heads = getattr(hf, "num_key_value_heads", n_heads)
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // n_heads
    rope_theta = float(getattr(hf, "rope_theta", 10000.0))
    # Partial RoPE: some models rotate only a fraction of head_dim
    partial = getattr(hf, "partial_rotary_factor", None)
    if partial is None:
        n_rot = head_dim
    else:
        n_rot = int(round(head_dim * float(partial)))
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "n_rot": n_rot,
    }


def install_triattention(
    llm,
    cfg: Optional[TriAttentionV3Config] = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> TriAttentionV3Engine:
    """Wire the TriAttention V3 engine into a vLLM LLM instance.

    Effects:
      - Builds the engine (RoPE constants, accumulators).
      - Installs the global hook so model attention forwards push pre-RoPE Q.
      - Returns the engine handle (caller can read .stats() etc.).

    The engine becomes active the moment install returns. Calibration runs
    automatically once `q_samples` crosses `cfg.warmup_tokens`. Eviction is
    triggered by a separate caller (typically the runtime after each forward
    pass) — V3 doesn't intercept the scheduler itself.
    """
    cfg = cfg or TriAttentionV3Config.from_env()
    dims = _resolve_model_dims(llm)
    dev = torch.device(device) if isinstance(device, str) else device
    eng = TriAttentionV3Engine(
        cfg=cfg,
        n_layers=dims["n_layers"],
        n_heads=dims["n_heads"],
        n_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        rope_theta=dims["rope_theta"],
        n_rot=dims["n_rot"],
        device=dev,
        dtype=dtype,
    )
    set_engine(eng)
    print(
        f"[TriAttention V3] installed. "
        f"layers={dims['n_layers']} heads={dims['n_heads']} kv={dims['n_kv_heads']} "
        f"head_dim={dims['head_dim']} n_rot={dims['n_rot']} theta={dims['rope_theta']:.1f} "
        f"budget={cfg.budget} window={cfg.window_size} prefix={cfg.prefix_protect} "
        f"warmup={cfg.warmup_tokens}"
    )
    return eng


def uninstall_triattention() -> None:
    set_engine(None)
