"""Top-level entry point for installing TriAttention V3 in vLLM.

Usage (must run BEFORE LLM() so env vars are inherited by the worker fork):

    from vllm import LLM
    from vllm.v1.attention.triattention import (
        TriAttentionV3Config,
        install_triattention,
    )

    install_triattention(
        model_path="/path/to/Qwen2.5-7B",
        cfg=TriAttentionV3Config(budget=29491),
    )
    llm = LLM(model="/path/to/Qwen2.5-7B", kv_cache_dtype="turboquant_k8v4", ...)
    # The worker process inherits VLLM_TRIATT_* env vars and lazy-initialises
    # the engine on the first Q-capture call.

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

import os
from typing import Optional

from transformers import AutoConfig

from vllm.logger import init_logger
from vllm.v1.attention.triattention.engine import TriAttentionV3Config
from vllm.v1.attention.triattention.hooks import (
    export_config_to_env,
    set_engine,
)

logger = init_logger(__name__)


def _resolve_dims_from_hf(model_path: str) -> dict:
    """Resolve attention dims from a HuggingFace model directory."""
    hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(hf, "text_config"):
        hf = hf.text_config
    n_layers = hf.num_hidden_layers
    n_heads = hf.num_attention_heads
    n_kv_heads = getattr(hf, "num_key_value_heads", n_heads)
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // n_heads
    rope_theta = float(getattr(hf, "rope_theta", 10000.0))
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
    model_path: str,
    cfg: Optional[TriAttentionV3Config] = None,
) -> dict:
    """Wire the TriAttention V3 engine into a vLLM LLM instance.

    vLLM V1 forks an EngineCore subprocess that owns model execution; the
    V3 engine has to live there. This call exports model dims + cfg via
    env vars (which the worker inherits) and the worker lazy-initialises
    its engine on the first Q-capture.

    Returns the resolved model dims as a dict (caller can introspect; the
    engine handle itself lives in the worker process).

    To override defaults, set VLLM_TRIATT_BUDGET / _PREFIX / _WINDOW /
    _SEGMENTS / _WARMUP / _ADAPTIVE / _HYBRID before calling LLM(...).
    """
    cfg = cfg or TriAttentionV3Config.from_env()
    dims = _resolve_dims_from_hf(model_path)
    export_config_to_env(
        n_layers=dims["n_layers"],
        n_heads=dims["n_heads"],
        n_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        n_rot=dims["n_rot"],
        rope_theta=dims["rope_theta"],
    )
    # Echo cfg knobs into env so the worker reads the same values.
    os.environ["VLLM_TRIATT_BUDGET"] = str(cfg.budget)
    os.environ["VLLM_TRIATT_HYBRID"] = str(cfg.hybrid_mode)
    os.environ["VLLM_TRIATT_PREFIX"] = str(cfg.prefix_protect)
    os.environ["VLLM_TRIATT_WINDOW"] = str(cfg.window_size)
    os.environ["VLLM_TRIATT_SEGMENTS"] = str(cfg.n_segments)
    os.environ["VLLM_TRIATT_WARMUP"] = str(cfg.warmup_tokens)
    logger.info(
        "TriAttention V3 config exported. layers=%d heads=%d kv=%d "
        "head_dim=%d n_rot=%d theta=%.1f budget=%d window=%d prefix=%d "
        "warmup=%d. Worker will lazy-init on first Q capture.",
        dims["n_layers"], dims["n_heads"], dims["n_kv_heads"],
        dims["head_dim"], dims["n_rot"], dims["rope_theta"],
        cfg.budget, cfg.window_size, cfg.prefix_protect, cfg.warmup_tokens,
    )
    return dims


def uninstall_triattention() -> None:
    """Disable V3. Effects in worker only after the next forward pass."""
    os.environ.pop("VLLM_TRIATT_ENABLED", None)
    set_engine(None)
