"""Top-level entry point for installing TriAttention V3 in vLLM.

V3 is enabled per-process via env vars. The worker inherits them from the
parent fork and lazy-initialises its engine on the first Q-capture call,
reading model dims from the active VllmConfig and tuning knobs from
VLLM_TRIATT_* env vars.

Two equivalent ways to enable:

1. Set env vars before constructing LLM():

    VLLM_TRIATT_ENABLED=1 \
    VLLM_TRIATT_BUDGET=29491 \
    VLLM_TRIATT_PREFIX=128 \
    python my_script.py

2. Use the helper, which sets the same env vars from a config object:

    from vllm import LLM
    from vllm.v1.attention.triattention import (
        TriAttentionV3Config, install_triattention,
    )
    install_triattention(TriAttentionV3Config(budget=29491))
    llm = LLM(model="/path/to/Qwen2.5-7B", kv_cache_dtype="turboquant_k8v4")

The helper must run before LLM() so the worker fork inherits the env vars.

Tuning knobs (all integer / boolean unless noted):
  VLLM_TRIATT_ENABLED   - master switch ("0" / "1", default "0")
  VLLM_TRIATT_BUDGET    - max live cells per sequence (default 2048)
  VLLM_TRIATT_HYBRID    - V1 / V2 / V3 selection mode (default 2 = V3)
  VLLM_TRIATT_PREFIX    - protected prefix length, V3 only (default 128)
  VLLM_TRIATT_WINDOW    - protected recent window length (default 128)
  VLLM_TRIATT_SEGMENTS  - per-segment quota bucket count (default 8)
  VLLM_TRIATT_WARMUP    - Q samples before calibration fires (default 1024)
  VLLM_TRIATT_ADAPTIVE  - update calibration centers via EMA each round
                          ("0" / "1", default "0")

V3 only composes with KV-cache presets that store K in a dequant-able
form. The validated path is `kv_cache_dtype="turboquant_k8v4"` (FP8 K +
4-bit V), which mirrors the K=q8_0 + V=turbo3 config the V3 paper validated
on llama.cpp. Pure-BF16 K is not yet supported because it bypasses the
TurboQuant attention backend where the V3 hooks live.
"""
from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.attention.triattention.engine import TriAttentionV3Config
from vllm.v1.attention.triattention.hooks import ENV_ENABLED, set_engine

logger = init_logger(__name__)


def install_triattention(
    cfg: Optional[TriAttentionV3Config] = None,
) -> None:
    """Mark TriAttention V3 enabled and export tuning knobs as env vars.

    Must run before constructing the vLLM `LLM(...)` instance so the
    EngineCore subprocess fork inherits the env vars. Model dims are
    resolved inside the worker from the active VllmConfig, so no model
    path is required here.
    """
    cfg = cfg or TriAttentionV3Config.from_env()
    os.environ[ENV_ENABLED] = "1"
    os.environ["VLLM_TRIATT_BUDGET"] = str(cfg.budget)
    os.environ["VLLM_TRIATT_HYBRID"] = str(cfg.hybrid_mode)
    os.environ["VLLM_TRIATT_PREFIX"] = str(cfg.prefix_protect)
    os.environ["VLLM_TRIATT_WINDOW"] = str(cfg.window_size)
    os.environ["VLLM_TRIATT_SEGMENTS"] = str(cfg.n_segments)
    os.environ["VLLM_TRIATT_WARMUP"] = str(cfg.warmup_tokens)
    if cfg.adaptive_calibration:
        os.environ["VLLM_TRIATT_ADAPTIVE"] = "1"
    logger.info(
        "TriAttention V3 enabled. budget=%d hybrid=%d prefix=%d window=%d "
        "segments=%d warmup=%d. Worker will lazy-init on first Q capture.",
        cfg.budget, cfg.hybrid_mode, cfg.prefix_protect, cfg.window_size,
        cfg.n_segments, cfg.warmup_tokens,
    )


def uninstall_triattention() -> None:
    """Disable V3 for this process. Worker takes effect on next pass."""
    os.environ.pop(ENV_ENABLED, None)
    set_engine(None)
