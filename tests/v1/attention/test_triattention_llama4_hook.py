# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity tests for the TriAttention V3 hook port to Llama 4.

Pure-Python: verifies the source-level wiring of `capture_q_pre_rope` in
`vllm/model_executor/models/llama4.py` and exercises a synthetic Q tensor
through the registered torch op + engine to confirm the Llama 4 forward
signature (qkv split → pre-rotary capture) works for both the chunked-local
RoPE layers and the global (NoPE) layers.

Heavy vLLM imports (distributed / Triton / cbor2) are stubbed the same way
test_triattention_rescue.py does, so this runs on a dev box without the
full vLLM dep tree.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub the heavy bits the hooks module pulls in transitively (mirrors
# test_triattention_rescue.py). MUST run before importing triattention.*.
# ---------------------------------------------------------------------------
try:
    import vllm.config  # noqa: F401
except ModuleNotFoundError:
    _stub_cfg = types.ModuleType("vllm.config")
    _stub_cfg.get_current_vllm_config = lambda: None
    sys.modules["vllm.config"] = _stub_cfg

try:
    from vllm.model_executor.models.utils import extract_layer_index  # noqa
except (ModuleNotFoundError, ImportError):
    import re as _re

    def _extract_layer_index(layer_name: str) -> int:
        m = _re.search(r"\.(\d+)\.", layer_name)
        return int(m.group(1)) if m else 0

    _stub_utils = types.ModuleType("vllm.model_executor.models.utils")
    _stub_utils.extract_layer_index = _extract_layer_index
    for pkg in ("vllm.model_executor", "vllm.model_executor.models"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    sys.modules["vllm.model_executor.models.utils"] = _stub_utils

from vllm.v1.attention.triattention import hooks  # noqa: E402
from vllm.v1.attention.triattention.engine import (  # noqa: E402
    TriAttentionV3Config,
    TriAttentionV3Engine,
)


LLAMA4_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm" / "model_executor" / "models" / "llama4.py"
)


# ---------------------------------------------------------------------------
# Source-level wiring checks. The forward must call `capture_q_pre_rope`
# AFTER `qkv.split` and BEFORE `self.rotary_emb(...)`. These checks exist
# so a future refactor that reorders the forward doesn't silently break
# the V3 hook on Llama 4.
# ---------------------------------------------------------------------------


def _llama4_attention_forward_src() -> str:
    src = LLAMA4_PATH.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Llama4Attention":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "forward":
                    return ast.unparse(sub)
    raise AssertionError("Llama4Attention.forward not found in llama4.py")


class TestLlama4SourceWiring:
    def test_import_present(self):
        src = LLAMA4_PATH.read_text()
        assert (
            "from vllm.v1.attention.triattention.hooks import capture_q_pre_rope"
            in src
        ), "capture_q_pre_rope import missing from llama4.py"

    def test_triatt_layer_idx_attribute_set(self):
        src = LLAMA4_PATH.read_text()
        assert "self._triatt_layer_idx = self.layer_idx" in src, (
            "Llama4Attention.__init__ must set self._triatt_layer_idx for the "
            "V3 hook to know which layer it is on."
        )

    def test_capture_called_in_forward_before_rotary(self):
        forward_src = _llama4_attention_forward_src()
        assert "capture_q_pre_rope(self._triatt_layer_idx, q)" in forward_src
        # Capture MUST come before rotary_emb so we feed pre-RoPE Q.
        cap_pos = forward_src.index("capture_q_pre_rope(self._triatt_layer_idx, q)")
        rope_pos = forward_src.index("self.rotary_emb(positions, q, k)")
        assert cap_pos < rope_pos, (
            "capture_q_pre_rope must run BEFORE self.rotary_emb on Llama 4 "
            "(otherwise V3 sees post-RoPE Q on chunked-local layers)."
        )
        # Capture MUST come after qkv.split so q is the post-projection
        # Q (not the fused qkv).
        split_pos = forward_src.index("qkv.split(")
        assert split_pos < cap_pos


# ---------------------------------------------------------------------------
# Engine integration: drive the hook with a Llama 4-shaped Q tensor and
# verify the engine's calibration accumulator picks it up correctly. Llama 4
# Scout text config: heads=40, kv_heads=8, head_dim=128. We use a smaller
# but Llama 4-shaped config (4 KV groups, 4 heads-per-group, head_dim=128)
# to keep the test cheap.
# ---------------------------------------------------------------------------


def _make_engine(
    n_layers: int = 4,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    head_dim: int = 128,
    rope_theta: float = 500_000.0,  # Llama 4 default
    warmup_tokens: int = 32,
) -> TriAttentionV3Engine:
    cfg = TriAttentionV3Config(
        budget=64,
        prefix_protect=8,
        window_size=8,
        n_segments=4,
        warmup_tokens=warmup_tokens,
        hybrid_mode=2,
    )
    return TriAttentionV3Engine(
        cfg=cfg,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )


@pytest.fixture(autouse=True)
def _isolate_engine():
    hooks.set_engine(None)
    yield
    hooks.set_engine(None)


class TestLlama4HookIntegration:
    """Mirrors what Llama4Attention.forward does to Q, then drives the hook."""

    def test_hook_accepts_llama4_q_shape_2d(self):
        """Llama4 forward keeps Q as [n_tokens, q_size] right before
        rotary_emb. The hook must accept this 2D layout."""
        eng = _make_engine()
        hooks.set_engine(eng)
        n_tokens = eng.cfg.warmup_tokens
        q = torch.randn(n_tokens, eng.n_heads * eng.head_dim)
        # Layer 0 is a chunked-local RoPE layer in Llama 4 Scout
        # (no_rope_layers[0] == 1).
        hooks.capture_q_pre_rope(layer_idx=0, q=q)
        assert eng.calibrated, (
            "engine should reach calibration after warmup_tokens of Q"
        )

    def test_hook_fires_on_nope_layer(self):
        """Llama 4 NoPE layer index 3 (no_rope_layers[3] == 0). The hook
        is called with the same Q shape as RoPE layers — pre-RoPE == post-RoPE
        on these layers because rotary_emb doesn't run."""
        eng = _make_engine(warmup_tokens=16)
        hooks.set_engine(eng)
        n_tokens = 16
        q = torch.randn(n_tokens, eng.n_heads, eng.head_dim)
        # Layer 3 is the first NoPE / global-attention layer in Llama 4 Scout.
        hooks.capture_q_pre_rope(layer_idx=3, q=q)
        assert eng.calibrated

    def test_hook_noop_when_engine_not_set(self):
        """V3 disabled path: capture_q_pre_rope must be a fast no-op."""
        hooks.set_engine(None)
        # Pretend the engine never lazy-inits (latch).
        hooks._lazy_init_attempted = True
        try:
            q = torch.randn(8, 16 * 128)
            # Should not raise.
            hooks.capture_q_pre_rope(layer_idx=0, q=q)
        finally:
            hooks._lazy_init_attempted = False
