"""TriAttention V3: self-calibrating trigonometric KV-cache token eviction.

Independent vLLM port of the implementation in
`TheTom/llama-cpp-turboquant @ experiment/triattention-integration`. See
`docs/papers/triattention-v3.md` for the algorithm, validation envelope, and
discussion of the V1/V2/V3 progression.

Public API:
  - TriAttentionV3Engine: per-(batch, sequence) eviction state + scoring
  - install_triattention(...): top-level entry point that hooks Q capture
    and registers the engine with the attention metadata builder.
"""
from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)
from vllm.v1.attention.triattention.integration import install_triattention

__all__ = [
    "TriAttentionV3Config",
    "TriAttentionV3Engine",
    "install_triattention",
]
