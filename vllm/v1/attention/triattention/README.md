# TriAttention V3

Independent vLLM port of the V3 KV-cache eviction policy from
`docs/papers/triattention-v3.md` and the
`experiment/triattention-integration` branch of
`TheTom/llama-cpp-turboquant`. Same algorithm, same calibration, same V3
selection logic, ported to Triton + vLLM's attention metadata.

## What it does

V3 is a *self-calibrating trigonometric KV cache token eviction*
policy. It scores cells by how aligned each token's K vector is with
the calibrated query distribution, then evicts the lowest-scoring
tokens to keep the cache under a budget. Two structural rules sit on
top of the trig score:

  - the first `prefix_protect` tokens are never evicted
  - the most recent `window_size` tokens are never evicted
  - what remains is bucketed into `n_segments` position quotas so
    eviction spreads evenly across the middle of context

Compression-wise, V3 is the *fewer tokens in the cache* axis. It stacks
cleanly with TurboQuant's *fewer bytes per token* axis: V3 at 90%
retention plus TQ K=FP8 + V=4-bit gives ~2.96x KV memory reduction with
no measured PPL or NIAH cost on the validated envelope.

## Validated envelope

  - Qwen2.5-7B-Instruct at 32K, paper-exact protocol: V3 at-baseline
    within ±0.5% PPL noise floor, TQ+V3 stack at-baseline.
  - Qwen3-8B at 8K / 16K / 32K: V3 at-baseline-or-better, NIAH at 32K
    PASS at start / middle / end on all four modes
    (baseline / TQ / V3 / stack).
  - Mistral-7B-Instruct-v0.3 at 32K: V3 +0.10%, stack +0.11%.
  - Qwen3-30B-A3B (MoE) at 32K: V3 +0.34% (stack hits a known second-run
    infrastructure flakiness; first-run lands cleanly).

## Outside the validated envelope

  - Hybrid Mamba+Attention models (e.g. Qwen3.5-class). Paper documents
    NIAH failure at middle / end positions on these even when PPL looks
    fine. Don't auto-enable.
  - Reasoning workloads, multi-needle retrieval, 128K+ context. Untested.
  - Aggressive retention (<90%). Paper saw partial NIAH at 85%; tighter
    is documented to break.

## Enabling

V3 is per-process. Set env vars before constructing `LLM(...)`:

    VLLM_TRIATT_ENABLED=1 \
    VLLM_TRIATT_BUDGET=29491 \
    VLLM_TRIATT_PREFIX=128 \
    VLLM_TRIATT_WINDOW=128 \
    python my_script.py

Or use the helper, which sets the same env vars from a config object:

    from vllm import LLM
    from vllm.v1.attention.triattention import (
        TriAttentionV3Config, install_triattention,
    )

    install_triattention(
        TriAttentionV3Config(budget=29491, prefix_protect=128)
    )
    llm = LLM(
        model="/path/to/Qwen2.5-7B-Instruct",
        kv_cache_dtype="turboquant_k8v4",
        max_model_len=32768,
    )

`install_triattention` must run before `LLM(...)` so the EngineCore
subprocess fork inherits the env vars.

## Tunable knobs

| Env var                  | Default | Meaning                                          |
| ------------------------ | ------: | ------------------------------------------------ |
| `VLLM_TRIATT_ENABLED`    |     `0` | master switch (`"0"` / `"1"`)                    |
| `VLLM_TRIATT_BUDGET`     |  `2048` | max live cells per sequence                      |
| `VLLM_TRIATT_HYBRID`     |     `2` | selection mode: `0`=V1, `1`=V2, `2`=V3           |
| `VLLM_TRIATT_PREFIX`     |   `128` | protected prefix length (V3 only)                |
| `VLLM_TRIATT_WINDOW`     |   `128` | protected recent-window length                   |
| `VLLM_TRIATT_SEGMENTS`   |     `8` | per-segment quota bucket count                   |
| `VLLM_TRIATT_WARMUP`     |  `1024` | Q samples before calibration fires               |
| `VLLM_TRIATT_ADAPTIVE`   |     `0` | EMA-update calibration centers each round        |
| `VLLM_TRIATT_EXPECTED_LAYERS` | unset | score-hook layer count before eviction finalizes; use when a backend intentionally skips boundary layers |

## Supported KV cache presets

V3 needs a dequant-able K (the engine reads it for scoring). The
validated path is `kv_cache_dtype="turboquant_k8v4"` (FP8 K + 4-bit V),
which mirrors the K=q8_0 + V=turbo3 config the V3 paper validated on
llama.cpp. Pure-BF16 K is not yet supported because it bypasses the
TurboQuant attention backend where the V3 hooks live. Other TQ presets
that store K in dequant-able form should also work.

For grouped (batched-Q) GQA dispatch, `kv_group = Hq / Hk` must be a
power of 2. Models with non-pow2 group sizes (e.g. Qwen2.5-7B has
Hq=28 / Hk=4 → kv_group=7) automatically fall back to single-Q
dispatch; no correctness issue, just leaves the +29-39% grouped-kernel
decode boost on the table for those models.

## Architecture

  - `engine.py` — `TriAttentionV3Engine`, calibration accumulators,
    per-layer score state, V3 selection trigger.
  - `policy.py` — V1 / V2 / V3 selection logic.
  - `scoring.py`, `scoring_kernel.py` — PyTorch reference + Triton
    kernel for the per-cell trig scoring.
  - `hooks.py` — Q-capture custom op (`vllm::triatt_capture_q_pre_rope`)
    + worker-side lazy init from `VllmConfig`.
  - `integration.py` — `install_triattention` helper (sets env vars).
  - `backend_helpers.py` — glue between the engine and the TurboQuant
    attention backend (per-layer K accumulation, validity-mask build).

## Implementation notes

vLLM V1 forks an EngineCore subprocess that owns model execution; the
V3 engine has to live there. Q capture is a `vllm::` custom op so
torch.compile fullgraph treats it as opaque. Eviction is mask-based:
we store a uint8 `valid_mask[B, max_seq_len]` in attention metadata,
and the TQ decode kernels (`_tq_decode_stage1` and
`_tq_decode_stage1_grouped`) score evicted positions as `-inf` via a
`VALID_MASK` constexpr. No physical block movement. Pass-completion is
detected by the engine watching for a layer-index that's already
accumulated this round.

Phase A scope is single-sequence batches (PPL / NIAH research workloads).
Multi-batch needs request-id plumbing through `CommonAttentionMetadata`
and is deferred.

## References

  - Paper: `TheTom/turboquant_plus/docs/papers/triattention-v3.md`
  - Source impl: `TheTom/llama-cpp-turboquant`,
    `experiment/triattention-integration` branch,
    `src/llama-triattention.cpp`
  - Original TriAttention paper: Mao et al., arXiv:2604.04921 (2026)
