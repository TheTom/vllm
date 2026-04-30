# Usage — `feature/turboquant_plus`

Two opt-in features sit on top of upstream vLLM in this fork:

1. **TurboQuant+ KV cache** — adds extra K/V quantization presets (asymmetric bit widths, centroid-V, 2-bit) on top of the upstream TurboQuant work in [#38479](https://github.com/vllm-project/vllm/pull/38479).
2. **TriAttention V3** — a self-calibrating trigonometric KV-cache token eviction policy. Stacks with TurboQuant+.

Everything else is upstream vLLM and works as documented at [docs.vllm.ai](https://docs.vllm.ai).

---

## Quick start

```bash
# TurboQuant+ KV cache only
vllm serve <model> --kv-cache-dtype turboquant_k8v4

# TurboQuant+ stacked with TriAttention V3 eviction
VLLM_TRIATT_ENABLED=1 \
VLLM_TRIATT_BUDGET=29491 \
vllm serve <model> --kv-cache-dtype turboquant_k8v4
```

`turboquant_k8v4` is the recommended default: FP8 keys + 4-bit values, 2.6× compression, the safest accuracy/throughput trade-off across the validated model set. See the table below for tighter compression at the cost of more accuracy delta.

---

## TurboQuant+ presets

Set via `--kv-cache-dtype <preset>` on the vLLM CLI, or `kv_cache_dtype="<preset>"` in the `LLM(...)` constructor.

### Naming convention

```
turboquant_k<K>v<V>[_nc][_rv][_cv_rv]
                 │     │     │
                 │     │     └── _cv_rv = centroid-quantized rotated V (TQ+ extension)
                 │     └──────── _rv = rotate V before quantization (TQ+ extension)
                 └────────────── _nc = norm correction (recommended for MSE keys)

K = key bits (3, 4, 8=FP8)
V = value bits (2, 3, 4)
```

Shorthand: when K==V (and K != 8), the `k<K>v<V>` is collapsed to `<K>bit` (e.g., `turboquant_4bit_nc` = K=4, V=4, NC).

### Recommended presets

| Preset | K bits | V bits | Compression | Notes |
|---|---|---|---|---|
| `turboquant_k8v4` | FP8 | 4 | ~2.6× | **Default.** Best accuracy/throughput trade-off. Works on all GQA/MHA models with FP8-capable hardware (H100, MI300X, etc.) |
| `turboquant_k8v3` | FP8 | 3 | ~3.0× | Tighter V quantization. Validated on Qwen-class. |
| `turboquant_4bit_nc` | 4 (MSE) | 4 | ~3.8× | TurboQuant paper-canonical preset. Lower throughput than k8v4 but more compression. |
| `turboquant_k3v4_nc` | 3 (MSE) | 4 | ~3.5× | Asymmetric, prioritizes V fidelity. |
| `turboquant_3bit_nc` | 3 (MSE) | 3 | ~4.9× | Aggressive compression. Validate per-model first. |
| `turboquant_2bit_nc` | 2 (MSE) | 2 | ~5.6× | **Experimental.** TURBO2_0. Boundary-layer skip is mandatory. PPL must be validated per model before serving. |

### TQ+ rotated-V variants (`_rv`)

Apply the WHT rotation to V before uniform quantization. Spreads concentrated information across dimensions, improving uniform quantization quality at the same bit width. Costs one extra GEMM per layer on decode.

Available for every base preset above: `turboquant_k8v4_rv`, `turboquant_4bit_nc_rv`, `turboquant_3bit_nc_rv`, etc.

### TQ+ centroid-V variants (`_cv_rv`)

V uses Lloyd-Max centroid indices (same table as MSE keys) instead of uniform scale/zero. Saves bytes and matches uniform quality when V is post-WHT. Requires `_rv`.

Available for non-FP8 K presets only: `turboquant_4bit_nc_cv_rv`, `turboquant_3bit_nc_cv_rv`, `turboquant_k3v4_nc_cv_rv`, etc.

### Asymmetric K/V combos

Beyond the upstream four, this fork adds:

```
turboquant_k8v3        FP8 K + 3-bit V
turboquant_k4v3_nc     4-bit K + 3-bit V + NC
turboquant_k4v2_nc     4-bit K + 2-bit V + NC
turboquant_k3v2_nc     3-bit K + 2-bit V + NC
turboquant_k2v4_nc     2-bit K + 4-bit V + NC
```

Each carries `_rv` and (where K != 8) `_cv_rv` variants.

---

## TriAttention V3 (KV eviction)

V3 is **opt-in** and **per-process**. It scores tokens by alignment with calibrated query distribution, then evicts the lowest-scoring tokens to keep the cache under a budget. Stacks cleanly with any TurboQuant+ preset.

### Enable via env vars

Set before `LLM(...)` or `vllm serve`:

```bash
VLLM_TRIATT_ENABLED=1                    # master switch
VLLM_TRIATT_BUDGET=29491                 # max live cells per sequence (90% of 32K)
VLLM_TRIATT_PREFIX=128                   # never-evicted prefix length
VLLM_TRIATT_WINDOW=128                   # never-evicted recent-window length
VLLM_TRIATT_SEGMENTS=8                   # per-segment quota bucket count
VLLM_TRIATT_HYBRID=2                     # selection mode: 1=V1, 2=V3 (default)
VLLM_TRIATT_WARMUP=1024                  # Q samples before calibration fires
VLLM_TRIATT_ADAPTIVE=0                   # 1 = update calibration via EMA each round
```

### Enable via Python helper

```python
from vllm import LLM, SamplingParams
from vllm.v1.attention.triattention import (
    TriAttentionV3Config,
    install_triattention,
)

cfg = TriAttentionV3Config(
    budget=29491,           # tokens kept (90% of 32K)
    prefix_protect=128,
    window_size=128,
    hybrid_mode=2,          # V3
)
install_triattention("Qwen/Qwen3-8B", cfg)   # MUST be called before LLM()

llm = LLM(
    model="Qwen/Qwen3-8B",
    kv_cache_dtype="turboquant_k8v4",        # stack with TQ+
    max_model_len=32768,
)
```

### Validated envelope

V3 is validated on:

- Qwen2.5-7B-Instruct at 32K (paper-exact protocol)
- Qwen3-8B at 8K / 16K / 32K
- Qwen3-30B-A3B (MoE) at 32K
- Mistral-7B-Instruct-v0.3 at 32K
- Llama / Gemma 4 architectures (kernel-level)

Stack (V3 + `turboquant_k8v4`) hits **2.96× total KV memory reduction at 90% retention** with no measured PPL or NIAH cost on Qwen2.5-7B at 32K.

### Outside the validated envelope (don't auto-enable)

- **Hybrid Mamba+Attention models** (Qwen3-Next, Mamba2-primed-HQwen3, Jamba). The vLLM page-size unifier rejects TQ on hybrids; V3 alone is also unverified at NIAH middle/end on these.
- **Reasoning workloads, multi-needle retrieval, 128K+ context.** Untested.
- **Aggressive retention (<90%).** Paper saw partial NIAH at 85%; tighter is documented to break.

Full V3 docs: [vllm/v1/attention/triattention/README.md](vllm/v1/attention/triattention/README.md)
Full V3 paper: [TheTom/turboquant_plus/docs/papers/triattention-v3.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/triattention-v3.md)

---

## Performance knobs

These are TQ+ kernel optimizations, default-on. Override via env var.

```bash
# Sparse V tile-skip on GQA (skip V load+dequant for tiles below softmax threshold)
VLLM_TQ_SPARSE_V=auto                    # auto | 1 | 0 (default auto: on if seq_len ≥ ctx_threshold)
VLLM_TQ_SPARSE_V_THRESHOLD=0.1           # softmax probability cutoff
VLLM_TQ_SPARSE_V_CTX_THRESHOLD=8192      # context-length gate for auto mode

# Batched-Q grouped decode kernel (multiple Q heads share KV head in one CTA)
VLLM_TQ_GROUPED_DECODE=1                 # 1 = on (default), 0 = single-Q kernel
VLLM_TQ_SPARSE_V_COMPOSE_GROUPED=0       # experimental: compose sparse V with grouped path

# AMD ROCm specific (required on MI300X to avoid unrelated AITER FP4 crash)
VLLM_ROCM_USE_AITER_FP4BMM=0
```

---

## Hardware

Tested on:

- **AMD Instinct MI300X** (gfx942), ROCm 7.2 — primary dev platform for this fork
- **NVIDIA H100 / H200** — TQ+ kernels expected to work (upstream's CUDA path is preserved); not actively benchmarked here
- **NVIDIA A100 / RTX PRO 6000 / consumer (4090, 1080 Ti)** — community-tested in upstream TurboQuant; TQ+ extensions in this fork are not yet validated. **Looking for community testers.**

---

## Where to file issues

This fork: https://github.com/TheTom/vllm-turboquant/issues

For upstream vLLM bugs unrelated to TQ+ or V3, file at https://github.com/vllm-project/vllm/issues.
