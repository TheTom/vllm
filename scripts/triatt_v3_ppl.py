"""TriAttention V3 PPL eval on wikitext-2-raw.

Mirrors the protocol in docs/papers/triattention-v3.md §3.2:
  - 3 chunks minimum (single-chunk PPL is too noisy at 32K)
  - Force chunked prefill with max_num_batched_tokens=512 so the V3
    eviction hook actually runs during the sweep (not once at the end)
  - Eviction confirmation: the engine reports n_evict_rounds in stats

Modes (mutually exclusive):
  --mode baseline    : kv_cache_dtype=auto, V3 disabled
  --mode v3          : kv_cache_dtype=turboquant_k8v4 (FP8 K + 4-bit V), V3 on
  --mode tq          : kv_cache_dtype=turboquant_4bit_nc_cv_rv, V3 disabled
  --mode stack       : kv_cache_dtype=turboquant_k8v4, V3 on (full stack)

The V3 modes use turboquant_k8v4 so K stays FP8 (dequant-able for V3 scoring),
matching the K=q8_0 + V=turbo3 config Tom validated on llama.cpp.

Usage:
  python3 triatt_v3_ppl.py --model /mnt/scratch/models/Qwen2.5-7B-Instruct \
      --ctx 32768 --chunks 3 --mode v3 --budget 29491 --prefix 128
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("VLLM_ROCM_USE_AITER_FP4BMM", "0")

from vllm import LLM, SamplingParams
from vllm.v1.attention.triattention import (
    TriAttentionV3Config,
    install_triattention,
)


MODE_KV: dict[str, str] = {
    "baseline": "auto",
    "v3": "turboquant_k8v4",
    "tq": "turboquant_4bit_nc_cv_rv",
    "stack": "turboquant_k8v4",
    # Custom modes for the TQ-variant ablation: --kv overrides the dtype.
    "tq_variant": "OVERRIDE",
    "stack_variant": "OVERRIDE",
}

V3_MODES = {"v3", "stack", "stack_variant"}


def load_chunks(path: str, ctx: int, tokenizer):
    text = open(path).read()
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids) - ctx, ctx):
        chunks.append(ids[i : i + ctx])
    return chunks


def measure_ppl(
    model_path: str,
    kv_dtype: str,
    ctx: int,
    n_chunks: int,
    triatt_cfg: TriAttentionV3Config | None,
    gpu_mem: float,
) -> dict:
    if triatt_cfg is not None:
        install_triattention(model_path, triatt_cfg)
    print(f"# loading {kv_dtype}", file=sys.stderr, flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        kv_cache_dtype=kv_dtype,
        max_model_len=ctx + 16,
        gpu_memory_utilization=gpu_mem,
        disable_log_stats=True,
        enable_prefix_caching=False,
        # Force chunked prefill so the V3 eviction hook (per-step) actually
        # fires during the sweep. See docs §3.2.
        max_num_batched_tokens=512,
    )
    print(f"# loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    tokenizer = llm.get_tokenizer()
    chunks = load_chunks("/root/wikitext-2-raw/wiki.test.raw", ctx, tokenizer)
    if n_chunks > 0:
        chunks = chunks[:n_chunks]
    print(
        f"# {len(chunks)} chunks of {ctx} tokens each",
        file=sys.stderr,
        flush=True,
    )

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5)
    total_logprob = 0.0
    total_tokens = 0
    t1 = time.time()
    for i, ids in enumerate(chunks):
        out = llm.generate(
            {"prompt_token_ids": ids},
            sampling_params=sp,
            use_tqdm=False,
        )[0]
        for pos, tok_lp in enumerate(out.prompt_logprobs):
            if tok_lp is None or pos == 0:
                continue
            entry = tok_lp.get(ids[pos])
            if entry is None:
                continue
            total_logprob += float(entry.logprob)
            total_tokens += 1
        ppl_so_far = math.exp(-total_logprob / max(total_tokens, 1))
        print(
            f"#   chunk {i+1}/{len(chunks)} ppl_so_far={ppl_so_far:.4f}",
            file=sys.stderr,
            flush=True,
        )

    elapsed = time.time() - t1
    avg_lp = total_logprob / total_tokens
    ppl = math.exp(-avg_lp)
    return {
        "kv": kv_dtype,
        "ctx": ctx,
        "ppl": ppl,
        "tokens": total_tokens,
        "seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/scratch/models/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--mode",
        choices=list(MODE_KV.keys()),
        required=True,
        help="baseline / v3 / tq / stack / tq_variant / stack_variant",
    )
    ap.add_argument(
        "--kv",
        default=None,
        help="kv_cache_dtype override (required for *_variant modes)",
    )
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--budget", type=int, default=29491)
    ap.add_argument("--prefix", type=int, default=128)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.50)
    ap.add_argument("--out", default="-", help="output csv path or '-' for stdout")
    args = ap.parse_args()

    kv = MODE_KV[args.mode]
    if kv == "OVERRIDE":
        if not args.kv:
            ap.error("--kv is required when --mode is *_variant")
        kv = args.kv
    triatt_cfg: TriAttentionV3Config | None = None
    if args.mode in V3_MODES:
        triatt_cfg = TriAttentionV3Config(
            budget=args.budget,
            prefix_protect=args.prefix,
            window_size=args.window,
            n_segments=args.segments,
            warmup_tokens=args.warmup,
            hybrid_mode=2,
        )
    print(
        "mode,kv,ctx,chunks,budget,prefix,window,ppl,tokens,seconds",
        flush=True,
    )
    result = measure_ppl(
        model_path=args.model,
        kv_dtype=kv,
        ctx=args.ctx,
        n_chunks=args.chunks,
        triatt_cfg=triatt_cfg,
        gpu_mem=args.gpu_mem,
    )
    line = (
        f"{args.mode},{kv},{args.ctx},{args.chunks},{args.budget},"
        f"{args.prefix},{args.window},{result['ppl']:.4f},{result['tokens']},"
        f"{result['seconds']:.1f}"
    )
    print(line, flush=True)
    if args.out != "-":
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
