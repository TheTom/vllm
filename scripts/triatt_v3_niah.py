"""TriAttention V3 NIAH eval (strict checker).

Mirrors docs/papers/triattention-v3.md §3.3:
  - Build a wikitext-derived haystack at the target context size
  - Insert needle "The secret code word is PURPLE ELEPHANT 7742." at a fixed
    character position (start/middle/end)
  - Greedy decode (temp=0)
  - Strict case-sensitive match for "PURPLE ELEPHANT 7742" in the generated
    output (NOT in the prompt — `--no-display-prompt` analog)

Three position triples by context:
  32K: 400 / 65000 / 120000 (chars)
  64K: 800 / 130000 / 240000

Modes mirror triatt_v3_ppl.py:
  baseline / v3 / tq / stack

Usage:
  python3 triatt_v3_niah.py --model /mnt/scratch/models/Qwen2.5-7B-Instruct \
      --ctx 32768 --mode v3 --budget 29491 --prefix 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_ROCM_USE_AITER_FP4BMM", "0")

from vllm import LLM, SamplingParams
from vllm.v1.attention.triattention import (
    TriAttentionV3Config,
    install_triattention,
)


NEEDLE = "The secret code word is PURPLE ELEPHANT 7742."
QUESTION = (
    "What is the secret code word mentioned earlier? "
    "Answer with just the code word and number, nothing else:"
)
PASS_STRING = "PURPLE ELEPHANT 7742"

POSITIONS_32K = [400, 65000, 120000]
POSITIONS_64K = [800, 130000, 240000]

MODE_KV: dict[str, str] = {
    "baseline": "auto",
    "v3": "turboquant_k8v4",
    "tq": "turboquant_4bit_nc_cv_rv",
    "stack": "turboquant_k8v4",
}


def build_prompt(
    haystack: str, char_pos: int, tokenizer, target_tokens: int
) -> list[int]:
    """Build a token-id list of length <= target_tokens with the needle
    inserted at `char_pos` chars into the haystack and the question
    appended at the end.
    """
    before_text = haystack[:char_pos]
    after_text = haystack[char_pos:]
    text = f"{before_text}\n{NEEDLE}\n{after_text}\n\n{QUESTION}\n"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_tokens:
        return ids
    # Truncate to fit. Keep the needle (~early in context for "start" pos,
    # need to also keep question at the very end). Trim the haystack tail.
    # NOTE: for "end" position the needle is near the end of the haystack;
    # truncating from the tail removes the question. So we truncate from the
    # MIDDLE: keep the first half (which contains the start/middle needle)
    # and the last few hundred tokens (which contain the question and the
    # end-position needle if any).
    keep_tail = 256
    head = ids[: target_tokens - keep_tail]
    tail = ids[-keep_tail:]
    return head + tail


def classify(out_text: str) -> str:
    if PASS_STRING in out_text:
        return "PASS"
    if "PURPLE ELEPHANT" in out_text:
        return "PARTIAL_WORD"
    if "7742" in out_text:
        return "PARTIAL_NUMBER"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/scratch/models/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--mode", choices=list(MODE_KV.keys()), required=True
    )
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--budget", type=int, default=29491)
    ap.add_argument("--prefix", type=int, default=128)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64,
                    help="generation budget (use 1024 for reasoning models)")
    ap.add_argument("--gpu-mem", type=float, default=0.50)
    ap.add_argument("--haystack", default="/root/wikitext-2-raw/wiki.test.raw")
    args = ap.parse_args()

    haystack = open(args.haystack).read()

    kv = MODE_KV[args.mode]

    triatt_cfg: TriAttentionV3Config | None = None
    if args.mode in ("v3", "stack"):
        triatt_cfg = TriAttentionV3Config(
            budget=args.budget,
            prefix_protect=args.prefix,
            window_size=args.window,
            hybrid_mode=2,
        )

    if triatt_cfg is not None:
        install_triattention(triatt_cfg)

    print(
        f"# loading {kv} mode={args.mode} ctx={args.ctx}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=kv,
        # Headroom for the inserted needle + question (~30-50 tokens) on top
        # of the haystack-derived prompt (~ctx tokens) plus generation budget.
        max_model_len=args.ctx + args.gen + 512,
        gpu_memory_utilization=args.gpu_mem,
        disable_log_stats=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=512,
    )
    print(f"# loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    sp = SamplingParams(max_tokens=args.gen, temperature=0.0)
    tokenizer = llm.get_tokenizer()

    positions = POSITIONS_32K if args.ctx <= 32 * 1024 else POSITIONS_64K
    pos_names = ["start", "middle", "end"]

    print("mode,kv,ctx,position,char_pos,result")
    for name, char_pos in zip(pos_names, positions):
        prompt_ids = build_prompt(haystack, char_pos, tokenizer, args.ctx)
        out = llm.generate(
            {"prompt_token_ids": prompt_ids},
            sampling_params=sp,
            use_tqdm=False,
        )[0]
        text = out.outputs[0].text
        verdict = classify(text)
        print(
            f"{args.mode},{kv},{args.ctx},{name},{char_pos},{verdict}",
            flush=True,
        )


if __name__ == "__main__":
    main()
