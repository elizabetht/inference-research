#!/usr/bin/env python3
"""
benchmark_isl_osl.py — ISL/OSL sweep against the llm-d endpoint (OpenAI-compatible API).

Target: llm-d serving google/gemma-4-26B-A4B via PD disaggregation
Endpoint: http://192.168.1.202/v1 (LoadBalancer on token-labs-gateway)

Benchmark design:
  - Sweep over ISL x OSL combinations from ISL_BUCKETS x OSL_BUCKETS
  - For each (ISL, OSL) pair: send CONCURRENCY requests in parallel, repeat ROUNDS times
  - Measure TTFT, ITL (inter-token latency), E2E latency, throughput
  - Output: JSON results + CSV summary to results/gemma4-26b-isl-osl-YYYYMMDD.{json,csv}

Usage:
  python3 benchmark_isl_osl.py [--endpoint URL] [--model MODEL] [--concurrency N]
                                [--rounds N] [--isl-buckets A,B,C] [--osl-buckets X,Y,Z]
                                [--output-dir PATH]

Requirements:
  pip install openai tiktoken numpy tqdm
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")  # close enough for token counting
except ImportError:
    _TOKENIZER = None
    print("tiktoken not found — using whitespace split for token estimates", file=sys.stderr)

try:
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    atqdm = None


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_ENDPOINT   = os.environ.get("LLMD_ENDPOINT", "http://192.168.1.202")
DEFAULT_MODEL      = os.environ.get("LLMD_MODEL",    "google/gemma-4-26B-A4B")
DEFAULT_CONCURRENCY = 4
DEFAULT_ROUNDS      = 3

# ISL buckets: number of input tokens (synthetic prompt padded to length)
DEFAULT_ISL_BUCKETS = [128, 512, 1024, 2048, 4096]
# OSL buckets: max_tokens for completion
DEFAULT_OSL_BUCKETS = [64, 128, 256, 512, 1024]


# ── token helpers ──────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text))
    return len(text.split())


def build_prompt(target_isl: int) -> str:
    """Build a prompt of approximately target_isl tokens."""
    base = (
        "You are a helpful assistant. "
        "Please read the following passage and then answer all questions thoroughly. "
    )
    filler_word = "context "
    tokens_so_far = count_tokens(base)
    needed = max(0, target_isl - tokens_so_far - 20)
    filler = (filler_word * (needed // len(filler_word.split()) + 1)).strip()
    prompt = base + filler + " Now, please summarize everything above."
    return prompt


# ── single request ─────────────────────────────────────────────────────────────

async def run_single(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Fire one streaming chat completion and collect timing metrics."""
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    output_tokens = 0
    error = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
        )
        async for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if ttft is None:
                    ttft = now - t0
                    token_times.append(now)
                else:
                    token_times.append(now)
                output_tokens += 1
    except Exception as exc:
        error = str(exc)

    t1 = time.perf_counter()
    e2e = t1 - t0

    # inter-token latencies
    itls = []
    if len(token_times) > 1:
        itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    return {
        "ttft_ms":      round(ttft * 1000, 2) if ttft is not None else None,
        "e2e_ms":       round(e2e * 1000, 2),
        "output_tokens": output_tokens,
        "itl_mean_ms":  round(np.mean(itls) * 1000, 2) if itls else None,
        "itl_p99_ms":   round(np.percentile(itls, 99) * 1000, 2) if itls else None,
        "error":        error,
    }


# ── batch runner ───────────────────────────────────────────────────────────────

async def run_batch(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> list[dict]:
    tasks = [run_single(client, model, prompt, max_tokens) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── aggregate stats ─────────────────────────────────────────────────────────────

def aggregate(samples: list[dict]) -> dict:
    """Compute p50/p99 across a list of single-request result dicts."""
    good = [s for s in samples if s["error"] is None]
    errors = len(samples) - len(good)

    def pct(key, p):
        vals = [s[key] for s in good if s[key] is not None]
        return round(float(np.percentile(vals, p)), 2) if vals else None

    output_toks = [s["output_tokens"] for s in good]
    e2e_s = [s["e2e_ms"] / 1000 for s in good]
    throughput = sum(output_toks) / sum(e2e_s) if e2e_s else 0

    return {
        "n_requests":       len(samples),
        "n_errors":         errors,
        "ttft_p50_ms":      pct("ttft_ms", 50),
        "ttft_p99_ms":      pct("ttft_ms", 99),
        "e2e_p50_ms":       pct("e2e_ms", 50),
        "e2e_p99_ms":       pct("e2e_ms", 99),
        "itl_p50_ms":       pct("itl_mean_ms", 50),
        "itl_p99_ms":       pct("itl_p99_ms", 99),
        "output_tok_p50":   pct("output_tokens", 50),
        "throughput_tok_s": round(throughput, 1),
    }


# ── main sweep ─────────────────────────────────────────────────────────────────

async def sweep(args):
    client = AsyncOpenAI(
        base_url=f"{args.endpoint.rstrip('/')}/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "none"),
        timeout=600.0,
    )

    results = []
    isl_buckets = [int(x) for x in args.isl_buckets.split(",")]
    osl_buckets = [int(x) for x in args.osl_buckets.split(",")]
    total = len(isl_buckets) * len(osl_buckets)
    done = 0

    print(f"Sweep: {len(isl_buckets)} ISL x {len(osl_buckets)} OSL = {total} combos, "
          f"concurrency={args.concurrency}, rounds={args.rounds}")
    print(f"Endpoint: {args.endpoint}  Model: {args.model}")

    for isl in isl_buckets:
        prompt = build_prompt(isl)
        actual_isl = count_tokens(prompt)
        for osl in osl_buckets:
            all_samples = []
            for r in range(args.rounds):
                batch = await run_batch(client, args.model, prompt, osl, args.concurrency)
                all_samples.extend(batch)
            stats = aggregate(all_samples)
            row = {
                "model":       args.model,
                "isl_target":  isl,
                "isl_actual":  actual_isl,
                "osl_target":  osl,
                **stats,
            }
            results.append(row)
            done += 1
            errors = stats["n_errors"]
            print(
                f"  [{done:3d}/{total}] ISL={isl:5d} OSL={osl:5d} | "
                f"TTFT_p50={stats['ttft_p50_ms']}ms  "
                f"E2E_p50={stats['e2e_p50_ms']}ms  "
                f"ITL_p50={stats['itl_p50_ms']}ms  "
                f"tput={stats['throughput_tok_s']}tok/s"
                + (f"  [ERR={errors}]" if errors else "")
            )

    return results


# ── output ─────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], output_dir: Path, model: str):
    today = datetime.now().strftime("%Y%m%d")
    slug = model.replace("/", "-").replace("_", "-").lower()
    stem = output_dir / f"{slug}-isl-osl-{today}"

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = stem.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    if results:
        csv_path = stem.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV:  {csv_path}")

    return json_path


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint",    default=DEFAULT_ENDPOINT,
                   help=f"Base URL of llm-d (default: {DEFAULT_ENDPOINT})")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help=f"Model name (default: {DEFAULT_MODEL})")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"Parallel requests per batch (default: {DEFAULT_CONCURRENCY})")
    p.add_argument("--rounds",      type=int, default=DEFAULT_ROUNDS,
                   help=f"Number of rounds per (ISL, OSL) combo (default: {DEFAULT_ROUNDS})")
    p.add_argument("--isl-buckets", default=",".join(str(x) for x in DEFAULT_ISL_BUCKETS),
                   help="Comma-separated ISL values")
    p.add_argument("--osl-buckets", default=",".join(str(x) for x in DEFAULT_OSL_BUCKETS),
                   help="Comma-separated OSL (max_tokens) values")
    p.add_argument("--output-dir",  default=str(Path(__file__).parent.parent / "results"),
                   help="Directory to write JSON/CSV output")
    return p.parse_args()


def main():
    args = parse_args()
    results = asyncio.run(sweep(args))
    save_results(results, Path(args.output_dir), args.model)
    # surface any error rate summary
    total_err = sum(r["n_errors"] for r in results)
    total_req = sum(r["n_requests"] for r in results)
    if total_err:
        print(f"\nWARNING: {total_err}/{total_req} requests failed — check endpoint/model.")
    else:
        print(f"\nAll {total_req} requests succeeded.")


if __name__ == "__main__":
    main()
