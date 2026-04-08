#!/usr/bin/env python3
"""
benchmark_isl_osl.py — ISL/OSL sweep benchmark for vLLM-compatible endpoints.

Tests a matrix of input/output sequence lengths across concurrency levels.
Measures TTFT, ITL, and throughput for each (ISL, OSL, concurrency) combination.
Results are written as JSON for downstream analysis and plotting.

Usage:
  python benchmark_isl_osl.py --base-url http://192.168.1.202:8000 \
      --model google/gemma-4-26B-A4B \
      --output results/gemma4-26b-isl-osl-2026-04-07.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

# ── Sweep parameters ──────────────────────────────────────────────────────────

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]
REQUESTS_PER_CELL  = 40   # per (ISL, OSL, concurrency) cell
WARMUP_REQUESTS    = 3
REQUEST_TIMEOUT_S  = 300

# ISL/OSL combinations — (label, isl_tokens, osl_tokens)
ISL_OSL_COMBOS = [
    ("short-short",  128,  128),
    ("short-long",   128,  512),
    ("long-short",  1024,  128),
    ("long-long",   1024,  512),
    ("very-long",   4096,  512),
]

# Approximate tokens per word for synthetic prompt generation
_TOKENS_PER_WORD = 1.3

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    ttft_ms: float
    itl_ms: float           # avg ITL across all output tokens
    e2e_ms: float
    output_tokens: int
    error: Optional[str] = None


@dataclass
class CellResult:
    label: str
    isl: int
    osl: int
    concurrency: int
    throughput_tok_s: float   # output tokens / total wall time
    ttft_p50_ms: float
    ttft_p95_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    requests_ok: int
    requests_err: int


# ── Synthetic prompt generation ───────────────────────────────────────────────

# Word bank for constructing prompts of approximate token length
_WORD_BANK = (
    "the inference engine processes tokens through attention layers computing "
    "key value cache entries for each position in the sequence the pipeline "
    "parallel configuration splits transformer blocks across multiple nodes "
    "connected via high speed interconnect enabling larger model capacity than "
    "fits on a single accelerator memory prefill phase processes input tokens "
    "in parallel while decode phase generates one token per forward pass the "
    "speculative decoding technique uses a draft model to propose multiple "
    "candidate tokens which are then verified by the target model in a single "
    "forward pass improving throughput at the cost of additional memory for "
    "the draft model weights quantization reduces memory footprint and improves "
    "throughput by representing weights and activations in lower precision "
    "formats such as int8 or fp8 while maintaining acceptable quality "
    "mixture of experts architectures route tokens to specialized sub-networks "
    "activating only a fraction of total parameters per forward pass which "
    "enables scaling to very large parameter counts without proportional "
    "increases in compute requirements for inference workloads "
).split()


def build_prompt(target_tokens: int) -> str:
    """Build a synthetic prompt of approximately target_tokens tokens."""
    target_words = int(target_tokens / _TOKENS_PER_WORD)
    words = []
    while len(words) < target_words:
        words.extend(_WORD_BANK)
    return " ".join(words[:target_words])


# Pre-build prompts for each ISL to avoid repeated construction
_PROMPTS: dict[int, str] = {}


def get_prompt(isl: int) -> str:
    if isl not in _PROMPTS:
        _PROMPTS[isl] = build_prompt(isl)
    return _PROMPTS[isl]


# ── Single async request ──────────────────────────────────────────────────────

async def single_request(
    session,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    t_first = None
    token_times: list[float] = []
    output_tokens = 0

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(0, 0, 0, 0, error=f"HTTP {resp.status}: {body[:200]}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        t_now = time.perf_counter()
                        if t_first is None:
                            t_first = t_now
                        else:
                            token_times.append(t_now)
                        output_tokens += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        t_end = time.perf_counter()

        if t_first is None:
            return RequestResult(0, 0, 0, 0, error="no tokens received")

        ttft_ms = (t_first - t_start) * 1000
        itl_ms = (
            statistics.mean(
                (token_times[i] - token_times[i - 1]) * 1000
                for i in range(1, len(token_times))
            )
            if len(token_times) > 1
            else 0.0
        )
        e2e_ms = (t_end - t_start) * 1000
        return RequestResult(ttft_ms, itl_ms, e2e_ms, output_tokens)

    except asyncio.TimeoutError:
        return RequestResult(0, 0, 0, 0, error="timeout")
    except Exception as exc:
        return RequestResult(0, 0, 0, 0, error=str(exc))


# ── Cell runner ───────────────────────────────────────────────────────────────

async def run_cell(
    base_url: str,
    model: str,
    isl: int,
    osl: int,
    concurrency: int,
    n: int,
    warmup: bool = False,
) -> tuple[list[RequestResult], float]:
    import aiohttp

    prompt = get_prompt(isl)
    sem = asyncio.Semaphore(concurrency)

    async def bounded(_: int) -> RequestResult:
        async with sem:
            return await single_request(session, base_url, model, prompt, osl)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded(i) for i in range(n)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    if not warmup:
        ok = [r for r in results if not r.error]
        errs = [r for r in results if r.error]
        total_out = sum(r.output_tokens for r in ok)
        label_str = f"isl={isl} osl={osl} c={concurrency}"
        print(
            f"  [{label_str}] {len(ok)}/{n} ok, {total_out} out-tokens "
            f"in {elapsed:.1f}s, {total_out / elapsed:.0f} tok/s",
            flush=True,
        )
        if errs:
            print(f"    errors: {[r.error for r in errs[:3]]}", flush=True)

    return list(results), elapsed


# ── Percentile helper ─────────────────────────────────────────────────────────

def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(
    results: list[RequestResult],
    label: str,
    isl: int,
    osl: int,
    concurrency: int,
    elapsed_s: float,
) -> CellResult:
    ok = [r for r in results if not r.error]
    if not ok:
        return CellResult(label, isl, osl, concurrency, 0, 0, 0, 0, 0, 0, 0, 0, len(results))

    total_tokens = sum(r.output_tokens for r in ok)
    throughput = total_tokens / elapsed_s if elapsed_s > 0 else 0

    ttfts = [r.ttft_ms for r in ok]
    itls  = [r.itl_ms  for r in ok if r.itl_ms > 0]
    e2es  = [r.e2e_ms  for r in ok]

    return CellResult(
        label=label,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        throughput_tok_s=round(throughput, 1),
        ttft_p50_ms=round(pct(ttfts, 50), 1),
        ttft_p95_ms=round(pct(ttfts, 95), 1),
        itl_p50_ms=round(pct(itls, 50), 2) if itls else 0.0,
        itl_p95_ms=round(pct(itls, 95), 2) if itls else 0.0,
        e2e_p50_ms=round(pct(e2es, 50), 1),
        e2e_p95_ms=round(pct(e2es, 95), 1),
        requests_ok=len(ok),
        requests_err=len(results) - len(ok),
    )


# ── Health check ──────────────────────────────────────────────────────────────

def wait_for_health(base_url: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    print(f"[benchmark] Waiting for {base_url}/health ...", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


# ── Main benchmark loop ───────────────────────────────────────────────────────

async def run_benchmark(base_url: str, model: str, output_path: Optional[str]) -> dict:
    print(f"[benchmark] ISL/OSL sweep: {model} @ {base_url}", flush=True)
    print(f"[benchmark] Combos: {[c[0] for c in ISL_OSL_COMBOS]}", flush=True)
    print(f"[benchmark] Concurrency levels: {CONCURRENCY_LEVELS}", flush=True)

    if not wait_for_health(base_url):
        sys.exit("ERROR: server not healthy after 10 minutes")
    print("[benchmark] Server healthy\n", flush=True)

    all_results: list[CellResult] = []

    for label, isl, osl in ISL_OSL_COMBOS:
        print(f"\n=== Combo: {label} (ISL={isl}, OSL={osl}) ===", flush=True)

        for concurrency in CONCURRENCY_LEVELS:
            # Warmup
            print(f"  Warmup c={concurrency} ...", flush=True)
            await run_cell(base_url, model, isl, osl, concurrency, WARMUP_REQUESTS, warmup=True)

            # Timed run
            results, elapsed = await run_cell(
                base_url, model, isl, osl, concurrency, REQUESTS_PER_CELL
            )
            cell = aggregate(results, label, isl, osl, concurrency, elapsed)
            all_results.append(cell)

    # Print full summary table
    print("\n\n=== ISL/OSL SWEEP SUMMARY ===", flush=True)
    header = (
        f"{'combo':<14} {'c':>4} {'tput':>8} "
        f"{'TTFT_p50':>10} {'TTFT_p95':>10} "
        f"{'ITL_p50':>9} {'ITL_p95':>9} "
        f"{'E2E_p50':>9} {'ok/n':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in all_results:
        print(
            f"{r.label:<14} {r.concurrency:>4} {r.throughput_tok_s:>7.1f}t "
            f"{r.ttft_p50_ms:>9.0f}ms {r.ttft_p95_ms:>9.0f}ms "
            f"{r.itl_p50_ms:>8.1f}ms {r.itl_p95_ms:>8.1f}ms "
            f"{r.e2e_p50_ms:>8.0f}ms "
            f"{r.requests_ok:>3}/{r.requests_ok + r.requests_err}",
            flush=True,
        )

    output = {
        "model": model,
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concurrency_levels": CONCURRENCY_LEVELS,
        "isl_osl_combos": [{"label": l, "isl": i, "osl": o} for l, i, o in ISL_OSL_COMBOS],
        "results": [asdict(r) for r in all_results],
    }

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[benchmark] Results written to {output_path}", flush=True)

    return output


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ISL/OSL sweep benchmark for vLLM-compatible endpoints"
    )
    parser.add_argument(
        "--base-url",
        default="http://192.168.1.202:8000",
        help="Base URL of the inference endpoint",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-26B-A4B",
        help="Model name to pass to the API",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: stdout only)",
    )
    args = parser.parse_args()

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        sys.exit("ERROR: aiohttp not installed. Run: pip install aiohttp")

    asyncio.run(run_benchmark(args.base_url, args.model, args.output))


if __name__ == "__main__":
    main()
