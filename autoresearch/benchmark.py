#!/usr/bin/env python3
"""
IMMUTABLE GROUND TRUTH — do not modify.

Benchmark harness for OpenAI-compatible inference endpoints.
Measures TTFT, ITL, throughput, and E2E latency at three concurrency levels.

Usage:
  python benchmark.py --base-url http://192.168.1.200:8000 --model Qwen/Qwen3-Coder-Next-FP8
  python benchmark.py --base-url http://192.168.1.200:8000 --model Qwen/Qwen3-Coder-Next-FP8 --output /tmp/result.json
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

# ── Fixed benchmark parameters — DO NOT CHANGE ───────────────────────────────
CONCURRENCY_LEVELS = [1, 8, 32]
REQUESTS_PER_LEVEL = 50
MAX_OUTPUT_TOKENS   = 256
WARMUP_REQUESTS     = 3
REQUEST_TIMEOUT_S   = 180
MAX_RETRIES         = 2

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers concisely and accurately."
)
USER_PROMPT = (
    "Explain the key architectural difference between paged attention and "
    "radix attention for LLM inference serving, and describe when each is "
    "preferable in production systems."
)
# Combined prompt is ~120 tokens; with system = ~150 tokens total


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    ttft_ms: float          # time to first token
    itl_ms: float           # inter-token latency (avg across tokens)
    e2e_ms: float           # total request duration
    output_tokens: int
    error: Optional[str] = None


@dataclass
class LevelResult:
    concurrency: int
    throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p99_ms: float
    e2e_p50_ms: float
    requests_ok: int
    requests_err: int


# ── HTTP streaming request ────────────────────────────────────────────────────

async def single_request(session_url: str, model: str) -> RequestResult:
    import aiohttp

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first = None
    token_times = []
    output_tokens = 0

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            ) as session:
                async with session.post(
                    f"{session_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return RequestResult(0, 0, 0, 0, error=f"HTTP {resp.status}: {body[:200]}")

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
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
                    (token_times[i] - token_times[i-1]) * 1000
                    for i in range(1, len(token_times))
                ) if len(token_times) > 1 else 0.0
            )
            e2e_ms = (t_end - t_start) * 1000
            return RequestResult(ttft_ms, itl_ms, e2e_ms, output_tokens)

        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return RequestResult(0, 0, 0, 0, error="timeout")
        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return RequestResult(0, 0, 0, 0, error=str(e))

    return RequestResult(0, 0, 0, 0, error="max retries exceeded")


# ── Level runner ──────────────────────────────────────────────────────────────

async def run_level(base_url: str, model: str, concurrency: int, n: int, warmup: bool = False) -> list[RequestResult]:
    sem = asyncio.Semaphore(concurrency)
    label = "warmup" if warmup else f"c={concurrency}"

    async def bounded(i: int) -> RequestResult:
        async with sem:
            return await single_request(base_url, model)

    tasks = [bounded(i) for i in range(n)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    if not warmup:
        ok = [r for r in results if not r.error]
        total_tokens = sum(r.output_tokens for r in ok)
        print(f"  [{label}] {len(ok)}/{n} ok, {total_tokens} tokens in {elapsed:.1f}s", flush=True)

    return list(results)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(results: list[RequestResult], concurrency: int, elapsed_s: float) -> LevelResult:
    ok = [r for r in results if not r.error]
    if not ok:
        return LevelResult(concurrency, 0, 0, 0, 0, 0, 0, 0, len(results))

    total_tokens = sum(r.output_tokens for r in ok)
    throughput = total_tokens / elapsed_s if elapsed_s > 0 else 0

    def pct(vals, p):
        vals = sorted(vals)
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    ttfts = [r.ttft_ms for r in ok]
    itls  = [r.itl_ms  for r in ok if r.itl_ms > 0]
    e2es  = [r.e2e_ms  for r in ok]

    return LevelResult(
        concurrency=concurrency,
        throughput_tok_s=round(throughput, 1),
        ttft_p50_ms=round(pct(ttfts, 50), 1),
        ttft_p99_ms=round(pct(ttfts, 99), 1),
        itl_p50_ms=round(pct(itls, 50), 2) if itls else 0,
        itl_p99_ms=round(pct(itls, 99), 2) if itls else 0,
        e2e_p50_ms=round(pct(e2es, 50), 1),
        requests_ok=len(ok),
        requests_err=len(results) - len(ok),
    )


# ── Health check ──────────────────────────────────────────────────────────────

def wait_for_health(base_url: str, timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_benchmark(base_url: str, model: str) -> dict:
    print(f"[benchmark] {model} @ {base_url}", flush=True)

    print("[benchmark] Waiting for /health ...", flush=True)
    if not wait_for_health(base_url, timeout_s=300):
        sys.exit("ERROR: server not healthy after 5 minutes")
    print("[benchmark] Server healthy", flush=True)

    level_results = []

    for concurrency in CONCURRENCY_LEVELS:
        # Warmup
        print(f"[benchmark] Warmup c={concurrency} ...", flush=True)
        await run_level(base_url, model, concurrency, WARMUP_REQUESTS, warmup=True)

        # Benchmark
        print(f"[benchmark] Running c={concurrency} ({REQUESTS_PER_LEVEL} requests) ...", flush=True)
        t0 = time.perf_counter()
        results = await run_level(base_url, model, concurrency, REQUESTS_PER_LEVEL)
        elapsed = time.perf_counter() - t0
        level_results.append(aggregate(results, concurrency, elapsed))

    # Aggregate across all levels (use c=32 as primary)
    primary = next(r for r in level_results if r.concurrency == 32)

    output = {
        "model": model,
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": asdict(primary),
        "levels": [asdict(r) for r in level_results],
    }

    # Print summary
    print("\n=== BENCHMARK SUMMARY ===", flush=True)
    print(f"Model: {model}", flush=True)
    for r in level_results:
        print(
            f"  c={r.concurrency:2d}: {r.throughput_tok_s:7.1f} tok/s | "
            f"TTFT p50={r.ttft_p50_ms:.0f}ms p99={r.ttft_p99_ms:.0f}ms | "
            f"ITL p50={r.itl_p50_ms:.1f}ms p99={r.itl_p99_ms:.1f}ms | "
            f"ok={r.requests_ok}/{r.requests_ok + r.requests_err}",
            flush=True,
        )
    print(f"RESULT throughput={primary.throughput_tok_s} ttft_p50={primary.ttft_p50_ms} "
          f"ttft_p99={primary.ttft_p99_ms} itl_p50={primary.itl_p50_ms} itl_p99={primary.itl_p99_ms}",
          flush=True)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.1.200:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-Next-FP8")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args()

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        sys.exit("ERROR: aiohttp not installed. Run: pip install aiohttp")

    result = asyncio.run(run_benchmark(args.base_url, args.model))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[benchmark] Results written to {args.output}", flush=True)

    return result


if __name__ == "__main__":
    main()
