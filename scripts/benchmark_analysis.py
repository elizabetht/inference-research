#!/usr/bin/env python3
"""
benchmark_analysis.py — Reads the latest curation and generates a concrete
benchmark plan for the 2-node DGX Spark cluster.

DGX Spark spec:
  - Each node: NVIDIA GB10 Grace Blackwell Superchip
  - 128 GB unified memory (GPU+CPU via NVLink-C2C)
  - 2 nodes: spark-01 (192.168.1.76), spark-02 (192.168.1.77)
  - Controller: 192.168.1.75

Outputs: benchmarks/YYYY-MM-DD-plan.md
"""

import os
import sys
import json
import datetime
import textwrap
import urllib.request
from pathlib import Path


ANTHROPIC_MODEL = "claude-opus-4-6"

CLUSTER_SPEC = """
2-node DGX Spark cluster:
- spark-01: 192.168.1.76, NVIDIA GB10 Grace Blackwell Superchip, 128GB unified memory (NVLink-C2C), Ubuntu 24.04
- spark-02: 192.168.1.77, NVIDIA GB10 Grace Blackwell Superchip, 128GB unified memory (NVLink-C2C), Ubuntu 24.04
- Controller: 192.168.1.75 (CPU only, orchestrates workers)
- Inter-node: standard 10GbE (assume no NVLink between nodes unless otherwise configured)
- Each node has full Blackwell GPU architecture support
- Total: 256GB GPU memory across 2 nodes for tensor/pipeline parallelism experiments
"""


def claude_analyze(prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["content"][0]["text"]


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    today = datetime.date.today()
    repo_root = Path(__file__).parent.parent

    # Find today's or most recent curation
    curations_dir = repo_root / "curations"
    curation_file = curations_dir / f"{today}.md"
    if not curation_file.exists():
        # Find most recent
        md_files = sorted(curations_dir.glob("*.md"), reverse=True)
        if not md_files:
            sys.exit("No curation files found. Run research.py first.")
        curation_file = md_files[0]
        print(f"  Using most recent curation: {curation_file.name}")

    curation_text = curation_file.read_text()

    prompt = textwrap.dedent(f"""
    You are a senior MLSys engineer responsible for benchmarking LLM inference improvements.

    You have access to the following cluster:
    {CLUSTER_SPEC}

    Below is today's inference-radar curation report. Your job is to produce a concrete, runnable benchmark plan.

    FORMAT:
    ## Benchmark Plan — {today}

    ### Selected Experiments (rank by expected ROI)
    For each experiment (2-5 total):

    #### [N]. [Experiment Name] — [Framework]
    **Hypothesis**: what improvement we expect and why
    **Setup**:
      - Model: (e.g., Llama-3-8B, Llama-3-70B, Mistral-7B — pick what fits in 128GB/node)
      - Framework version: (specific tag/commit if known)
      - Node config: single-node or both nodes (TP=2 across nodes)
      - Parallelism: TP/PP dimensions
    **Benchmark script** (runnable bash, use venv at ~/inference-bench-env or docker):
    ```bash
    # ... actual commands
    ```
    **Metrics to capture**: TTFT (ms), TPOT (ms/tok), throughput (tok/s), GPU memory (GB), MFU (%)
    **Baseline**: what to compare against (e.g., vLLM main, sglang v0.4.2)
    **Expected delta**: quantified guess based on what the PR/paper claims
    **Pass/fail criteria**: specific number that would confirm the improvement is real

    ### Quick Wins (can run in <30 min total)
    List 1-2 single-command checks that give signal fast.

    ### Infrastructure Notes
    Any cluster setup required (e.g., passwordless SSH between spark-01/spark-02, NCCL env vars, HuggingFace cache mounts).

    ---
    CURATION REPORT:
    {curation_text[:6000]}
    """).strip()

    print(f"[benchmark-analysis] Generating benchmark plan for {today}...")
    plan = claude_analyze(prompt, api_key)

    out_dir = repo_root / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{today}-plan.md"
    out_path.write_text(f"---\ndate: {today}\nsource_curation: curations/{curation_file.name}\n---\n\n{plan}")
    print(f"  ✓ Saved → benchmarks/{today}-plan.md")
    return str(out_path)


if __name__ == "__main__":
    main()
