#!/usr/bin/env python3
"""
run_experiments.py — Reads today's benchmark plan, generates a self-contained
run script via Claude, executes it on the cluster, parses results, and saves
JSON + markdown summaries.

Flow:
  1. Read benchmarks/YYYY-MM-DD-plan.md
  2. Claude → benchmarks/YYYY-MM-DD-run.sh  (self-contained, executable)
  3. Execute the run script (with 4h wall-clock timeout)
  4. Claude → benchmarks/YYYY-MM-DD-results.json + results.md
  5. Caller (run_nightly.sh) commits everything
"""

import os
import sys
import json
import subprocess
import datetime
import textwrap
import urllib.request
from pathlib import Path

# Local LLM endpoint (SGLang on spark-01, tunnelled to localhost:30001 by run_nightly.sh)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.1.200:8000")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "Qwen/Qwen3-Coder-Next-FP8")

EXPERIMENT_TIMEOUT_SEC = 4 * 3600  # 4 hours max for all experiments

CLUSTER = {
    "controller": "192.168.1.75",
    "spark_01":   "192.168.1.76",
    "spark_02":   "192.168.1.77",
    "ssh_user":   "nvidia",
    "hf_cache":   "/data/hf_cache",
}


# ── Claude helper ─────────────────────────────────────────────────────────────

def llm(prompt: str, max_tokens: int = 4096) -> str:
    payload = json.dumps({
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# ── Step 1: generate run script ───────────────────────────────────────────────

GENERATE_SCRIPT_PROMPT = """\
You are an MLSys engineer. Below is today's inference benchmark plan for a 2-node DGX Spark cluster.

Cluster:
- spark-01: nvidia@192.168.1.76 — NVIDIA GB10 Grace Blackwell, 128GB unified memory
- spark-02: nvidia@192.168.1.77 — NVIDIA GB10 Grace Blackwell, 128GB unified memory
- controller: 192.168.1.75 (runs this script)
- SSH: passwordless key-based auth, user=nvidia
- HF model cache: /data/hf_cache on each node (pre-populated)
- Python: python3 available on each node; use venvs under ~/bench/

Your task: produce a SINGLE self-contained bash script that runs ALL prioritized experiments from the plan.

REQUIREMENTS:
1. The script must be safe to run unattended (no interactive prompts)
2. Each experiment is wrapped in a function; failures are caught and logged, not fatal
3. Results are written as JSON to stdout as the LAST line: a JSON array of experiment result objects
4. Each result object has: name, framework, version, model, node, status (pass|fail|skip),
   metrics (dict with any of: ttft_p50_ms, ttft_p99_ms, tpot_p50_ms, tpot_p99_ms,
   throughput_tok_s, gpu_memory_gb, mfu_pct), notes (list of strings), raw_output (string, truncated)
5. Use set -euo pipefail inside each SSH heredoc but wrap the whole function call in a subshell
6. Install framework versions in fresh venvs named after the version (e.g. ~/bench/sglang-0.5.9/)
   and reuse them if they already exist (check with [ -d ~/bench/... ])
7. HF_HOME=/data/hf_cache on remote nodes
8. Add a 60-second sleep after starting any server before running benchmark
9. Always kill servers (pkill -f <pattern>) in a trap or finally block
10. Write progress to stderr; only the final JSON array goes to stdout

Output ONLY the bash script (no markdown fences, no explanation).

---
BENCHMARK PLAN:
{plan}
"""


def generate_run_script(plan_text: str, out_path: Path) -> Path:
    print(f"[run_experiments] Generating run script via {LLM_MODEL}...")
    prompt = GENERATE_SCRIPT_PROMPT.format(plan=plan_text[:8000])
    script = llm(prompt, max_tokens=6000)

    # Strip markdown fences if Claude added them anyway
    lines = script.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    script = "\n".join(lines)

    out_path.write_text(script)
    out_path.chmod(0o755)
    print(f"  ✓ Run script → {out_path.name}")
    return out_path


# ── Step 2: execute ───────────────────────────────────────────────────────────

def execute_experiments(script_path: Path, log_path: Path) -> tuple[list[dict], str]:
    """Run the script, capture output. Returns (results_list, raw_output)."""
    print(f"[run_experiments] Executing {script_path.name} (timeout {EXPERIMENT_TIMEOUT_SEC//3600}h)...")

    env = os.environ.copy()
    env["HF_HOME"] = "/data/hf_cache"

    with open(log_path, "a") as logf:
        logf.write(f"\n[{datetime.datetime.utcnow():%H:%M:%S}] === run_experiments start ===\n")
        proc = subprocess.run(
            ["/bin/bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=EXPERIMENT_TIMEOUT_SEC,
            env=env,
        )
        logf.write(proc.stderr)
        logf.write(f"\n[{datetime.datetime.utcnow():%H:%M:%S}] exit={proc.returncode}\n")

    raw = proc.stdout.strip()
    stderr_tail = proc.stderr[-3000:] if proc.stderr else ""

    # Last line should be JSON array
    results = []
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                results = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if not results and raw:
        print("  ⚠ Could not parse JSON results from script output — will parse via Claude")

    return results, raw + "\nSTDERR:\n" + stderr_tail


# ── Step 3: parse results if script didn't emit clean JSON ───────────────────

PARSE_RESULTS_PROMPT = """\
You are parsing the output of an LLM inference benchmark run.

Produce a JSON array of experiment result objects. Each object must have:
  name (string), framework (string), version (string), model (string), node (string),
  status ("pass"|"fail"|"skip"), metrics (object — include any of: ttft_p50_ms, ttft_p99_ms,
  tpot_p50_ms, tpot_p99_ms, throughput_tok_s, gpu_memory_gb, mfu_pct),
  notes (array of strings), raw_output (string, max 300 chars)

Output ONLY valid JSON — no explanation, no fences.

RAW OUTPUT:
{raw}
"""


def parse_results_via_llm(raw_output: str) -> list[dict]:
    print(f"[run_experiments] Parsing results via {LLM_MODEL}...")
    prompt = PARSE_RESULTS_PROMPT.format(raw=raw_output[:6000])
    text = llm(prompt, max_tokens=2048)
    # Strip fences
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    try:
        return json.loads("\n".join(lines))
    except json.JSONDecodeError as e:
        print(f"  ⚠ Claude returned unparseable JSON: {e}")
        return [{"name": "parse_error", "status": "fail", "notes": [str(e)], "metrics": {}}]


# ── Step 4: results markdown ──────────────────────────────────────────────────

def results_to_markdown(results: list[dict], today: datetime.date) -> str:
    lines = [f"# Benchmark Results — {today}\n"]
    for r in results:
        status_icon = {"pass": "✅", "fail": "❌", "skip": "⏭"}.get(r.get("status", ""), "❓")
        lines.append(f"## {status_icon} {r.get('name', 'Unknown')} — {r.get('framework', '')} {r.get('version', '')}")
        lines.append(f"**Model**: `{r.get('model', 'N/A')}` | **Node**: {r.get('node', 'N/A')}\n")
        m = r.get("metrics", {})
        if m:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            metric_labels = {
                "ttft_p50_ms": "TTFT p50 (ms)",
                "ttft_p99_ms": "TTFT p99 (ms)",
                "tpot_p50_ms": "TPOT p50 (ms/tok)",
                "tpot_p99_ms": "TPOT p99 (ms/tok)",
                "throughput_tok_s": "Throughput (tok/s)",
                "gpu_memory_gb": "GPU Memory (GB)",
                "mfu_pct": "MFU (%)",
            }
            for key, label in metric_labels.items():
                if key in m:
                    lines.append(f"| {label} | {m[key]} |")
            lines.append("")
        notes = r.get("notes", [])
        if notes:
            for note in notes:
                if note:
                    lines.append(f"- {note}")
            lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.date.today()
    repo_root = Path(__file__).parent.parent
    bench_dir = repo_root / "benchmarks"
    log_dir = repo_root / "logs"
    bench_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Find today's plan
    plan_path = bench_dir / f"{today}-plan.md"
    if not plan_path.exists():
        candidates = sorted(bench_dir.glob("*-plan.md"), reverse=True)
        if not candidates:
            sys.exit("No benchmark plan found. Run benchmark_analysis.py first.")
        plan_path = candidates[0]
        print(f"  Using most recent plan: {plan_path.name}")

    plan_text = plan_path.read_text()
    log_path = log_dir / f"{today}.log"

    # Step 1: generate run script
    run_script = bench_dir / f"{today}-run.sh"
    generate_run_script(plan_text, run_script)

    # Step 2: execute
    results, raw_output = execute_experiments(run_script, log_path)

    # Step 3: parse if needed
    if not results:
        results = parse_results_via_llm(raw_output)

    # Step 4: save JSON
    results_json = bench_dir / f"{today}-results.json"
    results_json.write_text(json.dumps(results, indent=2))
    print(f"  ✓ Results JSON → {results_json.name}")

    # Step 5: save markdown
    results_md = bench_dir / f"{today}-results.md"
    results_md.write_text(results_to_markdown(results, today))
    print(f"  ✓ Results markdown → {results_md.name}")

    print(f"\n[run_experiments] Done. {len(results)} experiments recorded.")
    return str(results_json)


if __name__ == "__main__":
    main()
