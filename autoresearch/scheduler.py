#!/usr/bin/env python3
"""
Experiment scheduler for inference-research.

Reads autoresearch/queue.yaml, picks the next queued experiment,
runs it via run_experiment.py, records results to results.tsv,
updates LEADERBOARD.md, and loops.

Exits when:
  - queue is fully drained (all done/failed)
  - a STOP file appears at repo root
  - fatal error

Usage:
  python autoresearch/scheduler.py [--once]
  --once: run exactly one experiment then exit (useful for cron)
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT    = Path(__file__).parent.parent
QUEUE_FILE   = REPO_ROOT / "autoresearch" / "queue.yaml"
RESULTS_FILE = REPO_ROOT / "autoresearch" / "results.tsv"
LEADERBOARD  = REPO_ROOT / "LEADERBOARD.md"
STOP_FILE    = REPO_ROOT / "STOP"
LOCK_FILE    = REPO_ROOT / "autoresearch" / ".scheduler.lock"
VENV         = Path.home() / ".venvs" / "inference-research"

POLL_SEC           = int(os.environ.get("SCHEDULER_POLL_SEC", "30"))
EXPERIMENT_TIMEOUT = int(os.environ.get("EXPERIMENT_TIMEOUT_SEC", str(90 * 60)))  # 90 min

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8622076160")

LLM_ENDPOINT = os.environ.get("LLM_BASE_URL", "http://192.168.1.200:8000")


# ── Queue helpers ─────────────────────────────────────────────────────────────

def load_queue() -> dict:
    return yaml.safe_load(QUEUE_FILE.read_text())


def save_queue(data: dict):
    QUEUE_FILE.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False))


def next_experiment(data: dict):
    queued = [e for e in data["experiments"] if e["status"] == "queued"]
    if not queued:
        return None
    return sorted(queued, key=lambda e: e.get("priority", 99))[0]


def update_experiment(data: dict, name: str, **kwargs):
    for exp in data["experiments"]:
        if exp["name"] == name:
            exp.update(kwargs)
            return
    raise KeyError(f"Experiment {name!r} not in queue")


# ── Result helpers ────────────────────────────────────────────────────────────

TSV_HEADER = "\t".join([
    "name", "framework", "model", "node_config",
    "throughput_tok_s", "ttft_p50_ms", "ttft_p99_ms",
    "itl_p50_ms", "itl_p99_ms", "status", "date_run", "notes",
])


def ensure_results_file():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(TSV_HEADER + "\n")


def append_result(exp: dict, result: dict | None, status: str, notes: str = ""):
    ensure_results_file()
    r = result or {}
    primary = r.get("primary", {})
    row = "\t".join(str(x) for x in [
        exp["name"],
        exp.get("framework", ""),
        exp.get("model", ""),
        exp.get("node_config", ""),
        primary.get("throughput_tok_s", ""),
        primary.get("ttft_p50_ms", ""),
        primary.get("ttft_p99_ms", ""),
        primary.get("itl_p50_ms", ""),
        primary.get("itl_p99_ms", ""),
        status,
        time.strftime("%Y-%m-%d"),
        notes.replace("\t", " "),
    ])
    with open(RESULTS_FILE, "a") as f:
        f.write(row + "\n")


# ── Leaderboard ───────────────────────────────────────────────────────────────

def rebuild_leaderboard():
    if not RESULTS_FILE.exists():
        return
    rows = []
    with open(RESULTS_FILE) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            rows.append(dict(zip(header, vals)))

    done = [r for r in rows if r.get("status") == "done" and r.get("throughput_tok_s")]
    done.sort(key=lambda r: float(r["throughput_tok_s"]), reverse=True)

    lines = ["# Inference Autoresearch Leaderboard\n",
             f"_Updated: {time.strftime('%Y-%m-%d %H:%M UTC')}_\n\n",
             "| Rank | Experiment | Framework | Model | Throughput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) |\n",
             "|------|-----------|-----------|-------|-------------------|---------------|-------------|\n"]

    for i, r in enumerate(done, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['framework']} | {r['model'].split('/')[-1]} | "
            f"{r['throughput_tok_s']} | {r['ttft_p50_ms']} | {r['itl_p50_ms']} |\n"
        )

    LEADERBOARD.write_text("".join(lines))
    print(f"[scheduler] Leaderboard updated ({len(done)} entries)", flush=True)


# ── Telegram ──────────────────────────────────────────────────────────────────

def telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=payload,
            headers={"content-type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[scheduler] Telegram error: {e}", flush=True)


# ── Git ───────────────────────────────────────────────────────────────────────

def git_commit(message: str):
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "add",
             "autoresearch/queue.yaml", "autoresearch/results.tsv", "LEADERBOARD.md"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "commit", "-m", message,
             "--author=inference-research[bot] <noreply@inference-research>"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "push", "origin", "main"],
            check=True, capture_output=True,
        )
        print(f"[scheduler] Committed: {message}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[scheduler] Git error: {e.stderr.decode()[:200]}", flush=True)


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one_experiment(exp: dict) -> tuple[dict | None, str]:
    """Deploy experiment, run benchmark, return (result_dict, notes)."""
    name = exp["name"]
    print(f"\n[scheduler] Starting experiment: {name}", flush=True)

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{time.strftime('%Y-%m-%d')}-{name}.log"
    result_file = REPO_ROOT / "autoresearch" / "results" / f"{name}.json"
    result_file.parent.mkdir(exist_ok=True)

    # Python in venv
    python = str(VENV / "bin" / "python3") if (VENV / "bin" / "python3").exists() else sys.executable

    # Ensure aiohttp is installed
    subprocess.run([python, "-m", "pip", "install", "-q", "aiohttp", "pyyaml"],
                   capture_output=True)

    runner = str(REPO_ROOT / "autoresearch" / "run_experiment.py")
    endpoint = exp.get("endpoint", LLM_ENDPOINT)
    cmd = [python, runner,
           "--name", name,
           "--queue", str(QUEUE_FILE),
           "--benchmark", str(REPO_ROOT / "autoresearch" / "benchmark.py"),
           "--output", str(result_file),
           "--log", str(log_file),
           "--endpoint", endpoint]

    print(f"[scheduler] Running: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            timeout=EXPERIMENT_TIMEOUT,
            capture_output=False,  # let output flow to terminal/cron log
        )
        if proc.returncode != 0:
            return None, f"run_experiment.py exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        return None, f"timeout after {EXPERIMENT_TIMEOUT//60}m"
    except Exception as e:
        return None, str(e)

    # Parse result
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
            primary = result.get("primary", {})
            notes = (f"throughput={primary.get('throughput_tok_s')} "
                     f"ttft_p50={primary.get('ttft_p50_ms')}")
            return result, notes
        except Exception as e:
            return None, f"result parse error: {e}"

    return None, "no result file produced"


# ── Main loop ─────────────────────────────────────────────────────────────────

def acquire_lock() -> int:
    """Acquire an exclusive lock file so only one scheduler instance runs at a time."""
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fd.close()
        sys.exit("[scheduler] Another scheduler instance is already running (lock held). Exiting.")
    lock_fd.write(str(os.getpid()) + "\n")
    lock_fd.flush()
    return lock_fd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one experiment then exit")
    args = parser.parse_args()

    lock_fd = acquire_lock()

    ensure_results_file()
    print(f"[scheduler] Starting. queue={QUEUE_FILE}", flush=True)

    while True:
        if STOP_FILE.exists():
            print("[scheduler] STOP file found — exiting", flush=True)
            break

        data = load_queue()
        exp = next_experiment(data)

        if exp is None:
            print("[scheduler] Queue drained — nothing to run", flush=True)
            break

        # Mark running
        update_experiment(data, exp["name"], status="running")
        save_queue(data)

        result, notes = run_one_experiment(exp)

        data = load_queue()
        status = "done" if result else "failed"
        update_experiment(data, exp["name"], status=status,
                          date_run=time.strftime("%Y-%m-%d"))

        if result:
            primary = result.get("primary", {})
            # Write result back into queue for reference
            update_experiment(data, exp["name"], result={
                "throughput_tok_s": primary.get("throughput_tok_s"),
                "ttft_p50_ms": primary.get("ttft_p50_ms"),
                "ttft_p99_ms": primary.get("ttft_p99_ms"),
                "itl_p50_ms": primary.get("itl_p50_ms"),
                "itl_p99_ms": primary.get("itl_p99_ms"),
                "notes": notes,
            })

        save_queue(data)
        append_result(exp, result, status, notes)
        rebuild_leaderboard()

        msg = (
            f"✅ {exp['name']}: {notes}" if status == "done"
            else f"❌ {exp['name']} FAILED: {notes}"
        )
        print(f"[scheduler] {msg}", flush=True)
        telegram(f"inference-research\n{msg}")
        git_commit(f"autoresearch: {status} {exp['name']}")

        if args.once:
            break

        time.sleep(POLL_SEC)

    print("[scheduler] Done", flush=True)


if __name__ == "__main__":
    main()
