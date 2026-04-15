#!/usr/bin/env bash
# run_nightly.sh — nightly inference-research pipeline:
#   1. SSH tunnel  → localhost:30001 → spark-01:30001 (SGLang / Qwen3-0.6B)
#   2. research.py      → curations/YYYY-MM-DD.md
#   3. benchmark_analysis.py → benchmarks/YYYY-MM-DD-plan.md
#   4. run_experiments.py    → benchmarks/YYYY-MM-DD-run.sh + results.json + results.md
#   5. plot_results.py       → benchmarks/charts/
#   6. commit + push
#
# Cron: 0 2 * * * /path/to/run_nightly.sh
# No API key required — uses local SGLang on spark-01.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/$DATE.log"
LLM_EXT="192.168.1.200"
LLM_PORT=8000

mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%T)] $*" | tee -a "$LOG_FILE"; }

# ── SSH tunnel to spark-01 SGLang ─────────────────────────────────────────────

if ! curl -sf "http://${LLM_EXT}:${LLM_PORT}/health" >/dev/null 2>&1; then
  log "ERROR: LLM gateway not reachable at ${LLM_EXT}:${LLM_PORT} — check inference pods in token-labs"
  exit 1
fi
log "LLM endpoint ready: http://${LLM_EXT}:${LLM_PORT}"

export LLM_BASE_URL="http://${LLM_EXT}:${LLM_PORT}"
export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# GitHub token: gh CLI fallback
if [ -z "${GITHUB_TOKEN:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
  export GH_TOKEN=$(GH_CONFIG_DIR="$HOME/.config/gh" gh auth token 2>/dev/null || true)
fi
export GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

[ -z "${GITHUB_TOKEN:-}" ] && log "WARNING: GITHUB_TOKEN not set — GitHub PR fetching will fail"

cd "$REPO_DIR"

log "=== inference-research nightly run: $DATE ==="

# ── Python venv ───────────────────────────────────────────────────────────────
VENV="$HOME/.venvs/inference-research"
if [ ! -d "$VENV" ]; then
  log "Creating virtualenv at $VENV ..."
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -q --upgrade anthropic requests 2>>"$LOG_FILE" || true

# ── 1. Research + curation ────────────────────────────────────────────────────
log "--- Step 1: research.py ---"
if python3 "$REPO_DIR/scripts/research.py" 2>&1 | tee -a "$LOG_FILE"; then
  log "research.py OK"
else
  log "ERROR: research.py failed — aborting nightly run"
  exit 1
fi

# ── 2. Benchmark plan generation ──────────────────────────────────────────────
log "--- Step 2: benchmark_analysis.py ---"
if python3 "$REPO_DIR/scripts/benchmark_analysis.py" 2>&1 | tee -a "$LOG_FILE"; then
  log "benchmark_analysis.py OK"
else
  log "ERROR: benchmark_analysis.py failed — skipping experiments"
  SKIP_EXPERIMENTS=1
fi

# ── 3. Run experiments ────────────────────────────────────────────────────────
if [ -z "${SKIP_EXPERIMENTS:-}" ]; then
  log "--- Step 3: run_experiments.py ---"
  if python3 "$REPO_DIR/scripts/run_experiments.py" 2>&1 | tee -a "$LOG_FILE"; then
    log "run_experiments.py OK"
  else
    log "WARNING: run_experiments.py failed — results may be incomplete"
  fi
else
  log "--- Step 3: skipped (no benchmark plan) ---"
fi

# ── 4. Plot results ───────────────────────────────────────────────────────────
log "--- Step 4: plot_results.py ---"
for json_file in "$REPO_DIR"/benchmarks/*-results.json; do
  [ -f "$json_file" ] || continue
  python3 "$REPO_DIR/scripts/plot_results.py" "$json_file" 2>&1 | tee -a "$LOG_FILE" || \
    log "WARNING: plot_results.py failed for $json_file"
done

# ── 5. Commit and push ────────────────────────────────────────────────────────
log "--- Step 5: commit + push ---"
git -C "$REPO_DIR" add curations/ benchmarks/ logs/
git -C "$REPO_DIR" diff --cached --quiet && \
  log "Nothing to commit" || \
  git -C "$REPO_DIR" commit -m "chore: nightly run $DATE" \
    --author="inference-research[bot] <noreply@inference-research>" \
    2>&1 | tee -a "$LOG_FILE"

git -C "$REPO_DIR" push origin main 2>&1 | tee -a "$LOG_FILE" || \
  log "WARNING: push failed (check remote / auth)"

log "=== Done: $DATE ==="
