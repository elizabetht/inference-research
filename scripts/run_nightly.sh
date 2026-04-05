#!/usr/bin/env bash
# run_nightly.sh — runs research.py + benchmark_analysis.py, commits results
# Designed to be invoked by cron at 2am.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%T)] $*" | tee -a "$LOG_FILE"; }

# ── Load env ─────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
[ -f "$HOME/.env" ] && source "$HOME/.env"
[ -f "$HOME/.anthropic_key" ] && export ANTHROPIC_API_KEY=$(cat "$HOME/.anthropic_key")

# GH_TOKEN: try gh CLI if env not set
if [ -z "${GITHUB_TOKEN:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
  export GH_TOKEN=$(gh auth token 2>/dev/null || true)
fi

export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"

cd "$REPO_DIR"

log "=== inference-radar nightly run: $DATE ==="

# ── Python venv ───────────────────────────────────────────────────────────────
VENV="$HOME/.venvs/inference-radar"
if [ ! -d "$VENV" ]; then
  log "Creating virtualenv..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --upgrade anthropic requests 2>>"$LOG_FILE" || true

# ── Run research ──────────────────────────────────────────────────────────────
log "Running research.py..."
python3 "$REPO_DIR/scripts/research.py" 2>&1 | tee -a "$LOG_FILE"

# ── Run benchmark analysis ────────────────────────────────────────────────────
log "Running benchmark_analysis.py..."
python3 "$REPO_DIR/scripts/benchmark_analysis.py" 2>&1 | tee -a "$LOG_FILE"

# ── Commit and push ───────────────────────────────────────────────────────────
log "Committing results..."
git -C "$REPO_DIR" add curations/ benchmarks/ logs/
git -C "$REPO_DIR" commit -m "chore: nightly curation $DATE" \
  --author="inference-radar[bot] <noreply@inference-radar>" \
  2>&1 | tee -a "$LOG_FILE" || log "Nothing to commit"

git -C "$REPO_DIR" push origin main 2>&1 | tee -a "$LOG_FILE" || log "Push failed (check remote)"

log "=== Done ==="
