#!/usr/bin/env bash
# run-daily.sh — Daily inference-research pipeline
#
# Two independent loops run each night at 2am:
#   1. Daily research scan  (claude /daily-research)
#   2. Autoresearch experiments  (autoresearch/scheduler.py --once)
#
# Cron: 0 2 * * * /home/nvidia/src/github.com/elizabetht/inference-research/run-daily.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$REPO/logs"
CRON_LOG="$LOG_DIR/cron.log"
DAILY_LOG="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR" "$REPO/reports"

log() { echo "[$(date -u +%T)] $*" | tee -a "$DAILY_LOG" "$CRON_LOG"; }

# ── GitHub token ──────────────────────────────────────────────────────────────
if [ -z "${GH_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  export GH_TOKEN=$(GH_CONFIG_DIR="$HOME/.config/gh" gh auth token 2>/dev/null || true)
fi
export GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(cat "$HOME/.claude/channels/telegram/.env" 2>/dev/null | grep TELEGRAM_BOT_TOKEN | cut -d= -f2 || true)}"
export TELEGRAM_CHAT_ID="8622076160"

log "=== inference-research daily run: $DATE ==="

# ── Python venv ───────────────────────────────────────────────────────────────
VENV="$HOME/.venvs/inference-research"
if [ ! -d "$VENV" ]; then
  log "Creating venv..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --upgrade aiohttp pyyaml requests 2>>"$DAILY_LOG" || true

# ── 1. Daily research scan ────────────────────────────────────────────────────
log "--- Step 1: daily research scan ---"
if command -v claude >/dev/null 2>&1; then
  claude --dangerously-skip-permissions \
    --print \
    "/daily-research $DATE" \
    2>&1 | tee -a "$DAILY_LOG" || log "WARNING: claude /daily-research failed"
else
  log "WARNING: claude CLI not found — falling back to research.py"
  python3 "$REPO/scripts/research.py" 2>&1 | tee -a "$DAILY_LOG" || \
    log "WARNING: research.py also failed"
fi

# ── 2. Autoresearch: run next queued experiment ───────────────────────────────
log "--- Step 2: autoresearch scheduler (--once) ---"
if kubectl get pods -n token-labs 2>/dev/null | grep -q "qwen3-coder-next"; then
  log "Existing vllm pods found — scheduler will reuse endpoint"
fi

python3 "$REPO/autoresearch/scheduler.py" --once 2>&1 | tee -a "$DAILY_LOG" || \
  log "WARNING: scheduler failed (check logs)"

log "=== Done: $DATE ==="
