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

if ! curl -sf -m 10 -H "Host: api.tokenlabs.run" "http://${LLM_EXT}:${LLM_PORT}/v1/models" -o /tmp/nightly-models.json 2>/dev/null; then
  log "ERROR: LLM gateway not reachable at ${LLM_EXT}:${LLM_PORT} — check inference pods in token-labs"
  exit 1
fi
log "LLM endpoint reachable: http://${LLM_EXT}:${LLM_PORT}"

export LLM_BASE_URL="http://${LLM_EXT}:${LLM_PORT}"

# /v1/models is served by the gateway control plane and answers 200 even when
# every backend is down, so it is not a readiness signal. Probe each advertised
# model with a real 1-token completion and take the first that actually serves.
MODELS=$(python3 -c 'import json;print("\n".join(m["id"] for m in json.load(open("/tmp/nightly-models.json"))["data"]))' 2>/dev/null || true)
[ -z "${MODELS:-}" ] && { log "ERROR: gateway advertised no models — aborting"; exit 1; }

LLM_MODEL=""
while IFS= read -r m; do
  [ -z "$m" ] && continue
  CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' \
    -H "Host: api.tokenlabs.run" -H 'Content-Type: application/json' \
    "http://${LLM_EXT}:${LLM_PORT}/v1/chat/completions" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" 2>/dev/null || echo 000)
  if [ "$CODE" = "200" ]; then
    LLM_MODEL="$m"; log "LLM model live: $m"; break
  fi
  log "model $m not serving (HTTP $CODE) — trying next"
done <<< "$MODELS"

if [ -z "$LLM_MODEL" ]; then
  log "ERROR: gateway advertises models but none serve completions — check inference pods in token-labs. Aborting before doing any work."
  exit 1
fi
export LLM_MODEL

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
git -C "$REPO_DIR" add curations/ benchmarks/
git -C "$REPO_DIR" diff --cached --quiet && \
  log "Nothing to commit" || \
  git -C "$REPO_DIR" commit -m "chore: nightly run $DATE" \
    --author="inference-research[bot] <noreply@inference-research>" \
    2>&1 | tee -a "$LOG_FILE"

git -C "$REPO_DIR" push origin main 2>&1 | tee -a "$LOG_FILE" || \
  log "WARNING: push failed (check remote / auth)"

# ── 6. Telegram summary ───────────────────────────────────────────────────────
TG_TOKEN="${TELEGRAM_BOT_TOKEN:-$(grep -h '^TELEGRAM_BOT_TOKEN=' "$HOME/.claude/channels/telegram/.env" 2>/dev/null | cut -d= -f2-)}"
TG_CHAT="8622076160"
CURATION="$REPO_DIR/curations/$DATE.md"
if [ -n "${TG_TOKEN:-}" ] && [ -f "$CURATION" ]; then
  log "--- Step 6: telegram summary ---"
  SUMMARY=$(head -c 2500 "$CURATION")
  curl -sf -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT}" \
    --data-urlencode "disable_web_page_preview=true" \
    --data-urlencode "text=inference-research nightly $DATE

${SUMMARY}

https://github.com/elizabetht/inference-research/tree/main/curations" \
    >/dev/null 2>&1 && log "telegram sent" || log "WARNING: telegram send failed"
else
  log "--- Step 6: telegram skipped (no token or no curation for $DATE) ---"
fi

log "=== Done: $DATE ==="
