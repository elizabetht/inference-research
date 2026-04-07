# daily-research

Run the daily inference-research research scan for $DATE (default: today).

## Step 1 — Load deduplication state

Read `.claude/memory/seen.jsonl`. Each line is a JSON object `{"id": "...", "first_seen": "YYYY-MM-DD", "title": "..."}`.
Build a set of known IDs to filter against.

## Step 2 — Fan out 7 parallel subagents

Launch ALL of the following agents simultaneously using the Agent tool:

**Framework scanners** (5):
- `framework-scanner` with FRAMEWORK=vLLM, REPO=vllm-project/vllm, TAGS=paged-attention,chunked-prefill,speculative-decoding
- `framework-scanner` with FRAMEWORK=SGLang, REPO=sgl-project/sglang, TAGS=radix-attention,prefix-cache,piecewise-cuda-graph
- `framework-scanner` with FRAMEWORK=TensorRT-LLM, REPO=NVIDIA/TensorRT-LLM, TAGS=inflight-batching,disaggregation,kv-cache
- `framework-scanner` with FRAMEWORK=llm-d, REPO=llm-d/llm-d, TAGS=kubernetes,disaggregation,kv-routing
- `framework-scanner` with FRAMEWORK=Dynamo, REPO=ai-dynamo/dynamo, TAGS=disaggregation,kv-routing,nixl

**Specialist scanners** (2):
- `arxiv-scanner` — recent inference papers
- `news-scanner` — HN, NVIDIA blog, vendor announcements

## Step 3 — Deduplicate

Filter all findings against seen.jsonl. Keep only items whose ID is NOT already in the set.
If a finding has no ID, derive one from its URL using the patterns in framework-scanner.md.

## Step 4 — Count and assess

If 0 new items: write a brief "No new findings today" report and skip to Step 6.
If 1-2 items: standard report.
If 3+ items: apply first-principles analysis to the top 2-3 most impactful findings.

## Step 5 — First-principles analysis (for top 2-3 findings only)

For each selected finding, apply Musk's 5-step process:
1. **Challenge**: What assumption does this change or invalidate?
2. **Delete**: What existing work becomes unnecessary if this is true?
3. **Simplify**: What is the minimal version of this idea that still delivers value?
4. **Accelerate**: How could this be validated faster on the DGX Spark cluster?
5. **Automate**: Should this become a recurring benchmark in `autoresearch/queue.yaml`?

## Step 6 — Write report

Save to `reports/YYYY-MM-DD.md` with this structure:

```markdown
---
date: YYYY-MM-DD
new_findings: N
frameworks_scanned: vLLM, SGLang, TensorRT-LLM, llm-d, Dynamo
---

# Inference Radar — YYYY-MM-DD

## Executive Summary
{3-5 bullets, most impactful items only}

## Framework Updates
{paste scanner outputs, deduplicated}

## arXiv
{paste arxiv-scanner output}

## Industry News
{paste news-scanner output}

## First-Principles Analysis
{Step 5 output for top findings, or "N/A — no novel findings today"}

## DGX Spark Benchmark Candidates
{List items flagged as benchmark candidates, with proposed queue.yaml entries}

## Open Questions
{1-3 questions this surfaced that aren't answered yet}
```

## Step 7 — Update seen.jsonl

Append one JSON line per new finding:
`{"id": "{id}", "first_seen": "YYYY-MM-DD", "title": "{title}"}`

## Step 8 — Commit and notify

```bash
cd ~/src/github.com/elizabetht/inference-research
git add reports/ .claude/memory/seen.jsonl
git commit -m "research: daily scan {YYYY-MM-DD} ({N} new findings)"
git push origin main
```

Send a Telegram summary (chat_id from environment) with:
- Date + finding count
- Top 1-2 items (title + one-line impact)
- Any new benchmark candidates added to queue

**Key principle**: Honest > padded. A short report on a slow day is correct. Never inflate findings.
