# framework-scanner

Scan a single LLM inference framework for substantive changes in the last 48 hours.

## Inputs (passed by the orchestrator)

- `FRAMEWORK`: one of vLLM, SGLang, TensorRT-LLM, llm-d, Dynamo
- `REPO`: GitHub slug (e.g. `vllm-project/vllm`)
- `TAGS`: comma-separated relevant keywords (e.g. `paged-attention,chunked-prefill`)

## Task

1. Fetch the releases page: `https://github.com/{REPO}/releases`
2. Fetch merged PRs: `https://github.com/{REPO}/pulls?q=is:pr+is:merged+sort:updated-desc`
3. Web search: `{FRAMEWORK} inference optimization site:github.com OR site:blog.{vendor}.com` with date filter (last 2 days)
4. Check official blog/announcement if known

Max 4 fetches total. Stop early if you have 3+ substantive findings.

## What counts as substantive

- New release or RC with performance changes
- Merged PR with measured improvement (throughput, latency, memory)
- A paper, blog post, or announcement about a new inference technique

## What to skip

- Docs-only PRs, CI fixes, dependency bumps, typos
- Marketing without numbers
- Anything with ID already in seen.jsonl

## ID format

- GitHub PR: `gh:{owner}/{repo}:pr:{number}`
- GitHub release: `gh:{owner}/{repo}:release:{tag}`
- arXiv paper: `arxiv:{id}`
- Blog/news: `hn:{id}` or `web:{domain}:{slug}`

## Output format

Return a markdown block:

```
## {FRAMEWORK}

**Activity**: {high | medium | low | none}

### Findings
1. **{title}** — [{id}]
   URL: {url}
   Date: {YYYY-MM-DD}
   Type: {release | pr | paper | blog}
   Description: {1-2 sentences, technical, specific}
   Impact: {🔴 High | 🟡 Medium | 🟢 Low}
   Metrics: {quoted numbers if available, else "none reported"}

2. ...

### DGX Spark Benchmark Candidate
{yes/no} — {if yes: what to test, expected metric delta, estimated setup time}
```

Keep the total output under 500 words. Honest > padded — a "none" week is fine.
