#!/usr/bin/env python3
"""
inference-research: Automated daily curation of inference engine improvements.

Targets: vLLM, SGLang, TensorRT-LLM, llm-d, Dynamo
Runs nightly at 2am. Saves dated markdown to curations/YYYY-MM-DD.md
"""

import os
import sys
import json
import time
import datetime
import textwrap
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

REPOS = [
    {"name": "vLLM",       "slug": "vllm-project/vllm",              "tags": ["paged-attention", "chunked-prefill", "speculative-decoding"]},
    {"name": "SGLang",     "slug": "sgl-project/sglang",             "tags": ["radix-attention", "prefix-cache", "constrained-decoding"]},
    {"name": "TensorRT-LLM","slug": "NVIDIA/TensorRT-LLM",           "tags": ["quantization", "inflight-batching", "speculative-decoding"]},
    {"name": "llm-d",      "slug": "llm-d/llm-d",                    "tags": ["kubernetes", "disaggregation", "kv-cache"]},
    {"name": "Dynamo",     "slug": "ai-dynamo/dynamo",               "tags": ["disaggregation", "kv-routing", "nixl"]},
]

ARXIV_QUERIES = [
    "ti:inference AND ti:serving",
    "ti:speculative AND ti:decoding",
    "ti:KV cache AND ti:LLM",
    "abs:prefill decode disaggregation",
    "abs:PagedAttention OR abs:RadixAttention",
]

LOOKBACK_DAYS = 1  # PRs merged in last N days (2 on weekends via cron logic)

# Local LLM endpoint (SGLang on spark-01, tunnelled to localhost:30001 by run_nightly.sh)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.1.200:8000")
# Llama-3.1-8B via SGLang: 32K context window, adequate for full research prompts.
# Qwen2.5-7B via TRT-LLM has max_seq_len ~5184 (constrained by free_gpu_memory_fraction=0.05).
LLM_MODEL    = os.environ.get("LLM_MODEL",    "meta-llama/Llama-3.1-8B-Instruct")

# ── Helpers ───────────────────────────────────────────────────────────────────

def gh_api(path: str, token: str) -> dict | list:
    url = f"https://api.github.com/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "inference-research/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_recent_prs(repo_slug: str, token: str, since_iso: str) -> list[dict]:
    """Fetch merged PRs since a given ISO timestamp."""
    results = []
    page = 1
    while page <= 5:
        items = gh_api(
            f"repos/{repo_slug}/pulls?state=closed&sort=updated&direction=desc&per_page=50&page={page}",
            token,
        )
        if not items:
            break
        for pr in items:
            if pr.get("merged_at") and pr["merged_at"] >= since_iso:
                results.append({
                    "title": pr["title"],
                    "body": (pr.get("body") or "")[:600],
                    "url": pr["html_url"],
                    "merged_at": pr["merged_at"],
                    "labels": [l["name"] for l in pr.get("labels", [])],
                    "author": pr["user"]["login"],
                })
            elif pr["updated_at"] < since_iso:
                return results  # sorted by updated desc, can bail early
        page += 1
        time.sleep(0.5)
    return results


def fetch_recent_releases(repo_slug: str, token: str) -> list[dict]:
    """Fetch latest 3 releases."""
    try:
        releases = gh_api(f"repos/{repo_slug}/releases?per_page=3", token)
        return [{"name": r["name"], "tag": r["tag_name"], "body": (r.get("body") or "")[:800], "url": r["html_url"], "published_at": r["published_at"]} for r in releases]
    except Exception:
        return []


def fetch_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for recent inference papers."""
    # Build URL manually to avoid double-encoding of boolean operators
    encoded_query = urllib.parse.quote(query, safe=":()+")
    params = f"search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "inference-research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        print(f"  arXiv error for '{query}': {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = (entry.find("atom:summary", ns).text or "").strip()[:500]
        link_el = entry.find("atom:link[@rel='alternate']", ns)
        link = link_el.attrib["href"] if link_el is not None else ""
        published = entry.find("atom:published", ns).text[:10]
        papers.append({"title": title, "summary": summary, "url": link, "published": published})
    return papers


def llm_curate(prompt: str) -> str:
    """Call the local LLM endpoint (OpenAI-compatible) to curate/analyze content."""
    # Truncate prompt to ~24k chars (~6k tokens) to stay within 32K context window
    if len(prompt) > 24000:
        prompt = prompt[:24000] + "\n\n[truncated for context window]"
    payload = json.dumps({
        "model": LLM_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json", "host": "api.tokenlabs.run"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("ERROR: GITHUB_TOKEN or GH_TOKEN not set")

    today = datetime.date.today()
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    since_iso = since

    print(f"[inference-research] {today} — fetching updates since {since_iso}")

    # ── 1. GitHub PRs and releases per repo ─────────────────────────────────
    all_prs = {}
    all_releases = {}
    for repo in REPOS:
        print(f"  → {repo['name']} ({repo['slug']})")
        prs = fetch_recent_prs(repo["slug"], token, since_iso)
        releases = fetch_recent_releases(repo["slug"], token)
        all_prs[repo["name"]] = prs
        all_releases[repo["name"]] = releases
        print(f"     {len(prs)} PRs, {len(releases)} releases")
        time.sleep(1)

    # ── 2. arXiv papers ─────────────────────────────────────────────────────
    print("  → arXiv")
    arxiv_papers = []
    for q in ARXIV_QUERIES:
        papers = fetch_arxiv(q, max_results=4)
        for p in papers:
            if p not in arxiv_papers:
                arxiv_papers.append(p)
        time.sleep(2)
    # Deduplicate by title
    seen_titles = set()
    deduped_papers = []
    for p in arxiv_papers:
        if p["title"] not in seen_titles:
            seen_titles.add(p["title"])
            deduped_papers.append(p)
    arxiv_papers = deduped_papers[:20]
    print(f"     {len(arxiv_papers)} unique papers")

    # ── 3. Build context for Claude ─────────────────────────────────────────
    context_parts = [f"# Inference Engine Updates — {today}\n"]

    for repo in REPOS:
        name = repo["name"]
        prs = all_prs.get(name, [])
        releases = all_releases.get(name, [])
        context_parts.append(f"\n## {name}\n")
        if releases:
            context_parts.append(f"### Latest Release: {releases[0]['name']} ({releases[0]['published_at'][:10]})\n{releases[0]['body'][:400]}\n")
        if prs:
            context_parts.append(f"### Merged PRs (last {LOOKBACK_DAYS}d):\n")
            for pr in prs[:15]:
                context_parts.append(f"- [{pr['title']}]({pr['url']}) by @{pr['author']} | labels: {', '.join(pr['labels']) or 'none'}\n  {pr['body'][:200]}\n")
        else:
            context_parts.append("_No new PRs in this window._\n")

    context_parts.append("\n## Recent arXiv Papers\n")
    for p in arxiv_papers:
        context_parts.append(f"- [{p['title']}]({p['url']}) ({p['published']})\n  {p['summary'][:200]}\n")

    raw_context = "".join(context_parts)

    # ── 4. Claude curation prompt ────────────────────────────────────────────
    prompt = textwrap.dedent(f"""
    You are an expert inference systems engineer. Below is a daily digest of updates from the five major LLM inference projects: vLLM, SGLang, TensorRT-LLM, llm-d, and NVIDIA Dynamo, plus recent arXiv papers.

    Your task: produce a structured daily curation report.

    FORMAT REQUIREMENTS:
    1. **Executive Summary** (3-5 bullet points): most impactful changes across all projects today
    2. **Per-Project Breakdown**: for each project, list improvements with:
       - What changed (technical detail)
       - Why it matters (performance/latency/throughput/memory impact)
       - Estimated impact: 🔴 High / 🟡 Medium / 🟢 Low
    3. **Notable arXiv Papers**: 3-5 most relevant to production inference systems
    4. **DGX Spark Benchmark Candidates**: identify 2-4 improvements that could be benchmarked on a 2-node DGX Spark cluster (each node: DGX Spark = NVIDIA GB10 Grace Blackwell Superchip, 128GB unified memory, NVLink-C2C). For each candidate:
       - What to benchmark
       - Metric to measure (TTFT, TPOT, throughput tokens/s, MFU)
       - Baseline vs expected improvement
       - Rough implementation steps
    5. **Trend Signal**: 1-2 sentences on the macro direction inference engineering is moving this week

    Be technical and specific. Skip anything that's purely docs/CI/dependency bumps with no performance relevance.

    ---
    {raw_context}
    """).strip()

    print(f"  → Calling local LLM ({LLM_MODEL}) for curation...")
    try:
        curation = llm_curate(prompt)
    except Exception as e:
        sys.exit(f"ERROR calling local LLM at {LLM_BASE_URL}: {e}")

    # ── 5. Save output ───────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "curations"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{today}.md"

    header = f"""---
date: {today}
generated_at: {datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
sources:
  prs_fetched: {sum(len(v) for v in all_prs.values())}
  arxiv_papers: {len(arxiv_papers)}
---

"""
    out_path.write_text(header + curation)
    print(f"  ✓ Saved → curations/{today}.md")

    # ── 6. Also save raw data for audit ─────────────────────────────────────
    raw_dir = Path(__file__).parent.parent / "curations" / ".raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"{today}.json").write_text(json.dumps({
        "prs": all_prs,
        "releases": {k: v[:1] for k, v in all_releases.items()},
        "arxiv": arxiv_papers,
    }, indent=2))

    print(f"\n[inference-research] Done. Curation saved to curations/{today}.md")
    return str(out_path)


if __name__ == "__main__":
    main()
