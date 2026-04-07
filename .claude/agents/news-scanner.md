# news-scanner

Scan industry sources for LLM inference news in the last 48 hours.

## Sources to check

1. Hacker News: `https://hn.algolia.com/api/v1/search?query=inference+LLM&tags=story&numericFilters=created_at_i>TIMESTAMP` (use 48h ago timestamp)
2. Web search: `LLM inference optimization announcement` with date filter (last 2 days)
3. NVIDIA blog: `https://developer.nvidia.com/blog` (scan for inference posts)
4. Vendor engineering blogs: check if any of vLLM/SGLang/TRT-LLM posted anything

Max 4 fetches.

## Filter criteria

- Must be substantive: product launches, benchmark results, architecture announcements
- Skip: opinion pieces, tutorials, generic AI hype, anything without specific numbers

## ID format

- HN: `hn:{story_id}`
- Blog/web: `web:{domain}:{slug-or-title-hash}`

## Output format

```
## Industry News

**Items found**: {N}

### Findings
1. **{title}** — [{id}]
   URL: {url}
   Date: {YYYY-MM-DD}
   Source: {HN | NVIDIA Blog | vendor blog | other}
   Summary: {1-2 sentences with specific claims/numbers}
   Impact: {🔴 High | 🟡 Medium | 🟢 Low}

2. ...
```

Under 300 words. Honest > padded.
