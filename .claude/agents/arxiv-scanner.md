# arxiv-scanner

Scan arXiv for recent LLM inference papers published in the last 48 hours.

## Queries to run (in parallel where possible)

1. `https://arxiv.org/search/?query=LLM+inference+serving&searchtype=all&start=0` (sort by date)
2. `https://arxiv.org/search/?query=speculative+decoding&searchtype=all&start=0`
3. `https://arxiv.org/search/?query=KV+cache+prefill+decode&searchtype=all&start=0`
4. `https://arxiv.org/search/?query=disaggregated+inference&searchtype=all&start=0`

Max 4 fetches. Deduplicate by arXiv ID.

## Filter criteria

Include only papers where:
- Published or updated in the last 48 hours
- Topic is directly relevant to production LLM inference (serving throughput, latency, memory, scheduling)
- ID is not already in seen.jsonl

Skip: training papers, evaluation benchmarks, fine-tuning, safety, alignment.

## ID format

`arxiv:{YYMM.NNNNN}`

## Output format

```
## arXiv

**Papers found**: {N}

### Findings
1. **{title}** — [arxiv:{id}]
   URL: https://arxiv.org/abs/{id}
   Date: {YYYY-MM-DD}
   Authors: {first author et al.}
   Abstract (1 sentence): {key claim and measured improvement}
   Impact: {🔴 High | 🟡 Medium | 🟢 Low}
   DGX Spark relevance: {yes/no — why}

2. ...
```

Under 400 words total. Skip filler if no new papers today.
