# inference-radar

Automated nightly curation of LLM inference engine improvements — vLLM, SGLang, TensorRT-LLM, llm-d, Dynamo — with concrete benchmark plans for the home DGX Spark cluster.

Inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) but focused on **inference systems** rather than training experiments.

---

## What it does

Every night at 2am:

1. **Fetches** merged PRs and latest releases from the five major inference repos
2. **Scrapes** recent arXiv papers on LLM inference / serving optimization
3. **Curates** with Claude Opus — ranks by impact, explains *why* each change matters
4. **Generates** a benchmark plan targeting the 2-node DGX Spark cluster
5. **Commits** dated markdown to this repo

---

## Repo layout

```
curations/
  YYYY-MM-DD.md       ← daily curation report (Claude-curated)
  .raw/YYYY-MM-DD.json ← raw GitHub + arXiv data for audit

benchmarks/
  YYYY-MM-DD-plan.md  ← runnable benchmark plan for DGX Spark cluster

scripts/
  research.py         ← GitHub + arXiv fetch → Claude curation
  benchmark_analysis.py ← turns curation into concrete benchmark steps
  run_nightly.sh      ← cron entrypoint: runs both, commits + pushes

logs/
  YYYY-MM-DD.log      ← nightly run logs
```

---

## Targets

| Project | Repo | Focus areas |
|---------|------|-------------|
| vLLM | vllm-project/vllm | PagedAttention, chunked prefill, speculative decoding |
| SGLang | sgl-project/sglang | RadixAttention, prefix caching, constrained decoding |
| TensorRT-LLM | NVIDIA/TensorRT-LLM | Quantization, inflight batching, Blackwell kernels |
| llm-d | llm-d/llm-d | K8s-native serving, prefill/decode disaggregation |
| Dynamo | ai-dynamo/dynamo | KV routing, NIXL, disaggregated inference OS |

---

## Cluster

```
spark-01  192.168.1.76  DGX Spark  128GB unified (NVLink-C2C)
spark-02  192.168.1.77  DGX Spark  128GB unified (NVLink-C2C)
controller 192.168.1.75  CPU only   orchestrates workers
```

---

## Setup

### 1. Dependencies

```bash
python3 -m venv ~/.venvs/inference-radar
source ~/.venvs/inference-radar/bin/activate
pip install anthropic requests
```

### 2. Environment

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GH_TOKEN=$(gh auth token)   # or set GITHUB_TOKEN
```

### 3. Run manually

```bash
python3 scripts/research.py
python3 scripts/benchmark_analysis.py
```

### 4. Cron (set up automatically)

Runs via `crontab` on the controller at `2:00am` daily:

```cron
0 2 * * * /home/nvidia/src/github.com/elizabetht/inference-radar/scripts/run_nightly.sh
```

---

## Reading the curations

Each `curations/YYYY-MM-DD.md` contains:

- **Executive Summary** — 3-5 bullet top changes across all projects
- **Per-Project Breakdown** — what changed, why it matters, 🔴/🟡/🟢 impact rating
- **Notable arXiv Papers** — top inference papers published that day
- **DGX Spark Benchmark Candidates** — specific experiments to run with metrics
- **Trend Signal** — macro direction inference engineering is moving

Each `benchmarks/YYYY-MM-DD-plan.md` contains runnable bash commands to execute on the cluster.

---

## Adding new targets

Edit `REPOS` in `scripts/research.py` to add more GitHub repos. Edit `ARXIV_QUERIES` to tune the paper search.
