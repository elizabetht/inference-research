# Inference Autoresearch Program

## Objective

Maximize throughput (tokens/second) on the DGX Spark cluster while maintaining acceptable latency.

**Primary metric**: throughput (tok/s) at concurrency=32
**Secondary metrics**: TTFT p50/p99 (ms), ITL p50/p99 (ms/tok)
**Pass criteria**: ≥3% throughput improvement without TTFT or ITL regressing >20%

## Cluster

- spark-01: nvidia@192.168.1.76 — GB10 Grace Blackwell, 128GB unified memory, leader node
- spark-02: nvidia@192.168.1.77 — GB10 Grace Blackwell, 128GB unified memory, worker node
- K8s namespace: `token-labs`
- Model cache: Longhorn PVCs (`model-cache-{experiment}-spark-01/02`)
- Manifests: `~/src/github.com/elizabetht/token-labs/deploy/`
- Serving endpoint: `http://192.168.1.200:8000` (LoadBalancer)

## Workflow

1. Scheduler reads `queue.yaml`, selects highest-priority `queued` experiment
2. Deploys the experiment via `run_experiment.py` (applies K8s manifests, waits for ready)
3. Runs `benchmark.py` against the serving endpoint
4. Parses results, appends to `results.tsv`, updates `LEADERBOARD.md`
5. Marks experiment `done` or `failed` in queue.yaml
6. Commits and pushes; sends Telegram notification

## Queue Rules

- **One variable at a time**: each experiment changes exactly one thing from baseline
- **No retries**: failed experiments (OOM, timeout, crash) are marked `failed`, not re-queued with same config
- **Isolation**: shut down previous serving pod before launching next
- **Timeouts**: 30min server startup, 15min per benchmark run, 90min total per experiment
- **Approval required** for experiments that require new PVC provisioning (>300GB)

## What You Can Modify

- `autoresearch/queue.yaml` — add, remove, reprioritize experiments
- `~/src/github.com/elizabetht/token-labs/deploy/` — K8s manifests for new experiments

## What You Cannot Modify

- `autoresearch/benchmark.py` — immutable ground truth
- `autoresearch/scheduler.py` — runtime infrastructure
- `autoresearch/run_experiment.py` — deployment logic

## Optimization Targets (prioritized)

1. **FP8 KV cache** — reduce memory pressure, enable longer contexts
2. **Piecewise CUDA graph** (SGLang) — lower TTFT under saturation
3. **Chunked prefill** — reduce TTFT at high concurrency
4. **Prefix caching / RadixAttention** — improve cache hit rate for repeated prompts
5. **Tensor parallelism TP=2** — scale throughput for larger models
6. **GPU memory utilization** — tune 0.75→0.85 once Ray overhead is measured
7. **Speculative decoding** — reduce TPOT for chat workloads
8. **Framework alternatives** — SGLang vs vLLM on Blackwell

## Known DGX Spark Constraints

- `VLLM_USE_RAY_COMPILED_DAG=0` — compiled DAG hangs on GB10 unified memory
- `RAY_memory_monitor_refresh_ms=0` — disable Ray OOM killer
- `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_P2P_DISABLE=1` — 10GbE inter-node
- SM121 (compute cap 12.1) — some kernels fall back to PTX; native SM121 kernels ~10-20% faster when available
- `--enforce-eager` may be needed if CUDA graph capture fails on aarch64
