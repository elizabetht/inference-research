# Lab Environment

## Cluster: DGX Spark (2-node K8s)

**controller** (192.168.1.75) — control-plane, runs Claude Code and all scripts
**spark-01** (192.168.1.76) — worker, GB10 Grace Blackwell Superchip, 128GB unified memory (NVLink-C2C)
**spark-02** (192.168.1.77) — worker, GB10 Grace Blackwell Superchip, 128GB unified memory (NVLink-C2C)

- Architecture: ARM64 (aarch64)
- GPU: NVIDIA GB10 (SM121 compute cap 12.1)
- Inter-node: 10GbE (no NVLink between nodes)
- K8s namespace: `token-labs`
- Storage: Longhorn PVCs (`model-cache-*-spark-01`, `model-cache-*-spark-02`)
- Model cache path in pods: `/model-cache/huggingface`
- vLLM image: `ghcr.io/elizabetht/inference-images/vllm-serve:0.0.1`

## Active Serving Endpoint

- Model: `Qwen/Qwen3-Coder-Next-FP8`
- URL: `http://192.168.1.200:8000` (LoadBalancer `qwen3-coder-next-vllm-ext`)
- Deployment: PP=2 across spark-01 (leader) + spark-02 (worker) via Ray
- Max context: 131072 tokens

## Known Constraints

- vLLM compiled DAG (`VLLM_USE_RAY_COMPILED_DAG=1`) hangs on GB10 unified memory — keep disabled
- Ray OOM killer must be disabled (`RAY_memory_monitor_refresh_ms=0`)
- NCCL over 10GbE: set `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_P2P_DISABLE=1`
- gpu-memory-utilization: use 0.75 for PP=2 (leaves headroom for Ray overhead)
- SGLang: prefix caching + CUDA graphs ON by default (use `--disable-radix-cache` / `--disable-cuda-graph` to opt out)

## Source Repos

All repos: `~/src/github.com/elizabetht/`
- `inference-research/` — this project
- `token-labs/` — K8s manifests for model deployments
- `spark/` — InfiniBand + NIXL benchmarks
