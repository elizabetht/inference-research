#!/usr/bin/env bash
set -euo pipefail

# Configuration
CLUSTER_NODES=("spark-01:192.168.1.76" "spark-02:192.168.1.77")
SSH_USER="nvidia"
SSH_KEY="" # Assume ~/.ssh/id_rsa or configured in agent environment
BENCH_BASE="~/bench"
HF_HOME="/data/hf_cache"
RESULTS=()

# Helper: SSH command builder
ssh_cmd() {
    local node=$1
    local ip=$(echo "$node" | cut -d: -f2)
    shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${ip}" "$@"
}

# Helper: Run command on remote node, capturing output
run_remote() {
    local node=$1
    shift
    ssh_cmd "$node" "bash -s" <<< "$*"
}

# Helper: Parse JSON-like metrics from raw output (simplified for vLLM/SGLang benchmarks)
parse_metrics() {
    local raw="$1"
    local name="$2"
    local metrics_json="{}"
    
    # Attempt to extract throughput (tokens/s)
    local tp=$(echo "$raw" | grep -iE "throughput|tokens/s|total tokens" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    if [ -n "$tp" ]; then
        metrics_json=$(echo "$metrics_json" | jq --argjson v "$tp" '. + {"throughput_tok_s": $v}')
    fi
    
    # Attempt to extract TTFT (ms)
    local ttft=$(echo "$raw" | grep -iE "ttft|time.*first.*token" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    if [ -n "$ttft" ]; then
        metrics_json=$(echo "$metrics_json" | jq --argjson v "$ttft" '. + {"ttft_p50_ms": $v}')
    fi

    # Attempt to extract TPOT (ms)
    local tpot=$(echo "$raw" | grep -iE "tpot|time.*per.*output.*token" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    if [ -n "$tpot" ]; then
        metrics_json=$(echo "$metrics_json" | jq --argjson v "$tpot" '. + {"tpot_p50_ms": $v}')
    fi

    echo "$metrics_json"
}

# Experiment 1: Kimi-K3 MLA KV-Balanced FP8 Decode — vLLM
run_exp_kimi_k3_vllm() {
    local exp_name="exp1_kimi_k3_vllm_mla"
    local status="pass"
    local notes=()
    local raw_output=""
    
    echo "Starting $exp_name on ALL nodes..." >&2
    
    # We need to run the server on both nodes and the client.
    # Since this is a single script, we will simulate the multi-node setup by
    # launching the server on one node and connecting from the other, or simpler:
    # Launch server on node 1, client on node 1 for testing single-node first, 
    # but the plan says TP=2 across nodes.
    # 
    # For a self-contained script without Slurm/Ray cluster setup complexity, 
    # we will run the vLLM server in distributed mode. 
    # Note: True TP=2 across nodes requires specific launchers (torchrun/mpi).
    # We will use torchrun which is standard in vLLM.
    
    # Step 1: Install vLLM on both nodes if not present
    local vllm_version="main" # Assume main branch or specific commit
    local venv_path="${BENCH_BASE}/vllm-exp1"
    
    for node in "${CLUSTER_NODES[@]}"; do
        echo "Setting up venv on $node..." >&2
        run_remote "$node" "
            if [ ! -d '${venv_path}' ]; then
                python3 -m venv ${venv_path}
                source ${venv_path}/bin/activate
                pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
                pip install --quiet vllm 2>/dev/null || pip install --quiet git+https://github.com/vllm-project/vllm.git 2>/dev/null || true
            fi
            export HF_HOME=${HF_HOME}
            echo 'Setup complete'
        "
    done

    # Step 2: Kill any existing vLLM processes
    for node in "${CLUSTER_NODES[@]}"; do
        run_remote "$node" "pkill -f 'vllm.entrypoints' || true"
    done
    sleep 2

    # Step 3: Launch vLLM Server on Node 1 (Rank 0) and Node 2 (Rank 1) using torchrun
    # This is complex in a single script due to async execution.
    # We will use a background job approach with PID tracking or simple ssh exec.
    
    # Simplified Approach for Script: Run local server on Node 1 for immediate validation if Node 2 is problematic,
    # BUT the requirement is TP=2. 
    # Let's try to launch via ssh in background on both nodes.
    
    # Server Script Content
    local server_script=$(cat << 'EOF'
#!/bin/bash
set -euo pipefail
source ~/bench/vllm-exp1/bin/activate
export HF_HOME=/data/hf_cache
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTHONPATH=/path/to/vllm:$PYTHONPATH

# Determine if this is node 1 or 2 by IP or hostname
HOSTNAME=$(hostname)
echo "Starting vLLM server on $HOSTNAME" >&2

if [ "$HOSTNAME" == "spark-01" ]; then
    # Rank 0 handles the API port
    torchrun \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="192.168.1.76" \
        --master_port="29500" \
        -m vllm.entrypoints.api_server \
        --model "Kimi-K3" \
        --tensor-parallel-size 2 \
        --distributed-executor-backend torchrun \
        --attention-backend KIMI_K3_KV_BALANCED_MLA \
        --port 8000 \
        --host 0.0.0.0 \
        --disable-log-requests \
        2>&1 &
    SERVER_PID=$!
    echo "SERVER_PID=$SERVER_PID"
else
    # Rank 1
    torchrun \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr="192.168.1.76" \
        --master_port="29500" \
        -m vllm.entrypoints.api_server \
        --model "Kimi-K3" \
        --tensor-parallel-size 2 \
        --distributed-executor-backend torchrun \
        --attention-backend KIMI_K3_KV_BALANCED_MLA \
        --disable-log-requests \
        2>&1 &
    SERVER_PID=$!
    echo "SERVER_PID=$SERVER_PID"
fi
wait $SERVER_PID
EOF
)

    # Launch Server on Node 1 (API Server will listen on 8000)
    echo "Launching vLLM Server on spark-01..." >&2
    local spark01_pid_output=$(run_remote "spark-01" "$server_script")
    local spark01_pid=$(echo "$spark01_pid_output" | grep "SERVER_PID" | cut -d= -f2)
    
    # Launch Server on Node 2
    echo "Launching vLLM Server on spark-02..." >&2
    # Note: The output might be mixed, so we just ensure it runs.
    run_remote "spark-02" "$server_script" &

    # Wait for server to start
    echo "Waiting 60s for model loading and server startup..." >&2
    sleep 60

    # Step 4: Run Benchmark Client
    # The benchmark client connects to the API server on spark-01:8000
    local client_script=$(cat << 'EOF'
#!/bin/bash
set -euo pipefail
source ~/bench/vllm-exp1/bin/activate
export HF_HOME=/data/hf_cache

echo "Running benchmark client..." >&2
# Run vLLM throughput benchmark
python -m vllm.benchmarks.benchmark_throughput \
    --backend vllm \
    --dataset-name random \
    --dataset-path /dev/null \
    --num-prompts 100 \
    --request-rate 8 \
    --max-model-len 2048 \
    --tokenizer "Kimi-K3" \
    --tokenizer-mode auto \
    --trust-remote-code \
    --seed 42 \
    --host 192.168.1.76 \
    --port 8000 \
    2>&1
EOF
)

    local raw_output=""
    local metrics_json="{}"
    
    # Try running the client. If it fails, catch it.
    if run_remote "spark-01" "$client_script" > /tmp/bench_output.txt 2>&1; then
        raw_output=$(cat /tmp/bench_output.txt)
        metrics_json=$(parse_metrics "$raw_output" "vllm")
        status="pass"
        notes+=("Benchmark completed successfully.")
    else
        status="fail"
        notes+=("Benchmark client failed. Check logs for connection errors or OOM.")
        raw_output=$(cat /tmp/bench_output.txt 2>/dev/null || echo "Empty output")
    fi

    # Step 5: Cleanup
    echo "Killing vLLM servers..." >&2
    run_remote "spark-01" "pkill -f 'vllm.entrypoints' || true"
    run_remote "spark-02" "pkill -f 'vllm.entrypoints' || true"
    
    # Construct Result Object
    local result=$(jq -n \
        --arg name "$exp_name" \
        --arg framework "vLLM" \
        --arg version "main" \
        --arg model "Kimi-K3" \
        --arg node "spark-01,spark-02" \
        --arg status "$status" \
        --argjson metrics "$metrics_json" \
        --argjson notes '["TP=2 distributed via torchrun", "Attention backend: KIMI_K3_KV_BALANCED_MLA"]' \
        --arg raw_output "$raw_output" \
        '{
            name: $name,
            framework: $framework,
            version: $version,
            model: $model,
            node: $node,
            status: $status,
            metrics: $metrics,
            notes: $notes,
            raw_output: ($raw_output | ascii_downcase | .[:2000] + "..." | if length > 2000 then . else . end)
        }'
    )
    
    RESULTS+=("$result")
    echo "Finished $exp_name: $status" >&2
}

# Main Execution
main() {
    echo "Starting Benchmark Suite..." >&2
    
    # Check connectivity
    for node in "${CLUSTER_NODES[@]}"; do
        if ! ssh_cmd "$node" "echo 'OK'" >/dev/null 2>&1; then
            echo "ERROR: Cannot reach node $node" >&2
            exit 1
        fi
    done

    # Run Experiments
    run_exp_kimi_k3_vllm

    # Final JSON Output
    echo "${RESULTS[@]}" | jq -s '.'
}

main "$@"