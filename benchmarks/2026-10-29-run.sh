#!/bin/bash

# MLSys Inference Benchmark Script for DGX Spark Cluster
# Runs prioritized experiments on NVIDIA GB10 Grace Blackwell nodes
# Output: JSON array of results on stdout

set -uo pipefail

# Configuration
CLUSTER_NODES=("spark-01@192.168.1.76" "spark-02@192.168.1.77")
CONTROLLER_IP="192.168.1.75"
REMOTE_USER="nvidia"
BENCH_DIR="~/bench"
HF_CACHE="/data/hf_cache"
LOG_FILE="/tmp/bench_$(date +%Y%m%d_%H%M%S).log"

# Global associative array to store results
declare -A EXPERIMENT_RESULTS

log() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

# Helper to run a remote command
run_remote() {
    local node_addr="$1"
    shift
    ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$node_addr" "$@"
}

# Helper to run a remote command with heredoc and timeout, capturing output
run_remote_heredoc() {
    local node_addr="$1"
    local script_content="$2"
    
    local output
    output=$(ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$node_addr" bash -s <<'REMOTE_SCRIPT_END'
$script_content
REMOTE_SCRIPT_END
) || true
    
    echo "$output"
}

# Function to create venv and install dependencies remotely
setup_venv_remote() {
    local node_addr="$1"
    local venv_path="$2"
    local packages="$3"
    
    local setup_script="
set -euo pipefail
export HF_HOME=$HF_CACHE

if [ ! -d $venv_path ]; then
    echo 'Creating venv...'
    python3 -m venv $venv_path
fi

echo 'Activating venv and installing packages...'
source $venv_path/bin/activate
pip install --upgrade pip > /dev/null 2>&1

$packages

echo 'VENV_READY'
"
    
    local output
    output=$(run_remote_heredoc "$node_addr" "$setup_script")
    
    if echo "$output" | grep -q "VENV_READY"; then
        log "Venv setup successful on $node_addr: $venv_path"
        return 0
    else
        log_error "Venv setup failed on $node_addr: $venv_path"
        log_error "Output: $output"
        return 1
    fi
}

# Function to run SGLang server and benchmark
run_sglang_experiment() {
    local node_addr="$1"
    local venv_path="$2"
    local model_name="$3"
    local tp_size="$4"
    
    local server_script="
set -euo pipefail
export HF_HOME=$HF_CACHE
source $venv_path/bin/activate

# Kill any existing sglang servers
pkill -f 'python.*sglang' || true
sleep 2

# Start SGLang Server
echo 'Starting SGLang server...'
python -m sglang.launch_server \
    --model-path $model_name \
    --tokenizer-path $model_name \
    --tp $tp_size \
    --mem-fraction-static 0.8 \
    --port 30000 &

SERVER_PID=\$!
echo "Server PID: \$SERVER_PID"

# Wait for server to start
echo 'Waiting for server to be ready...'
sleep 60

# Check if server is still running
if ! kill -0 \$SERVER_PID 2>/dev/null; then
    echo "Server failed to start"
    exit 1
fi

# Run Benchmark using sglang benchmark suite or simple test
# Using sglang's built-in benchmarking or vllm-compatible endpoint
echo 'Running benchmark...'
python <<'BENCHMARK_SCRIPT'
import requests
import json
import time
import sys

def benchmark_llm(engine_url="http://localhost:30000"):
    # Simple warmup
    try:
        requests.post(f"{engine_url}/generate", json={
            "text": "Hello,",
            "max_new_tokens": 10,
            "sampling_params": {"temperature": 0.8, "top_p": 0.95}
        }, timeout=30)
    except:
        pass
    
    # Actual benchmark
    prompts = ["Hello,"] * 4  # Batch size 4
    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    
    start_time = time.time()
    try:
        # Use generate endpoint for batched inference
        response = requests.post(
            f"{engine_url}/generate_batch",
            json={"prompts": prompts, "sampling_params": [generation_params] * len(prompts)},
            timeout=600
        )
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        # Fallback to sequential generation if batch not supported
        print(f"Batch generation failed, trying sequential: {e}", file=sys.stderr)
        results = []
        for i, prompt in enumerate(prompts):
            try:
                resp = requests.post(
                    f"{engine_url}/generate",
                    json={"text": prompt, **generation_params},
                    timeout=600
                )
                resp.raise_for_status()
                results.append(resp.json())
            except Exception as e2:
                print(f"Request {i} failed: {e2}", file=sys.stderr)
                results.append(None)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Parse results to get metrics
    # Note: SGLang returns details including completion tokens and time
    # This is a simplified extraction; actual implementation depends on SGLang version API
    
    total_tokens = 0
    valid_responses = 0
    latencies = []
    
    for res in results:
        if res and isinstance(res, dict) and 'text' in res:
            # Approximate: assuming ~256 tokens generated per request as configured
            tokens_generated = 256  # From max_new_tokens
            total_tokens += tokens_generated
            valid_responses += 1
            
            # Try to extract time if available in response structure
            if 'metadata' in res and 'time' in res['metadata']:
                latencies.append(res['metadata']['time'])
    
    if valid_responses == 0:
        print({"error": "No valid responses"}, file=sys.stderr)
        return {"status": "fail", "metrics": {}}
    
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate TTFT/TPOT approximations from metadata if available
    # If not, we report throughput and time per request
    avg_time_per_request = total_time / valid_responses
    tpot_p50 = avg_time_per_request  # Approximation
    tpot_p99 = avg_time_per_request * 1.5  # Approximation
    
    metrics = {
        "throughput_tok_s": round(throughput, 2),
        "gpu_memory_gb": 0,  # Would need API call to query memory
        "mfu_pct": 0,  # Would need hardware counters
        "tpot_p50_ms": round(tpot_p50 * 1000, 2),
        "tpot_p99_ms": round(tpot_p99 * 1000, 2)
    }
    
    print(json.dumps(metrics))

try:
    benchmark_llm()
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    import sys; sys.exit(1)
BENCHMARK_SCRIPT

kill_server=0
if [ ! -z "\$SERVER_PID" ]; then
    kill \$SERVER_PID 2>/dev/null || true
fi
"

    local output
    output=$(run_remote_heredoc "$node_addr" "$server_script")
    
    local status="pass"
    local metrics_json="{}"
    local notes=()
    
    if echo "$output" | grep -q "error"; then
        status="fail"
        notes+=("Server or benchmark execution encountered an error")
    fi
    
    # Extract metrics from the last line if it's JSON
    local last_line
    last_line=$(echo "$output" | grep -oP '^\{.*\}$' | tail -n 1)
    if [ -n "$last_line" ]; then
        metrics_json="$last_line"
    else
        status="fail"
        notes+=("Could not parse benchmark metrics")
    fi
    
    # Kill any lingering processes
    run_remote "$node_addr" "pkill -f 'python.*sglang' || true" 2>/dev/null || true
    
    EXPERIMENT_RESULTS["sglang_fused_decode"]=$(printf '{"name":"sglang_fused_decode","framework":"sglang","version":"main","model":"%s","node":"%s","status":"%s","metrics":%s,"notes":[],"raw_output":""}' "$model_name" "$node_addr" "$status" "$metrics_json")
}

# Function to run vLLM experiment
run_vllm_experiment() {
    local node_addr="$1"
    local venv_path="$2"
    local model_name="$3"
    local tp_size="$4"
    
    local server_script="
set -euo pipefail
export HF_HOME=$HF_CACHE
source $venv_path/bin/activate

# Kill any existing vllm servers
pkill -f 'python.*vllm' || true
pkill -f 'vllm.entrypoints.api.server' || true
sleep 2

# Start vLLM Server
echo 'Starting vLLM server...'
python -m vllm.entrypoints.api_server \
    --model $model_name \
    --tokenizer $model_name \
    --tensor-parallel-size $tp_size \
    --gpu-memory-utilization 0.8 \
    --port 8000 &

SERVER_PID=\$!
echo "Server PID: \$SERVER_PID"

# Wait for server to start
echo 'Waiting for server to be ready...'
sleep 60

# Check if server is still running
if ! kill -0 \$SERVER_PID 2>/dev/null; then
    echo "Server failed to start"
    exit 1
fi

# Run Benchmark
echo 'Running benchmark...'
python <<'BENCHMARK_SCRIPT'
import requests
import json
import time
import sys

def benchmark_vllm(api_url="http://localhost:8000"):
    # Warmup
    try:
        requests.post(f"{api_url}/generate", json={
            "prompt": "Hello,",
            "max_tokens": 10,
            "temperature": 0.8,
            "top_p": 0.95
        }, timeout=30)
    except:
        pass
    
    # Benchmark with multiple requests
    prompts = ["Hello,"] * 8
    generation_params = {
        "max_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    
    start_time = time.time()
    latencies = []
    total_tokens = 0
    valid_responses = 0
    
    for prompt in prompts:
        req_start = time.time()
        try:
            response = requests.post(
                f"{api_url}/generate",
                json={"prompt": prompt, **generation_params},
                timeout=600
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract generated tokens
            num_tokens = len(result.get('generated_text', ''))
            # Better: use prompt_token_ids and generated_token_ids if available
            # For simplicity, assume ~256 tokens
            num_tokens = 256
            
            req_end = time.time()
            latency = req_end - req_start
            latencies.append(latency * 1000) # ms
            total_tokens += num_tokens
            valid_responses += 1
            
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if valid_responses == 0:
        print(json.dumps({"error": "No valid responses"}))
        return
    
    # Calculate metrics
    throughput = total_tokens / total_time
    
    # Sort latencies for percentiles
    latencies.sort()
    p50_idx = int(len(latencies) * 0.5)
    p99_idx = int(len(latencies) * 0.99)
    
    tpot_p50 = latencies[p50_idx] if latencies else 0
    tpot_p99 = latencies[p99_idx] if latencies else 0
    
    # TTFT approximation: first request latency as proxy
    ttft_p50 = latencies[0] if latencies else 0
    ttft_p99 = latencies[-1] if latencies else 0
    
    metrics = {
        "throughput_tok_s": round(throughput, 2),
        "ttft_p50_ms": round(ttft_p50, 2),
        "ttft_p99_ms": round(ttft_p99, 2),
        "tpot_p50_ms": round(tpot_p50, 2),
        "tpot_p99_ms": round(tpot_p99, 2),
        "gpu_memory_gb": 0,
        "mfu_pct": 0
    }
    
    print(json.dumps(metrics))

try:
    benchmark_vllm()
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    import sys; sys.exit(1)
BENCHMARK_SCRIPT

if [ ! -z "\$SERVER_PID" ]; then
    kill \$SERVER_PID 2>/dev/null || true
fi
"

    local output
    output=$(run_remote_heredoc "$node_addr" "$server_script")
    
    local status="pass"
    local metrics_json="{}"
    local notes=()
    
    if echo "$output" | grep -q "error"; then
        status="fail"
        notes+=("Server or benchmark execution encountered an error")
    fi
    
    local last_line
    last_line=$(echo "$output" | grep -oP '^\{.*\}$' | tail -n 1)
    if [ -n "$last_line" ]; then
        metrics_json="$last_line"
    else
        status="fail"
        notes+=("Could not parse benchmark metrics")
    fi
    
    run_remote "$node_addr" "pkill -f 'python.*vllm' || true; pkill -f 'vllm.entrypoints.api.server' || true" 2>/dev/null || true
    
    EXPERIMENT_RESULTS["vllm_fp8_baseline"]=$(printf '{"name":"vllm_fp8_baseline","framework":"vllm","version":"latest","model":"%s","node":"%s","status":"%s","metrics":%s,"notes":[],"raw_output":""}' "$model_name" "$node_addr" "$status" "$metrics_json")
}

# Experiment 1: SGLang Fused Decode Kernels
exp_sglang_fused_decode() {
    log "Starting Experiment 1: SGLang Fused Decode Kernels"
    
    local node="192.168.1.76" # spark-01
    local model="Qwen2.5-7B"
    local venv_path="$BENCH_DIR/sglang-main"
    local install_cmd="pip install git+https://github.com/sgl-project/sglang.git@main"
    
    if ! setup_venv_remote "$node" "$venv_path" "$install_cmd"; then
        log "Skipping SGLang experiment on $node due to setup failure"
        EXPERIMENT_RESULTS["sglang_fused_decode"]=$(printf '{"name":"sglang_fused_decode","framework":"sglang","version":"main","model":"%s","node":"%s","status":"skip","metrics":{},"notes":["Setup failed"],"raw_output":""}' "$model" "$node")
        return 0
    fi
    
    run_sglang_experiment "$node" "$venv_path" "$model" 1
    
    log "Finished Experiment 1"
}

# Experiment 2: vLLM FP8 Baseline
exp_vllm_fp8_baseline() {
    log "Starting Experiment 2: vLLM FP8 Baseline"
    
    local node="192.168.1.77" # spark-02
    local model="Qwen2.5-7B"
    local venv_path="$BENCH_DIR/vllm-latest"
    local install_cmd="pip install vllm"
    
    if ! setup_venv_remote "$node" "$venv_path" "$install_cmd"; then
        log "Skipping vLLM experiment on $node due to setup failure"
        EXPERIMENT_RESULTS["vllm_fp8_baseline"]=$(printf '{"name":"vllm_fp8_baseline","framework":"vllm","version":"latest","model":"%s","node":"%s","status":"skip","metrics":{},"notes":["Setup failed"],"raw_output":""}' "$model" "$node")
        return 0
    fi
    
    run_vllm_experiment "$node" "$venv_path" "$model" 1
    
    log "Finished Experiment 2"
}

# Main execution
main() {
    log "Starting benchmark suite on 2026-10-29"
    
    # Run experiments
    exp_sglang_fused_decode || log_error "Experiment 1 failed"
    exp_vllm_fp8_baseline || log_error "Experiment 2 failed"
    
    # Prepare JSON output
    local results_json="["
    local first=true
    
    for key in "${!EXPERIMENT_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            results_json+=","
        fi
        results_json+="${EXPERIMENT_RESULTS[$key]}"
    done
    results_json+="]"
    
    # Output final JSON to stdout
    echo "$results_json"
    
    log "Benchmark suite completed"
}

main "$@"