#!/usr/bin/env python3
"""
run_experiment.py — Deploys a named experiment from queue.yaml to the DGX Spark
K8s cluster, waits for the serving endpoint to become healthy, runs benchmark.py,
then tears down the deployment.

Called by scheduler.py. Do not call directly unless debugging.

Per-experiment vLLM arg patches are defined in VLLM_PATCHES below.
The base manifest is selected based on the experiment's `framework:` field in queue.yaml:
  - framework: vllm   → pods-vllm.yaml  (patched in-memory and applied from /tmp)
  - framework: sglang → pods-sglang.yaml (applied as-is; no in-memory patching needed)
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


NAMESPACE          = "token-labs"
TOKEN_LABS         = Path.home() / "src" / "github.com" / "elizabetht" / "token-labs"
MANIFEST_VLLM      = TOKEN_LABS / "deploy" / "qwen3-coder-next" / "pods-vllm.yaml"
MANIFEST_SGLANG    = TOKEN_LABS / "deploy" / "qwen3-coder-next" / "pods-sglang.yaml"
STARTUP_TIMEOUT_S  = 90 * 60   # 90 min for model to load (GB10 sm_121 JIT-compiles CUDA kernels, ~40 safetensors shards)
SHUTDOWN_TIMEOUT_S = 5  * 60   # 5 min for teardown
BENCHMARK_TIMEOUT_S = 15 * 60  # 15 min per benchmark run

# Framework → manifest and pod label selector
FRAMEWORK_CONFIG = {
    "vllm":   {"manifest": MANIFEST_VLLM,   "label": "app=qwen3-coder-next-vllm",   "leader_pod": "qwen3-coder-next-vllm-leader"},
    "sglang": {"manifest": MANIFEST_SGLANG, "label": "app=qwen3-coder-next-sglang", "leader_pod": "qwen3-coder-next-sglang-leader"},
}

# ── Per-experiment vLLM arg patches ──────────────────────────────────────────
# Keys match experiment names. Each entry is a dict of flag→value to add/replace
# in the `vllm serve` command inside the leader pod's args.
# Special value None means "remove this flag".
VLLM_PATCHES = {
    "baseline-qwen3-coder-next-fp8-pp2": {},  # no changes

    "vllm-chunked-prefill-pp2": {
        "--enable-chunked-prefill": "",        # flag with no value
    },

    "vllm-higher-mem-utilization-pp2": {
        "--gpu-memory-utilization": "0.85",
    },

    "vllm-prefix-caching-pp2": {
        "--enable-prefix-caching": "",         # already in base, ensures it's there
    },

    "vllm-no-enforce-eager-pp2": {
        "--enforce-eager": None,               # remove this flag
    },

    "vllm-speculative-decoding-pp2": {
        # vLLM 0.18.0 replaced --speculative-model / --num-speculative-tokens
        # with a single --speculative-config JSON flag.
        "--speculative-config": '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}',
    },

    "vllm-longer-context-pp2": {
        "--max-model-len": "200000",
    },
}


# ── Manifest patching ─────────────────────────────────────────────────────────

def generate_manifest(exp: dict) -> str:
    """
    Load the correct manifest for this experiment's framework and patch args.
    For sglang experiments, returns the manifest as-is (no in-memory patching).
    For vllm experiments, patches the vllm serve command using VLLM_PATCHES.
    patches: {flag: value} — value="" for boolean flags, None to remove the flag.
    """
    name = exp["name"]
    framework = exp.get("framework", "vllm")
    fw_cfg = FRAMEWORK_CONFIG.get(framework, FRAMEWORK_CONFIG["vllm"])
    manifest_path = fw_cfg["manifest"]

    patches = VLLM_PATCHES.get(name, {})

    raw = manifest_path.read_text()
    if framework != "vllm" or not patches:
        return raw

    # Parse all YAML documents
    docs = list(yaml.safe_load_all(raw))

    for doc in docs:
        if doc is None or doc.get("kind") != "Pod":
            continue
        if doc.get("metadata", {}).get("name", "") != "qwen3-coder-next-vllm-leader":
            continue

        containers = doc["spec"]["containers"]
        for container in containers:
            if container.get("name") != "vllm":
                continue

            # args is a list with one element: the bash script string
            script = container["args"][0]

            # Find the vllm serve block and extract its flags
            # The command spans multiple lines joined with backslashes
            vllm_pattern = re.compile(
                r'(vllm serve [^\n\\]*(?:\\.[^\n\\]*)*)',
                re.DOTALL
            )
            m = vllm_pattern.search(script)
            if not m:
                print(f"[run_experiment] WARNING: could not find vllm serve in args", flush=True)
                continue

            cmd_block = m.group(1)
            # Flatten to single string for parsing
            flat = re.sub(r'\\\n\s*', ' ', cmd_block).strip()

            # Parse flags: split on ' --' boundaries
            # Keep 'vllm serve MODEL' as prefix
            parts = re.split(r'\s+(--\S+)', flat)
            # parts[0] = "vllm serve MODEL", parts[1::2] = flag names, parts[2::2] = values
            prefix = parts[0]  # "vllm serve Qwen/..."
            flags = {}
            i = 1
            while i < len(parts):
                flag = parts[i]
                # Next element is either a value (no leading --) or next flag
                val = ""
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    val = parts[i + 1].strip()
                    i += 2
                else:
                    i += 1
                flags[flag] = val

            # Apply patches
            for flag, value in patches.items():
                if value is None:
                    flags.pop(flag, None)
                else:
                    flags[flag] = value

            # Reconstruct the vllm serve command with consistent formatting
            indent = "        "  # 8 spaces to match original
            new_lines = [f"{indent}vllm serve {prefix.split()[-1]} \\"]
            flag_items = list(flags.items())
            for j, (flag, val) in enumerate(flag_items):
                is_last = (j == len(flag_items) - 1)
                cont = "" if is_last else " \\"
                if val:
                    new_lines.append(f"{indent}  {flag} {val}{cont}")
                else:
                    new_lines.append(f"{indent}  {flag}{cont}")

            new_cmd = "\n".join(new_lines)
            script = vllm_pattern.sub(new_cmd, script, count=1)
            container["args"][0] = script

    # Serialize back — use yaml.dump_all but preserve multi-doc structure
    return yaml.dump_all(docs, allow_unicode=True, default_flow_style=False)


# ── K8s helpers ───────────────────────────────────────────────────────────────

def kubectl(*args, check=True, capture=True) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "-n", NAMESPACE] + list(args)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def wait_for_healthy(endpoint: str, timeout_s: int) -> bool:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{endpoint}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def teardown_pods(label_selector: str):
    try:
        kubectl("delete", "pod", "-l", label_selector, "--grace-period=30", check=False)
        deadline = time.time() + SHUTDOWN_TIMEOUT_S
        while time.time() < deadline:
            result = kubectl("get", "pods", "-l", label_selector, "--no-headers", check=False)
            if not result.stdout.strip():
                return
            time.sleep(5)
        print("[run_experiment] WARNING: pods still present after teardown timeout", flush=True)
    except Exception as e:
        print(f"[run_experiment] teardown error: {e}", flush=True)


def apply_manifest_text(manifest_text: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest_text)
        tmp_path = f.name
    result = subprocess.run(
        ["kubectl", "apply", "-f", tmp_path],
        capture_output=True, text=True
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"[run_experiment] apply failed: {result.stderr}", flush=True)
        return False
    return True


def get_experiment(queue_path: Path, name: str) -> dict:
    data = yaml.safe_load(queue_path.read_text())
    for exp in data["experiments"]:
        if exp["name"] == name:
            return exp
    raise KeyError(f"Experiment {name!r} not found in {queue_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--endpoint", default="http://192.168.1.200:8000")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        ts = time.strftime("%H:%M:%S", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== run_experiment: {args.name} ===")

    try:
        exp = get_experiment(Path(args.queue), args.name)
    except KeyError as e:
        sys.exit(str(e))

    # ── 1. Resolve framework config and generate manifest ──────────────────
    framework = exp.get("framework", "vllm")
    fw_cfg = FRAMEWORK_CONFIG.get(framework, FRAMEWORK_CONFIG["vllm"])
    pod_label = fw_cfg["label"]
    log(f"Framework: {framework} | manifest: {fw_cfg['manifest'].name} | label: {pod_label}")
    patches = VLLM_PATCHES.get(args.name, {})
    log(f"vLLM patches for {args.name}: {patches or 'none (baseline or sglang)'}")
    manifest_text = generate_manifest(exp)

    # ── 2. Tear down existing pods (correct label for this framework) ──────
    log("Tearing down existing pods...")
    teardown_pods(pod_label)
    time.sleep(15)  # let GPU memory fully release

    # ── 3. Apply manifest ─────────────────────────────────────────────────
    log("Applying manifest...")
    if not apply_manifest_text(manifest_text):
        sys.exit("ERROR: manifest apply failed")

    # ── 4. Wait for endpoint ───────────────────────────────────────────────
    log(f"Waiting for {args.endpoint}/health (up to {STARTUP_TIMEOUT_S//60}m)...")
    if not wait_for_healthy(args.endpoint, STARTUP_TIMEOUT_S):
        log("ERROR: endpoint not healthy — tearing down")
        teardown_pods(pod_label)
        sys.exit(f"ERROR: {args.endpoint} not healthy after {STARTUP_TIMEOUT_S//60}m")

    log("Endpoint healthy. Waiting 30s for warmup...")
    time.sleep(30)

    # ── 5. Run benchmark ───────────────────────────────────────────────────
    model = exp.get("model", "Qwen/Qwen3-Coder-Next-FP8")
    log(f"Running benchmark (model={model})...")

    with open(log_path, "a") as logf:
        proc = subprocess.run(
            [sys.executable, args.benchmark,
             "--base-url", args.endpoint,
             "--model", model,
             "--output", args.output],
            timeout=BENCHMARK_TIMEOUT_S,
            stdout=logf,
            stderr=logf,
        )

    if proc.returncode != 0:
        log(f"WARNING: benchmark exited {proc.returncode}")

    # ── 6. Verify and report ───────────────────────────────────────────────
    out = Path(args.output)
    if not out.exists():
        sys.exit("ERROR: benchmark produced no output file")

    data = json.loads(out.read_text())
    primary = data.get("primary", {})
    log(f"Result: throughput={primary.get('throughput_tok_s')} tok/s | "
        f"TTFT_p50={primary.get('ttft_p50_ms')}ms | ITL_p50={primary.get('itl_p50_ms')}ms")
    log(f"=== Done: {args.name} ===")


if __name__ == "__main__":
    main()
