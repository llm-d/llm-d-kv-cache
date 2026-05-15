# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performance test: KV-offloading backend sustained throughput
(cold -> hot -> steady-state), mirroring vllm-offloading-tests/test_cold_hot_req.py.

Measures:
  - Cold latency (req 1): FS / CPU populate + full prefill
  - Hot average    (req 2..N): reload from offload backend
  - Last-10 avg:   steady-state latency
  - Throughput (GB/s): KV-cache bytes / avg_time for the chosen model

Backends (configurable via --backend CLI / @pytest.parametrize):
  - "base":              No offloading at all
  - "gpu":               GPU prefix cache only
  - "cpu":               CPU offloading only
  - "storage":           Storage offloading only
  - "multi-cpu-storage": MultiConnector chaining CPU + storage

Storage implementation flavor (for "storage" / "multi-cpu-storage"):
  - --storage-type fs (default, CPU staging)
  - --storage-type gds (full read/write GPUDirect Storage)
"""

import argparse
import time

import pytest
from vllm import LLM, SamplingParams

from .utils import (
    BACKENDS,
    LOG_LEVELS,
    STORAGE_TYPES,
    build_tokens_prompt,
    calculate_throughput_gb_s,
    cleanup_storage_dir,
    del_llm_and_cleanup,
    get_kv_transfer_config,
    quiet_vllm_logs,
    set_storage_log_level,
    should_enable_prefix_caching,
    warmup_req,
)


def run_throughput_test(
    model_name: str,
    backend: str,
    gpu_memory_utilization: float,
    storage_path: str,
    num_requests: int = 40,
    num_tokens: int = 10000,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    storage_log_level: str | None = None,
    gpu_block_size: int = 16,
    storage_block_size: int = 256,
    cpu_block_size: int = 48,
    threads_per_gpu: int = 24,
    with_gpu_prefix_cache: bool = False,
    storage_type: str = "fs",
) -> tuple[float, float, float, float, float]:
    """
    Run sustained throughput test: cold -> hot -> steady-state.

    Returns:
        (cold_time, hot_avg, last_10_avg, total_time, throughput_gb_s)
    """
    print(f"\n===== Throughput Test: backend={backend} model={model_name} =====")
    print(f"Requests: {num_requests}, Tokens/req: {num_tokens}")

    # -------- Phase 1: Setup (config, LLM, warmup) --------
    if storage_log_level:
        set_storage_log_level(storage_log_level)

    kv_transfer_config = get_kv_transfer_config(
        backend=backend,
        storage_path=storage_path,
        storage_block_size=storage_block_size,
        cpu_block_size=cpu_block_size,
        threads_per_gpu=threads_per_gpu,
        storage_type=storage_type,
    )

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_transfer_config=kv_transfer_config,
        max_model_len=num_tokens + 1000,
        seed=seed,
        # "gpu" always on, "base" always off, offload backends opt-in
        # via --with-gpu-prefix-cache. See should_enable_prefix_caching().
        enable_prefix_caching=should_enable_prefix_caching(
            backend, with_gpu_prefix_cache
        ),
        tensor_parallel_size=tensor_parallel_size,
        block_size=gpu_block_size,
    )

    quiet_vllm_logs()

    sampling_params = SamplingParams(
        detokenize=False,
        ignore_eos=True,
        seed=seed,
        max_tokens=1,
    )

    prompt = build_tokens_prompt(num_tokens)

    warmup_req(llm, sampling_params)

    # -------- Phase 2: Run cold -> hot test --------
    times = []
    print(f"[INFO] Running {num_requests} requests...")
    for i in range(num_requests):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [{i + 1:3d}] {dt:.3f}s")

    # -------- Phase 3: Print results --------
    cold_time = times[0]
    hot_avg = sum(times[1:]) / (len(times) - 1) if len(times) > 1 else 0.0
    total_time = sum(times)
    last_10 = times[-10:] if len(times) >= 10 else times
    last_10_avg = sum(last_10) / len(last_10)

    # Throughput (GB/s) based on KV bytes per token of the chosen model.
    # Computed against the steady-state avg (last-10) for a representative value.
    throughput_gb_s = calculate_throughput_gb_s(model_name, num_tokens, last_10_avg)

    # Print the summary block: cold / hot / steady-state latencies + KV GB/s.
    input_tokens = len(outputs[0].prompt_token_ids)
    print(f"\n[RESULTS] backend={backend}  model={model_name}")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Cold (req 1):          {cold_time:.3f}s")
    print(f"  Hot avg (req 2..{num_requests}):   {hot_avg:.3f}s")
    print(f"  Last-10 avg:           {last_10_avg:.3f}s")
    print(f"  Total ({num_requests} reqs):       {total_time:.3f}s")
    if throughput_gb_s > 0:
        print(f"  Throughput (KV):       {throughput_gb_s:.2f} GB/s")
    else:
        print(f"  Throughput (KV):       NA (unknown model '{model_name}')")

    # Warn if steady-state (last-10) is >10% slower than hot_avg — signals
    # possible memory leak, FS fragmentation, or page-cache churn.
    degradation_pct = (last_10_avg - hot_avg) / hot_avg * 100 if hot_avg > 0 else 0
    if degradation_pct > 10:
        print(f"  [WARN] Steady-state degradation: {degradation_pct:.1f}%")

    del_llm_and_cleanup(llm)
    return cold_time, hot_avg, last_10_avg, total_time, throughput_gb_s


# ---------------- pytest entry points ----------------


# Default parameterization: storage tier only (primary target of these tests).
# Run other backends by passing --backend via CLI (see __main__) or by
# extending the parametrize list below.
@pytest.mark.parametrize("backend", ["storage"])
@pytest.mark.parametrize("num_requests", [40])
# (model_name, num_tokens) pairs - num_tokens sized to each model's max context.
@pytest.mark.parametrize(
    "model_name,num_tokens",
    [
        ("Qwen/Qwen3-0.6B", 31000),  # ~32K ctx
        ("meta-llama/Llama-3.2-1B-Instruct", 31000),  # 128K ctx (keep small for speed)
        ("Qwen/Qwen2.5-7B-Instruct", 31000),  # 32K ctx
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 128000),  # 128K ctx
        ("openai/gpt-oss-20b", 31000),  # 128K ctx (keep small for speed)
        ("openai/gpt-oss-120b", 31000),  # 128K ctx
        ("meta-llama/Meta-Llama-3.1-70B", 31000),  # 128K ctx (multi-GPU)
    ],
    ids=[
        "qwen3-0.6b",
        "llama-3.2-1b",
        "qwen2.5-7b",
        "llama-3.1-8b",
        "gpt-oss-20b",
        "gpt-oss-120b",
        "llama-3.1-70b",
    ],
)
def test_throughput(
    storage_path: str,
    backend: str,
    model_name: str,
    num_requests: int,
    num_tokens: int,
):
    """
    Sustained throughput test for the selected backend.

    Verifies:
      - Hot (reload) is faster than cold (populate)
      - Steady-state does not degrade by >20% vs hot avg
    """
    cold, hot_avg, last_10_avg, _total, _tput = run_throughput_test(
        model_name=model_name,
        backend=backend,
        gpu_memory_utilization=0.5,
        storage_path=storage_path,
        num_requests=num_requests,
        num_tokens=num_tokens,
    )

    assert hot_avg < cold, (
        f"Hot latency ({hot_avg:.3f}s) should be faster than cold ({cold:.3f}s). "
        "Offload reload path did not help vs cold populate."
    )

    degradation = (last_10_avg - hot_avg) / hot_avg if hot_avg > 0 else 0
    assert degradation < 0.20, (
        f"Steady-state degradation too high: {degradation * 100:.1f}% "
        "(possible memory leak or fragmentation)."
    )


# ---------------- standalone CLI ----------------

if __name__ == "__main__":
    # Example manual runs (see --help for the full flag list):
    #   # Storage tier - default fs (use CPU staging)
    #   python -m tests.performance.test_throughput --backend=storage
    #   # CPU-only baseline
    #   python -m tests.performance.test_throughput --backend=cpu
    parser = argparse.ArgumentParser(
        description="Run KV-offload throughput test (cold/hot/steady)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="storage",
        choices=BACKENDS,
        help=f"Backend to benchmark (one of {BACKENDS})",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="/tmp/fs_connector_perf",
        help="Storage path for storage / multi-cpu-storage tiers",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--num-tokens", type=int, default=31000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--gpu-block-size",
        type=int,
        default=16,
        help="GPU KV cache block size (vLLM block_size)",
    )
    parser.add_argument(
        "--storage-block-size",
        type=int,
        default=256,
        help="Storage tier offloaded block_size (tokens per file)",
    )
    parser.add_argument(
        "--cpu-block-size",
        type=int,
        default=48,
        help="CPU offload block_size (for cpu / multi-cpu-storage tiers)",
    )
    parser.add_argument(
        "--threads-per-gpu",
        type=int,
        default=64,
        help="I/O worker threads per GPU for storage tier (default 24)",
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        default="fs",
        choices=STORAGE_TYPES,
        help=(
            f"Storage implementation flavor (one of {STORAGE_TYPES}). "
            "Only meaningful for --backend=storage / multi-cpu-storage."
        ),
    )
    parser.add_argument(
        "--with-gpu-prefix-cache",
        action="store_true",
        help=(
            "Enable GPU prefix cache alongside the offload backend. "
            "Ignored for 'gpu' (always on) and 'base' (always off)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=LOG_LEVELS,
        help=f"Set STORAGE_LOG_LEVEL for storage tier (one of {LOG_LEVELS})",
    )
    args = parser.parse_args()

    try:
        run_throughput_test(
            model_name=args.model,
            backend=args.backend,
            gpu_memory_utilization=args.gpu_mem_util,
            storage_path=args.storage_path,
            num_requests=args.num_requests,
            num_tokens=args.num_tokens,
            tensor_parallel_size=args.tp_size,
            storage_log_level=args.log_level,
            gpu_block_size=args.gpu_block_size,
            storage_block_size=args.storage_block_size,
            cpu_block_size=args.cpu_block_size,
            threads_per_gpu=args.threads_per_gpu,
            with_gpu_prefix_cache=args.with_gpu_prefix_cache,
            storage_type=args.storage_type,
        )
    finally:
        cleanup_storage_dir(args.storage_path)
