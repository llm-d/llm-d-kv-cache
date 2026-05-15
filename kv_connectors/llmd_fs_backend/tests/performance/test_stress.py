# Copyright 2025 The llm-d Authors.
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
Stress test: batched KV-offload load with a hot/cold ratio sweep.

Each generate() call submits `batch_size` prompts mixed by `hot_ratio`:
  - hot : reused from the pool of already-seen prompts (READ / reload path)
  - cold: freshly generated unique prompts             (WRITE / populate path)

    ratio = 0.0 -> all cold  (pure write stress)
    ratio = 1.0 -> all hot   (pure read stress)
    in between  -> mix; exercises the scheduler's read_preferring_ratio.

The default sweep walks `hot_ratios` = [0.0, 0.1, ..., 1.0] with
`num_iterations` iters per ratio. Pass `--num-repeats=N>=2` to repeat the
full sweep against a growing working set (reveals drift / fragmentation /
page-cache pressure).

Default budget: batch=32, 10K tokens, 11 ratios, 5 iters, 1 repeat
  = 55 iterations x 32 prompts = 1,760 prompts total.

Metrics per (repeat, ratio): batch mean / p50 / p99, tokens/s, KV GB/s.

Backends (--backend CLI / @pytest.parametrize):
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
import random
import statistics
import time

import pytest
from vllm import LLM, SamplingParams, TokensPrompt

from .utils import (
    BACKENDS,
    LOG_LEVELS,
    STORAGE_TYPES,
    calculate_throughput_gb_s,
    cleanup_storage_dir,
    del_llm_and_cleanup,
    get_kv_transfer_config,
    quiet_vllm_logs,
    set_storage_log_level,
    should_enable_prefix_caching,
    warmup_req,
)

# Default sweep: 0.0, 0.1, 0.2, ..., 1.0 (11 values), 0.1 granularity reveals
# latency plateaus/cliffs as the scheduler shifts from all-writes to all-reads.
DEFAULT_HOT_RATIOS: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))


def _build_mixed_batch(
    pool: list[TokensPrompt],
    next_prefix_id: int,
    batch_size: int,
    num_tokens: int,
    hot_ratio: float,
    rng: random.Random,
) -> tuple[list[TokensPrompt], int]:
    """
    Build a batch of `batch_size` prompts mixing `hot_ratio` reused prompts
    from `pool` with fresh unique prompts (appended to `pool` for later reuse).
    If the pool is smaller than the requested hot count, the shortfall is
    filled with additional cold prompts.

    Returns (batch, next_prefix_id).
    """
    n_hot = round(batch_size * hot_ratio)
    n_hot = min(n_hot, len(pool))
    n_cold = batch_size - n_hot

    batch: list[TokensPrompt] = list(rng.sample(pool, n_hot)) if n_hot else []
    for _ in range(n_cold):
        # Unique prefix_id -> unique KV block hashes -> no FS overlap with pool.
        prompt_ids = [next_prefix_id] + [2] * (num_tokens - 1)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        pool.append(prompt)
        batch.append(prompt)
        next_prefix_id += 1

    return batch, next_prefix_id


def _percentile(values: list[float], pct: float) -> float:
    """Simple nearest-rank percentile (sorted index)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct), len(s) - 1)
    return s[idx]


def run_stress_test(
    model_name: str,
    backend: str,
    gpu_memory_utilization: float,
    storage_path: str,
    batch_size: int = 32,
    num_iterations: int = 5,
    num_tokens: int = 10000,
    hot_ratios: list[float] | None = None,
    num_repeats: int = 1,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    storage_log_level: str | None = None,
    gpu_block_size: int = 16,
    storage_block_size: int = 256,
    cpu_block_size: int = 48,
    threads_per_gpu: int = 24,
    with_gpu_prefix_cache: bool = False,
    storage_type: str = "fs",
) -> tuple[float, float, float, float, float, float]:
    """
    Run a hot/cold-ratio sweep stress test (see module docstring).

    Args:
        batch_size: Number of prompts per generate() call.
        num_iterations: Iterations per (repeat, ratio) combination.
        num_tokens: Tokens per prompt.
        hot_ratios: List of fractions in [0.0, 1.0] — fraction of each batch
            sampled from the already-seen pool. Default: 0.0..1.0 step 0.1.
        num_repeats: Number of full sweeps through `hot_ratios`. Multiple
            repeats grow the working set and reveal drift.

    Returns (for pytest assertions — computed from the final repeat at the
    maximum ratio in `hot_ratios`, i.e. the steady-state pure-read phase):
        (tokens_per_sec, kv_gb_s, batch_mean, batch_p50, batch_p99, batch_p100)
    """
    if hot_ratios is None:
        hot_ratios = list(DEFAULT_HOT_RATIOS)

    total_iters = num_repeats * len(hot_ratios) * num_iterations
    print(f"\n===== Stress Test: backend={backend} model={model_name} =====")
    print(
        f"Batch size: {batch_size}, Tokens/prompt: {num_tokens}\n"
        f"Sweep: hot_ratios={hot_ratios}\n"
        f"Repeats: {num_repeats}, Iterations/ratio: {num_iterations} "
        f"-> Total iterations: {total_iters} "
        f"(= {total_iters * batch_size} prompts)"
    )

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

    warmup_req(llm, sampling_params)

    # -------- Phase 2: Run batch --------
    rng = random.Random(seed)
    pool: list[TokensPrompt] = []
    next_prefix_id = 10  # kept distinct from warmup prefix 999

    # results[(repeat, ratio)] = list[latency_sec]
    results: dict = {}
    total_prompts = 0
    t_start = time.perf_counter()

    for repeat in range(num_repeats):
        for ratio in hot_ratios:
            key = (repeat, ratio)
            results[key] = []
            print(
                f"\n[sweep {repeat + 1}/{num_repeats}] ratio={ratio:.2f}  "
                f"pool_size={len(pool)}"
            )
            for i in range(num_iterations):
                batch, next_prefix_id = _build_mixed_batch(
                    pool, next_prefix_id, batch_size, num_tokens, ratio, rng
                )
                t0 = time.perf_counter()
                outputs = llm.generate(batch, sampling_params, use_tqdm=False)
                dt = time.perf_counter() - t0
                results[key].append(dt)
                total_prompts += len(outputs)
                print(
                    f"  iter {i + 1:2d}/{num_iterations}: {dt:.3f}s "
                    f"({batch_size / dt:.2f} req/s)"
                )

    t_total = time.perf_counter() - t_start

    # -------- Phase 3: Print results --------
    print("\n==================== Per-ratio summary ====================")
    header = (
        f"{'ratio':>5}  "
        + "  ".join(
            f"{'rep' + str(r + 1) + ' mean':>10}  {'p50':>7}  {'p99':>7}  "
            f"{'KV GB/s':>8}"
            for r in range(num_repeats)
        )
        + (f"  {'Δmean':>7}" if num_repeats >= 2 else "")
    )
    print(header)
    print("-" * len(header))

    # We use pure-ratio=1.0 of the last repeat as the "steady" representative
    # to return for pytest assertions (falls back to last ratio if 1.0 absent).
    steady_ratio = 1.0 if 1.0 in hot_ratios else hot_ratios[-1]
    steady_lats = results[(num_repeats - 1, steady_ratio)]

    for ratio in hot_ratios:
        row = f"{ratio:>5.2f}  "
        means = []
        for r in range(num_repeats):
            lats = results[(r, ratio)]
            mean = statistics.mean(lats)
            means.append(mean)
            p50 = statistics.median(lats)
            p99 = _percentile(lats, 0.99)
            kv = calculate_throughput_gb_s(model_name, batch_size * num_tokens, mean)
            row += f"  {mean:>10.3f}  {p50:>7.3f}  {p99:>7.3f}  {kv:>8.2f}"
        if num_repeats >= 2 and means[0] > 0:
            drift = (means[-1] - means[0]) / means[0] * 100.0
            row += f"  {drift:>+6.1f}%"
        print(row)

    # Steady-state metrics (return value + key line below the table).
    steady_mean = statistics.mean(steady_lats) if steady_lats else 0.0
    steady_p50 = statistics.median(steady_lats) if steady_lats else 0.0
    steady_p99 = _percentile(steady_lats, 0.99)
    steady_p100 = max(steady_lats) if steady_lats else 0.0
    tokens_per_sec = (batch_size * num_tokens) / steady_mean if steady_mean > 0 else 0.0
    kv_gb_s = calculate_throughput_gb_s(
        model_name, batch_size * num_tokens, steady_mean
    )

    print(f"\n[RESULTS] backend={backend}  model={model_name}")
    print(
        f"  Total prompts: {total_prompts}, wall time: {t_total:.2f}s, "
        f"pool_size={len(pool)}"
    )
    print(
        f"  Steady-state (repeat {num_repeats}, ratio={steady_ratio}): "
        f"tokens/s={tokens_per_sec:,.0f}  "
        + (f"KV={kv_gb_s:.2f} GB/s  " if kv_gb_s > 0 else "KV=NA  ")
        + f"mean={steady_mean:.3f}s p50={steady_p50:.3f}s "
        f"p99={steady_p99:.3f}s p100={steady_p100:.3f}s"
    )

    del_llm_and_cleanup(llm)
    return (
        tokens_per_sec,
        kv_gb_s,
        steady_mean,
        steady_p50,
        steady_p99,
        steady_p100,
    )


# ---------------- pytest entry points ----------------


@pytest.mark.parametrize("backend", ["storage"])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-7B-Instruct"])
def test_stress(
    storage_path: str,
    backend: str,
    batch_size: int,
    model_name: str,
):
    """
    Hot/cold-ratio sweep stress for the selected backend.

    Pytest runs a short single-repeat sweep (hot_ratios=[0.0, 0.5, 1.0],
    2 iters each) to keep CI fast. For a full stress run, use the CLI.

    Verifies:
      - Throughput > 0 (no deadlocks in concurrent I/O)
      - Steady-state p99 latency bounded (backend handles pure-reads well)
      - p100 not a wild outlier (no tail pathology)
    """
    _tps, _gbs, _mean, _p50, p99, p100 = run_stress_test(
        model_name=model_name,
        backend=backend,
        gpu_memory_utilization=0.5,
        storage_path=storage_path,
        batch_size=batch_size,
        num_iterations=2,
        num_tokens=10000,
        hot_ratios=[0.0, 0.5, 1.0],
        num_repeats=1,
    )

    assert p99 < 10.0, (
        f"Steady-state p99 latency too high: {p99:.3f}s "
        "(possible backend contention or I/O queue starvation)."
    )
    assert (p100 - p99) < 5.0, (
        f"Tail latency spike (p100 - p99 = {p100 - p99:.3f}s). "
        "Possible GC pause, lock contention, or I/O stall."
    )


# ---------------- standalone CLI ----------------

if __name__ == "__main__":
    # Example manual runs (see --help for the full flag list):
    #   # Storage tier - default fs (use CPU staging)
    #   python -m tests.performance.test_stress --backend=storage
    #   # CPU-only baseline
    #   python -m tests.performance.test_stress --backend=cpu
    parser = argparse.ArgumentParser(
        description=(
            "Run KV-offload batched stress test with hot/cold-ratio sweep. "
            "By default sweeps hot_ratio from 0.0 to 1.0 in 0.1 steps "
            "(single pass) to stress both write and read paths. Pass "
            "--num-repeats=2 (or more) to detect drift across repeated sweeps."
        )
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
        default="/tmp/fs_connector_stress",
        help="Storage path for storage / multi-cpu-storage tiers",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Iterations per (repeat, hot_ratio) combination (default 5)",
    )
    parser.add_argument("--num-tokens", type=int, default=10000)
    parser.add_argument(
        "--hot-ratios",
        type=str,
        default=",".join(str(r) for r in DEFAULT_HOT_RATIOS),
        help=(
            "Comma-separated fractions in [0.0, 1.0] defining the sweep. "
            "hot_ratio=1.0 -> all prompts reused (pure read); "
            "0.0 -> all fresh (pure write). "
            f"Default: {','.join(str(r) for r in DEFAULT_HOT_RATIOS)}"
        ),
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help=(
            "Number of full sweeps through hot_ratios (default 1). "
            "Use >=2 to catch drift / FS fragmentation across repeated passes."
        ),
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.5)
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
        default=24,
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

    hot_ratios = [float(x.strip()) for x in args.hot_ratios.split(",") if x.strip()]
    for r in hot_ratios:
        if not 0.0 <= r <= 1.0:
            raise SystemExit(f"hot_ratio {r} out of [0.0, 1.0]")

    try:
        run_stress_test(
            model_name=args.model,
            backend=args.backend,
            gpu_memory_utilization=args.gpu_mem_util,
            storage_path=args.storage_path,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_tokens=args.num_tokens,
            hot_ratios=hot_ratios,
            num_repeats=args.num_repeats,
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
