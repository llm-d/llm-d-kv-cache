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
Shared helpers for the fs_connector performance/stress tests:
prompt construction, backend KVTransferConfig builder, KV-throughput math,
storage log-level setup, and LLM/storage cleanup.
"""

import gc
import logging
import os
import shutil

from vllm import TokensPrompt
from vllm.config import KVTransferConfig

# Valid STORAGE_LOG_LEVEL values (applies to both C++ and Python sides).
LOG_LEVELS = ("trace", "debug", "info", "warn", "error")


def set_storage_log_level(level: str) -> None:
    """
    Set the fs_connector storage log level via STORAGE_LOG_LEVEL env var.
    Must be called BEFORE LLM() / connector import so the C++ side reads it.
    """
    level = level.lower()
    if level not in LOG_LEVELS:
        raise ValueError(f"Unknown log level '{level}'. Expected one of {LOG_LEVELS}.")
    os.environ["STORAGE_LOG_LEVEL"] = level
    print(f"[INFO] STORAGE_LOG_LEVEL={level}")
    # Legacy flag: some code paths still check STORAGE_CONNECTOR_DEBUG.
    if level in ("trace", "debug"):
        os.environ["STORAGE_CONNECTOR_DEBUG"] = "1"


# Supported backend choices for parameterized tests / CLI runs.
BACKENDS = ("base", "gpu", "cpu", "storage", "multi-cpu-storage")

# Storage implementation flavors (only used by "storage" / "multi-cpu-storage").
STORAGE_TYPES = ("fs", "gds")

# Per-token KV cache size (bytes) for known models. Used to compute GB/s.
# Formula: layers × kv_heads × head_dim × 2 (K+V) × dtype_bytes (bf16 = 2)
KV_BYTES_PER_TOKEN = {
    "Qwen/Qwen3-0.6B": 28 * 8 * 128 * 2 * 2,  # 114,688
    "Qwen/Qwen2.5-1.5B-Instruct": 28 * 2 * 128 * 2 * 2,  # 28,672
    "Qwen/Qwen2.5-3B-Instruct": 36 * 2 * 128 * 2 * 2,  # 36,864
    "Qwen/Qwen2.5-7B-Instruct": 28 * 4 * 128 * 2 * 2,  # 57,344
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 32 * 8 * 128 * 2 * 2,  # 131,072
    "meta-llama/Llama-3.2-1B-Instruct": 16 * 8 * 64 * 2 * 2,  # 32,768
    "meta-llama/Meta-Llama-3.1-70B": 80 * 8 * 128 * 2 * 2,  # 327,680
}


def build_tokens_prompt(
    num_tokens: int, prefix_id: int = 1, fill_id: int = 2
) -> TokensPrompt:
    """
    Build a TokensPrompt directly from token IDs without using a tokenizer.

    Args:
        num_tokens: Total number of tokens in the prompt.
        prefix_id: Token ID for the first token (default: 1).
        fill_id:   Token ID used for the rest of the prompt (default: 2).

    Returns:
        TokensPrompt with the specified token IDs.
    """
    prompt_token_ids = [prefix_id] + [fill_id] * (num_tokens - 1)
    return TokensPrompt(prompt_token_ids=prompt_token_ids)


def get_kv_transfer_config(
    backend: str,
    storage_path: str | None = None,
    cpu_bytes_to_use: int = 4 << 30,  # 4 GiB CPU cache
    storage_block_size: int = 256,
    cpu_block_size: int = 48,
    threads_per_gpu: int = 24,
    max_staging_memory_gb: int = 150,
    storage_type: str = "fs",
) -> KVTransferConfig | None:
    """
    Build a KVTransferConfig for the specified backend.

    Args:
        backend: One of BACKENDS:
                   - "base":              No offloading at all
                   - "gpu":               GPU prefix cache only
                   - "cpu":               CPU offloading only
                   - "storage":           Storage offloading only
                   - "multi-cpu-storage": MultiConnector chaining CPU + storage
        storage_path: Filesystem path for "storage" / "multi-cpu-storage".
        cpu_bytes_to_use: CPU cache size in bytes for "cpu" / "multi-cpu-storage".
        storage_block_size: Block size for the storage tier.
        cpu_block_size: Block size for cpu backend.
        threads_per_gpu: I/O worker count for the storage tier.
        max_staging_memory_gb: Max CPU staging buffer for the storage tier.
        storage_type: Storage implementation flavor — "fs" (default, CPU
            staging) or "gds" (full read/write GPUDirect Storage).

    Returns:
        A KVTransferConfig, or None if backend is "base" or "gpu".

    Raises:
        ValueError: on an unknown backend.
    """
    # "base" (no offload, no prefix cache) and "gpu" (no offload, GPU prefix
    # cache enabled in the caller) both skip the KVTransferConfig entirely.
    if backend in ("base", "gpu"):
        return None

    gds_mode = "read_write" if storage_type == "gds" else "disabled"

    if backend == "storage":
        if storage_path is None:
            raise ValueError("backend='storage' requires storage_path")
        return KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SharedStorageOffloadingSpec",
                "spec_module_path": "llmd_fs_backend.spec",
                "shared_storage_path": storage_path,
                "threads_per_gpu": threads_per_gpu,
                "block_size": storage_block_size,
                "max_staging_memory_gb": max_staging_memory_gb,
                "gds_mode": gds_mode,
                "read_preferring_ratio": 0.75,
            },
        )

    if backend == "cpu":
        return KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": cpu_bytes_to_use,
                "block_size": cpu_block_size,
            },
        )

    if backend == "multi-cpu-storage":
        if storage_path is None:
            raise ValueError("backend='multi-cpu-storage' requires storage_path")
        return KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "cpu_bytes_to_use": cpu_bytes_to_use,
                            "block_size": cpu_block_size,
                        },
                    },
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "spec_name": "SharedStorageOffloadingSpec",
                            "spec_module_path": "llmd_fs_backend.spec",
                            "shared_storage_path": storage_path,
                            "threads_per_gpu": threads_per_gpu,
                            "block_size": storage_block_size,
                            "max_staging_memory_gb": max_staging_memory_gb,
                            "gds_mode": gds_mode,
                        },
                    },
                ],
            },
        )

    raise ValueError(f"Unknown backend '{backend}'. Expected one of {BACKENDS}.")


def should_enable_prefix_caching(backend: str, with_gpu_prefix_cache: bool) -> bool:
    """
    Decide whether vLLM's GPU prefix caching should be enabled.

    - "gpu"  -> always True  (the whole point of this baseline)
    - "base" -> always False (no caching at all, by definition)
    - offload backends (cpu/storage/multi-cpu-storage) -> opt-in via flag
      (useful to benchmark GPU-cache + offload together, e.g. to check
       whether the connector steals too much GPU memory from the cache).
    """
    if backend == "gpu":
        return True
    if backend == "base":
        return False
    return with_gpu_prefix_cache


def calculate_throughput_gb_s(
    model_name: str, num_tokens: int, avg_time: float
) -> float:
    """
    Compute KV-cache throughput in GB/s.

    Throughput = (num_tokens × kv_bytes_per_token) / avg_time.
    For unknown models returns 0.0 (caller should print NA).

    Args:
        model_name: HF model identifier.
        num_tokens: Tokens transferred per request.
        avg_time:   Average time per request (seconds).

    Returns:
        Throughput in GiB/s, or 0.0 if the model is unknown.
    """
    if avg_time <= 0:
        return 0.0
    bytes_per_token = KV_BYTES_PER_TOKEN.get(model_name, 0)
    if bytes_per_token == 0:
        return 0.0
    total_bytes = num_tokens * bytes_per_token
    return (total_bytes / avg_time) / (1 << 30)  # GiB/s


def del_llm_and_cleanup(llm) -> None:
    """
    Release the LLM, collect garbage, empty CUDA cache and tear down the
    torch.distributed process group if initialized.

    Mirrors vllm-offloading-tests/tests/test_utils.py:del_llm_and_cleanup so
    multiple LLM runs in the same process don't leak GPU memory.
    """
    try:
        del llm
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
                print("[INFO] torch.distributed process group destroyed.")
        except ImportError:
            pass
    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")


def cleanup_storage_dir(path: str | None) -> None:
    """Remove a storage directory if it exists (CLI cleanup helper)."""
    if not path:
        return
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"[CLEANUP] Finish Removed {path}")
        except Exception as e:
            print(f"[CLEANUP] Failed to remove {path}: {e}")


def quiet_vllm_logs() -> None:
    """
    Silence vLLM's INFO-level inference-loop logs so per-iteration test
    output stays readable. Model-load / init logs still show.
    """
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("vllm.engine").setLevel(logging.WARNING)


def warmup_req(llm, sampling_params) -> None:
    """
    Run a tiny 10-token prompt so model/compile paths are warm but the
    offload backend is NOT populated with any of the test prompts.
    Uses distinct prefix/fill token IDs (999 / 998) to avoid any hash
    overlap with the real test prompts.
    """
    warmup_prompt = build_tokens_prompt(10, prefix_id=999, fill_id=998)
    print("[INFO] Warming up model (10-token prompt)...")
    llm.generate([warmup_prompt], sampling_params, use_tqdm=False)
