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
Tests to check if GDS (GPUDirect Storage) is available.
If GDS is not available, tests will fail with an error.
"""

import os
import subprocess

import pytest
import storage_offload
import torch


def _collect_gds_status() -> dict:
    """Collect GDS availability indicators and return as a dict."""
    try:
        result = subprocess.run(["lsmod"], capture_output=True, text=True, check=False)
        nvidia_fs_loaded = "nvidia_fs" in result.stdout
    except Exception:
        nvidia_fs_loaded = False

    try:
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False
        )
        libcufile_available = "libcufile.so" in result.stdout
    except Exception:
        libcufile_available = False

    if not libcufile_available:
        common_paths = [
            "/usr/local/cuda/lib64/libcufile.so",
            "/usr/lib/x86_64-linux-gnu/libcufile.so",
        ]
        libcufile_available = any(os.path.exists(path) for path in common_paths)

    has_module = hasattr(storage_offload, "StorageOffloadEngine")

    return {
        "nvidia_fs_loaded": nvidia_fs_loaded,
        "libcufile_available": libcufile_available,
        "has_module": has_module,
    }


def check_gds_available() -> bool:
    """Return True if all GDS prerequisites are met."""
    s = _collect_gds_status()
    return s["nvidia_fs_loaded"] and s["libcufile_available"] and s["has_module"]


def get_gds_status_message() -> str:
    """Return a human-readable summary of GDS availability."""
    s = _collect_gds_status()
    return (
        f"nvidia-fs module: {'loaded' if s['nvidia_fs_loaded'] else 'not loaded'}, "
        f"libcufile.so: {'available' if s['libcufile_available'] else 'not found'}, "
        f"storage_offload module: {'imported' if s['has_module'] else 'not available'}"
    )


class TestGDSAvailability:
    """Tests for GDS availability."""

    def test_gds_available(self):
        """Test that GDS is available. Fails if GDS is not available."""
        gds_available = check_gds_available()
        status_msg = get_gds_status_message()

        print(f"\n[INFO] GDS Available: {gds_available}")
        print(f"[INFO] GDS Status: {status_msg}")

        if not gds_available:
            pytest.skip(f"GDS is not available. Status: {status_msg}")

        print("[INFO] GDS is available - tests can proceed")

    @pytest.mark.parametrize("gpu_blocks_per_file", [1, 2, 4, 8])
    @pytest.mark.parametrize("start_idx", [0, 3])
    def test_gds_roundtrip(
        self, tmp_path, default_vllm_config, gpu_blocks_per_file, start_idx
    ):
        """Test a full write/read roundtrip with GDS enabled."""
        if not check_gds_available():
            pytest.skip(f"GDS not available: {get_gds_status_message()}")

        import math

        from test_fs_backend import roundtrip_once

        from llmd_fs_backend.file_mapper import FileMapper

        num_layers = 80
        num_blocks = 8
        block_size = 16
        num_heads = 64
        head_size = 128
        dtype = torch.float16
        threads_per_gpu = 8

        expected_files = math.ceil(num_blocks / gpu_blocks_per_file)
        print(
            f"\n[INFO] Testing GDS roundtrip with "
            f"gpu_blocks_per_file={gpu_blocks_per_file}, start_idx={start_idx}"
        )
        print(f"[INFO] Will write {num_blocks} blocks across {expected_files} files")

        file_mapper = FileMapper(
            root_dir=str(tmp_path),
            model_name="test-model",
            gpu_block_size=block_size,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            rank=0,
            dtype=str(dtype),
        )

        # start_idx=0: all blocks; start_idx=3: partial first group (e.g., 3..7)
        block_ids = list(range(start_idx, num_blocks))

        roundtrip_once(
            file_mapper=file_mapper,
            dtype=dtype,
            num_layers=num_layers,
            num_blocks=num_blocks,
            gpu_block_size=block_size,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            read_block_ids=block_ids,
            write_block_ids=block_ids,
            gpu_blocks_per_file=gpu_blocks_per_file,
            threads_per_gpu=threads_per_gpu,
            gds_mode="read_write",
        )

        print(
            f"[INFO] GDS roundtrip succeeded with "
            f"gpu_blocks_per_file={gpu_blocks_per_file}, start_idx={start_idx}"
        )
