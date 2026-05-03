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
Integration tests for the OBJ storage backend.

Assumes the object store bucket already exists and is reachable.
Credentials are supplied via environment variables or pytest CLI options:

  OBJ_ENDPOINT   (or --obj-endpoint)   e.g. "minio.example.com:9000"
  OBJ_BUCKET     (or --obj-bucket)     e.g. "kv-cache"
  OBJ_ACCESS_KEY (or --obj-access-key) e.g. "minioadmin"
  OBJ_SECRET_KEY (or --obj-secret-key) e.g. "minioadmin"
  OBJ_SCHEME     (or --obj-scheme)     "http" or "https" (default: "http")
  OBJ_CA_BUNDLE  (or --obj-ca-bundle)  path to CA cert file (optional)

Run:
  pytest tests/test_obj_backend.py \
      --obj-endpoint minio:9000 \
      --obj-bucket kv-cache \
      --obj-access-key minioadmin \
      --obj-secret-key minioadmin
"""

import os
import time

import pytest
import torch

from llmd_fs_backend.file_mapper import FileMapper
from llmd_nixl.nixl_lookup import NixlLookup
from llmd_nixl.worker import NixlStorageOffloadingHandlers
from tests.test_fs_backend import roundtrip_once

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# pytest CLI options (registered in conftest.py)
# ---------------------------------------------------------------------------


def _get_param(request, cli_opt: str, env_var: str, default: str = "") -> str:
    """Return value from pytest CLI option, falling back to env var, then default."""
    val = request.config.getoption(cli_opt, default=None)
    if val:
        return val
    return os.environ.get(env_var, default)


@pytest.fixture(scope="session")
def obj_config(request):
    """Session-scoped fixture that collects OBJ connection parameters.
    Skips the entire session if required credentials are not provided."""
    endpoint = _get_param(request, "--obj-endpoint", "OBJ_ENDPOINT")
    bucket = _get_param(request, "--obj-bucket", "OBJ_BUCKET")
    access_key = _get_param(request, "--obj-access-key", "OBJ_ACCESS_KEY")
    secret_key = _get_param(request, "--obj-secret-key", "OBJ_SECRET_KEY")
    scheme = _get_param(request, "--obj-scheme", "OBJ_SCHEME", "http")
    ca_bundle = _get_param(request, "--obj-ca-bundle", "OBJ_CA_BUNDLE", "")
    if not endpoint or not bucket or not access_key or not secret_key:
        pytest.skip("OBJ endpoint, bucket, access_key and secret_key must be set")

    cfg = {
        "endpoint_override": endpoint,
        "bucket": bucket,
        "access_key": access_key,
        "secret_key": secret_key,
        "scheme": scheme,
        "ca_bundle": ca_bundle,
    }

    try:
        NixlLookup(cfg).exists("__connectivity_check__")
    except Exception as e:
        pytest.skip(f"Object store not reachable: {e}")

    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_blocks_per_file", [1, 2, 4])
@pytest.mark.parametrize("start_idx", [0, 3])
def test_obj_backend_roundtrip(
    gpu_blocks_per_file: int, start_idx: int, obj_config, default_vllm_config
):
    """End-to-end write/read roundtrip through the OBJ (S3/MinIO) backend.

    Writes num_blocks GPU blocks to S3, reads back a subset starting at
    start_idx, and verifies bit-exact equality. Covers gpu_blocks_per_file > 1
    to validate multi-block packing and partial reads.
    """
    num_layers = 80
    num_blocks = 8
    block_size = 16
    num_heads = 64
    head_size = 128
    dtype = torch.float16
    threads_per_gpu = 8
    gpu_block_size = 16

    file_mapper = FileMapper(
        root_dir=f"kv-test/{int(time.time())}/gpf{gpu_blocks_per_file}",
        model_name="test-model",
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=str(dtype),
    )

    roundtrip_once(
        file_mapper=file_mapper,
        dtype=dtype,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        gpu_block_size=gpu_block_size,
        num_heads=num_heads,
        head_size=head_size,
        write_block_ids=list(range(num_blocks)),
        read_block_ids=list(range(start_idx, num_blocks)),
        gpu_blocks_per_file=gpu_blocks_per_file,
        threads_per_gpu=threads_per_gpu,
        extra_config=obj_config,
        handlers_cls=NixlStorageOffloadingHandlers,
        wait_timeout=30.0,
        cleanup=False,
    )
