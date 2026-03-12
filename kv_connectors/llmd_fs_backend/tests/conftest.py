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

# tests/conftest.py
import gc
import time

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    """Skip all tests in this session if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def cuda_teardown():
    """Ensure CUDA and C++ thread-pool resources from one test are fully
    released before the next test starts. Without this, async destructors
    can cause 'cudaErrorUnknown' or stale file-open errors in subsequent tests.
    """
    yield
    gc.collect()  # force Python GC to call C++ destructors immediately
    torch.cuda.synchronize()  # surface any async CUDA errors in the right test
    torch.cuda.empty_cache()  # free cached allocations so next test starts clean
    time.sleep(0.5)  # allow C++ thread-pool shutdown to complete


@pytest.fixture(scope="function")
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    This matches vLLM's internal test fixture pattern.
    """
    # Use empty VllmConfig() which provides sensible defaults
    with set_current_vllm_config(VllmConfig()):
        yield
