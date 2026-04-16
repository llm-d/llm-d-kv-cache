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

import os
import shutil
import tempfile

import pytest

from .utils import set_storage_log_level


def pytest_addoption(parser):
    """Register CLI options for fs_connector performance tests."""
    parser.addoption(
        "--storage-root",
        action="store",
        default=None,
        help=(
            "Root directory for fs_connector storage. Any local Linux path "
            "(e.g., /tmp, /mnt/pvc/kv-cache, /dev/shm). "
            "A unique subdirectory is created under this root per test. "
            "Default: $FS_CONNECTOR_STORAGE_ROOT env var, or system /tmp."
        ),
    )
    parser.addoption(
        "--storage-log-level",
        action="store",
        default=None,
        choices=("trace", "debug", "info", "warn", "error"),
        help=(
            "Set STORAGE_LOG_LEVEL for the fs_connector (trace/debug/info/"
            "warn/error). Applied before the LLM is built."
        ),
    )
    parser.addoption(
        "--debug-storage",
        action="store_true",
        default=False,
        help="Shortcut for --storage-log-level=debug.",
    )


@pytest.fixture(autouse=True)
def _apply_storage_log_level(request):
    """
    Apply --storage-log-level / --debug-storage to STORAGE_LOG_LEVEL before
    any fs_connector code runs. autouse so all tests pick it up.
    """
    level = request.config.getoption("--storage-log-level")
    if request.config.getoption("--debug-storage"):
        level = "debug"
    if level:
        set_storage_log_level(level)


@pytest.fixture
def storage_root(request):
    """
    Resolve the storage root directory for fs_connector tests.

    Precedence:
      1. --storage-root CLI flag (if set)
      2. FS_CONNECTOR_STORAGE_ROOT env var (if set)
      3. System /tmp (default)
    """
    root = request.config.getoption("--storage-root")
    if root is None:
        root = os.environ.get("FS_CONNECTOR_STORAGE_ROOT")
    if root is None:
        root = tempfile.gettempdir()  # typically /tmp
    os.makedirs(root, exist_ok=True)
    return root


@pytest.fixture
def storage_path(storage_root):
    """
    Provide a unique temporary storage subdirectory for each test.

    Created under `storage_root` and cleaned up after the test.
    """
    tmpdir = tempfile.mkdtemp(prefix="fs_connector_perf_", dir=storage_root)
    yield tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def fs_connector_config(storage_path):
    """Base fs_connector configuration for tests."""
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SharedStorageOffloadingSpec",
            "spec_module_path": "llmd_fs_backend.spec",
            "shared_storage_path": storage_path,
            "threads_per_gpu": 64,
            "block_size": 256,
            "max_staging_memory_gb": 150,
            "gds_mode": "disabled",
            "read_preferring_ratio": 0.75,
        },
    }
