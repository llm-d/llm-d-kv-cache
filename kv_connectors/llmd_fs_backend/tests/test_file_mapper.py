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

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_cuda_required

CONNECTOR_ROOT = Path(__file__).resolve().parents[1]


def load_file_mapper_class():
    module_path = CONNECTOR_ROOT / "llmd_fs_backend" / "file_mapper.py"
    spec = importlib.util.spec_from_file_location("file_mapper_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FileMapper


def test_file_mapper_masks_hashes_to_lower_64_bits():
    file_mapper = load_file_mapper_class()(
        root_dir="/tmp/kv-cache",
        model_name="test-model",
        gpu_block_size=16,
        gpu_blocks_per_file=16,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype="float16",
    )

    assert file_mapper.get_file_name((1 << 72) + 0x1234).endswith(
        "/000/00/0000000000001234.bin"
    )
    assert file_mapper.get_file_name(b"\x01\x02").endswith(
        "/000/00/0000000000000102.bin"
    )
