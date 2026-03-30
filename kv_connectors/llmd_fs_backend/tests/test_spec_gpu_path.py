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

# tests/test_spec_gpu_path.py
"""
Tests for GPU SM version path isolation in SharedStorageOffloadingSpec.

When a hybrid model (mamba/linear-attention + full-attention) is used, the
mamba/SSM recurrent state is numerically incompatible across GPU SM
architectures.  SharedStorageOffloadingSpec embeds the SM version in the NFS
path so each GPU architecture maintains an isolated namespace on shared
storage.  These tests verify the path-building logic without requiring a full
vLLM engine.
"""

import pytest
from unittest.mock import MagicMock, patch

from vllm.v1.kv_offload.spec import OffloadingSpec

from llmd_fs_backend.spec import SharedStorageOffloadingSpec

FAKE_STORAGE = "/tmp/test-kv-spec"
FAKE_MODEL = "fake-org/fake-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(storage_path: str = FAKE_STORAGE) -> MagicMock:
    """Minimal VllmConfig mock — only the fields read by SharedStorageOffloadingSpec."""
    cfg = MagicMock()
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.world_size = 1
    cfg.parallel_config.rank = 0
    cfg.cache_config.cache_dtype = "fp16"
    cfg.model_config.model = FAKE_MODEL
    # extra_config is read from kv_transfer_config.kv_connector_extra_config
    # in OffloadingSpec.__init__, but we bypass that with the patched init below.
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "shared_storage_path": storage_path,
    }
    return cfg


def _make_kv_cache_config(num_groups: int = 1) -> MagicMock:
    """Minimal KVCacheConfig mock with ``num_groups`` KV cache groups."""
    cfg = MagicMock()
    groups = []
    for i in range(num_groups):
        g = MagicMock()
        g.layer_names = [f"layer_{i}"]
        groups.append(g)
    cfg.kv_cache_groups = groups
    return cfg


def _build_spec(
    *,
    hybrid: bool,
    sm: tuple[int, int] | None = (8, 9),
    storage_path: str = FAKE_STORAGE,
    num_groups: int = 1,
) -> SharedStorageOffloadingSpec:
    """
    Construct a SharedStorageOffloadingSpec with mocked vLLM internals.

    Patches ``OffloadingSpec.__init__`` so the test never touches real vLLM
    config objects or the CUDA runtime, then injects the minimal set of
    attributes that the child ``__init__`` reads from ``self``:

    * ``extra_config`` — forwarded from vllm_config mock
    * ``hybrid_offload_enabled`` — controls whether a GPU tag is inserted
    * ``gpu_block_size`` / ``group_hash_block_size`` / ``offloaded_block_size``
      — used to compute ``gpu_blocks_per_file``

    ``torch.cuda.get_device_capability`` is patched separately so tests can
    exercise normal SM detection and the error-fallback path.

    Args:
        hybrid: Whether to simulate a hybrid (mamba + attention) model.
        sm: SM version to report, e.g. ``(8, 9)`` for SM 8.9.
            Pass ``None`` to make the capability query raise RuntimeError.
        storage_path: Root directory passed via extra_config.
        num_groups: Number of KV cache groups (1 for pure attention,
            4 for Qwen3.5-style hybrid with 3 attention + 1 mamba group).

    Returns:
        A fully constructed SharedStorageOffloadingSpec.
    """
    gpu_block_size = 16
    offloaded_block_size = gpu_block_size * 8  # 8 GPU blocks per file

    def patched_parent_init(self_obj, vllm_cfg, kv_cfg):
        """Replaces OffloadingSpec.__init__ with minimal attribute setup."""
        self_obj.vllm_config = vllm_cfg
        self_obj.kv_cache_config = kv_cfg
        self_obj.extra_config = vllm_cfg.kv_transfer_config.kv_connector_extra_config
        self_obj.hybrid_offload_enabled = hybrid
        self_obj.gpu_block_size = (gpu_block_size,) * num_groups
        self_obj.group_hash_block_size = (gpu_block_size,) * num_groups
        self_obj.offloaded_block_size = offloaded_block_size

    def fake_get_device_capability():
        if sm is None:
            raise RuntimeError("CUDA device not available in test environment")
        return sm

    vllm_config = _make_vllm_config(storage_path)
    kv_cache_config = _make_kv_cache_config(num_groups)

    with (
        patch.object(OffloadingSpec, "__init__", patched_parent_init),
        patch("torch.cuda.get_device_capability", fake_get_device_capability),
    ):
        spec = SharedStorageOffloadingSpec(vllm_config, kv_cache_config)

    return spec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHybridModelGPUTagPath:
    """
    SharedStorageOffloadingSpec must embed the GPU SM version in the NFS path
    for hybrid models so different GPU architectures never share mamba SSM state.
    """

    @pytest.mark.parametrize(
        "sm_major, sm_minor, expected_tag",
        [
            (8, 9, "sm_89"),    # RTX 3090 / RTX 4080 Super family
            (10, 0, "sm_100"),  # RTX 5000 Ada (SM 10.0)
            (12, 0, "sm_120"),  # RTX 5090 Blackwell
        ],
    )
    def test_sm_tag_format(self, sm_major: int, sm_minor: int, expected_tag: str):
        """
        The GPU tag is ``sm_{major}{minor}`` with no separator (e.g. ``sm_89``,
        not ``sm_8_9`` or ``sm_8.9``).
        """
        spec = _build_spec(hybrid=True, sm=(sm_major, sm_minor))
        base = spec.file_mappers[0].base_path
        assert expected_tag in base, (
            f"Expected tag '{expected_tag}' in path '{base}' "
            f"for SM ({sm_major}, {sm_minor})"
        )

    def test_hybrid_path_structure(self):
        """
        For a hybrid model the full path has the form:
          ``{storage_path}/group_0/{sm_tag}/{model}/…``
        """
        spec = _build_spec(hybrid=True, sm=(8, 9), storage_path=FAKE_STORAGE)
        base = spec.file_mappers[0].base_path
        assert base.startswith(f"{FAKE_STORAGE}/group_0/sm_89/{FAKE_MODEL}/"), (
            f"Unexpected path layout: {base}"
        )

    def test_all_groups_tagged(self):
        """
        Every KV cache group's FileMapper gets the same GPU tag, not just
        group 0.  This is critical: mamba state lives in group 3 and all
        groups must be consistent.
        """
        num_groups = 4  # matches Qwen3.5-4B-FP8 (3 attention + 1 mamba)
        spec = _build_spec(hybrid=True, sm=(8, 9), num_groups=num_groups)
        for gi in range(num_groups):
            base = spec.file_mappers[gi].base_path
            assert "sm_89" in base, (
                f"group {gi} FileMapper missing GPU tag: {base}"
            )
            assert f"group_{gi}" in base, (
                f"group {gi} FileMapper missing group prefix: {base}"
            )

    def test_different_sm_versions_produce_different_paths(self):
        """
        Two hosts with different GPU architectures must write to completely
        separate NFS namespaces — the paths must not share a common prefix
        beyond the storage root and group directory.
        """
        spec_89 = _build_spec(hybrid=True, sm=(8, 9))
        spec_120 = _build_spec(hybrid=True, sm=(12, 0))

        path_89 = spec_89.file_mappers[0].base_path
        path_120 = spec_120.file_mappers[0].base_path

        assert path_89 != path_120, "Different SM versions must produce different paths"
        assert "sm_89" in path_89
        assert "sm_120" in path_120
        # The paths diverge immediately after the group prefix
        assert f"group_0/sm_89/" in path_89
        assert f"group_0/sm_120/" in path_120

    def test_sm_tag_fallback_on_cuda_error(self):
        """
        When ``torch.cuda.get_device_capability()`` raises (e.g. no GPU in
        the environment), the spec falls back to the literal tag
        ``sm_unknown`` rather than crashing.  The path is still valid and
        isolated from any real SM version.
        """
        spec = _build_spec(hybrid=True, sm=None)  # None → RuntimeError
        base = spec.file_mappers[0].base_path
        assert "sm_unknown" in base, (
            f"Expected fallback tag 'sm_unknown' in path '{base}'"
        )

    def test_sm_unknown_does_not_match_real_tag(self):
        """
        The fallback ``sm_unknown`` path must not collide with any real
        SM version tag (which always follows the ``sm_NNN`` numeric pattern).
        """
        spec_fallback = _build_spec(hybrid=True, sm=None)
        spec_real = _build_spec(hybrid=True, sm=(8, 9))

        assert spec_fallback.file_mappers[0].base_path != spec_real.file_mappers[0].base_path


class TestNonHybridModelNoGPUTag:
    """
    For non-hybrid (attention-only) models the KV state is fully portable
    across GPU architectures — no SM tag should appear in the path.
    """

    def test_non_hybrid_path_has_no_sm_tag(self):
        """
        Non-hybrid model paths must not contain any ``sm_`` component,
        regardless of the current GPU.
        """
        spec = _build_spec(hybrid=False, sm=(8, 9))
        base = spec.file_mappers[0].base_path
        assert "sm_" not in base, (
            f"Non-hybrid path must not contain GPU tag, got: {base}"
        )

    def test_non_hybrid_path_structure(self):
        """
        For a non-hybrid model the path goes directly:
          ``{storage_path}/group_0/{model}/…``  (no SM component).
        """
        spec = _build_spec(hybrid=False, sm=(8, 9), storage_path=FAKE_STORAGE)
        base = spec.file_mappers[0].base_path
        assert base.startswith(f"{FAKE_STORAGE}/group_0/{FAKE_MODEL}/"), (
            f"Unexpected non-hybrid path layout: {base}"
        )

    def test_non_hybrid_paths_are_same_across_sm_versions(self):
        """
        Two non-hybrid deployments on different GPU architectures should
        produce identical paths — they share the same KV cache on NFS.
        """
        spec_89 = _build_spec(hybrid=False, sm=(8, 9))
        spec_120 = _build_spec(hybrid=False, sm=(12, 0))

        assert spec_89.file_mappers[0].base_path == spec_120.file_mappers[0].base_path, (
            "Non-hybrid paths should be GPU-architecture-agnostic"
        )

    def test_non_hybrid_multiple_groups_no_sm_tag(self):
        """All groups in a non-hybrid model must be free of SM tags."""
        num_groups = 3
        spec = _build_spec(hybrid=False, sm=(8, 9), num_groups=num_groups)
        for gi in range(num_groups):
            base = spec.file_mappers[gi].base_path
            assert "sm_" not in base, (
                f"Non-hybrid group {gi} path unexpectedly contains SM tag: {base}"
            )


class TestSpecFileMapperConsistency:
    """Sanity checks on FileMapper wiring in SharedStorageOffloadingSpec."""

    def test_file_mapper_count_matches_groups(self):
        """``spec.file_mappers`` has one entry per KV cache group."""
        for num_groups in (1, 2, 4):
            spec = _build_spec(hybrid=True, sm=(8, 9), num_groups=num_groups)
            assert len(spec.file_mappers) == num_groups, (
                f"Expected {num_groups} file_mappers, got {len(spec.file_mappers)}"
            )

    def test_file_mapper_is_first_group(self):
        """``spec.file_mapper`` (singular) is the same object as ``file_mappers[0]``."""
        spec = _build_spec(hybrid=True, sm=(8, 9), num_groups=2)
        assert spec.file_mapper is spec.file_mappers[0]

    def test_each_group_uses_its_own_subdirectory(self):
        """
        Groups must not share a storage subdirectory.  Each group_N path
        must differ from every other group_M path.
        """
        num_groups = 4
        spec = _build_spec(hybrid=True, sm=(8, 9), num_groups=num_groups)
        paths = [spec.file_mappers[gi].base_path for gi in range(num_groups)]
        assert len(set(paths)) == num_groups, (
            f"Expected {num_groups} distinct group paths, got duplicates: {paths}"
        )
        for gi in range(num_groups):
            assert f"group_{gi}" in paths[gi], (
                f"group_{gi} not found in path: {paths[gi]}"
            )
