# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
import pytest
from test_spec import make_spec
from vllm.v1.kv_offload.base import ReqContext

from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.request_manager import RequestStorageTTLManager

pytestmark = pytest.mark.no_cuda_required


def test_allowlist_is_opt_in_and_fixed_default_unchanged(tmp_path):
    old = make_spec(tmp_path / "fixed", {"storage_ttl_seconds": 30})
    assert type(old.get_manager()) is SharedStorageOffloadingManager
    new = make_spec(
        tmp_path / "mixed",
        {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1, 30]},
    )
    m = new.get_manager()
    assert isinstance(m, RequestStorageTTLManager) and new.get_manager() is m
    key = b"prefix" + (0).to_bytes(4, "big")
    c = ReqContext("r", {"llmd_storage_ttl_seconds": 1})
    m.on_new_request(c)
    out = m.prepare_store([key], c)
    source, target = next(iter(out.store_spec.source_paths.items()))
    assert "/.llmd-ttl-v1-30/" in source and "/.llmd-ttl-v1-1/" in target
    assert source == new.file_mapper.get_file_name(key)
    m.complete_store([key], c, success=False)
    m.on_request_finished(c)
    m.shutdown()
    old.get_manager().shutdown()


@pytest.mark.parametrize(
    "extra",
    [
        {"storage_ttl_allowlist": [1, 30]},
        {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1]},
        {
            "storage_ttl_seconds": 30,
            "storage_ttl_allowlist": [1, 30],
            "enable_events": True,
        },
        {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1, 30], "backend": "OBJ"},
        {
            "storage_ttl_seconds": 30,
            "storage_ttl_allowlist": [1, 30],
            "gds_mode": "read_write",
        },
    ],
)
def test_spec_rejects_unsupported_allowlist_configuration(tmp_path, extra):
    with pytest.raises(ValueError):
        make_spec(tmp_path, extra)
