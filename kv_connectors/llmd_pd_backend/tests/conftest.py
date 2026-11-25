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

"""Shared test fixtures and vLLM/NIXL mock infrastructure.

Since vllm and nixl are not installed in the test environment, we mock
the required modules before importing llmd_pd_backend code.
"""

from __future__ import annotations

import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, NewType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal type stubs matching vLLM's real types
# ---------------------------------------------------------------------------

OffloadKey = NewType("OffloadKey", bytes)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


class LookupResult(Enum):
    MISS = auto()
    HIT = auto()
    HIT_PENDING = auto()
    RETRY = auto()


class Medium(Enum):
    CPU = "CPU"
    STORAGE = "STORAGE"


class Locality(Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class OffloadPolicy(Enum):
    CHUNK_LEVEL = "chunk_level"
    REQUEST_LEVEL = "request_level"


@dataclass
class RequestOffloadingContext:
    policy: OffloadPolicy = OffloadPolicy.CHUNK_LEVEL


@dataclass
class ScheduleEndContext:
    new_req_ids: Collection[str] = ()
    preempted_req_ids: Collection[str] = ()


@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict[str, Any] | None = None
    _state: dict[type, Any] = field(default_factory=dict, repr=False, init=False)

    def set_state(self, val: Any) -> None:
        self._state[type(val)] = val

    def get_state(self, cls: type) -> Any | None:
        return self._state.get(cls)


@dataclass
class TransferJob:
    job_id: int
    keys: Collection[OffloadKey]
    chunk_ids: np.ndarray
    is_promotion: bool
    req_context: ReqContext


@dataclass
class JobResult:
    job_id: int
    success: bool
    successful_keys: Collection[OffloadKey] | None = None
    transfer_time: float | None = None


class SecondaryTierManager(ABC):
    medium: ClassVar[Medium]
    locality: ClassVar[Locality | None] = None

    def __init__(
        self,
        offloading_spec: Any,
        primary_kv_view: memoryview,
        tier_type: str,
    ):
        self.offloading_spec = offloading_spec
        self.primary_kv_view = primary_kv_view
        self.tier_type = tier_type

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext: ...

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult: ...

    @abstractmethod
    def submit_store(self, job: TransferJob) -> None: ...

    @abstractmethod
    def submit_load(self, job: TransferJob) -> None: ...

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]: ...

    @abstractmethod
    def drain_jobs(self) -> None: ...

    @abstractmethod
    def has_pending_work(self) -> bool: ...

    def shutdown(self) -> None:  # noqa: B027
        pass


class ParentManager(ABC):
    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext: ...

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult: ...


# ---------------------------------------------------------------------------
# Module mocking
# ---------------------------------------------------------------------------


def _install_mock_modules():
    """Install mock vllm and nixl modules into sys.modules."""

    def _make_mod(name: str, attrs: dict | None = None) -> types.ModuleType:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        return mod

    vllm_base_attrs = {
        "OffloadKey": OffloadKey,
        "make_offload_key": make_offload_key,
        "LookupResult": LookupResult,
        "Medium": Medium,
        "Locality": Locality,
        "ReqContext": ReqContext,
        "RequestOffloadingContext": RequestOffloadingContext,
        "ScheduleEndContext": ScheduleEndContext,
        "OffloadPolicy": OffloadPolicy,
    }

    tiering_base_attrs = {
        "SecondaryTierManager": SecondaryTierManager,
        "TransferJob": TransferJob,
        "JobResult": JobResult,
        "ParentManager": ParentManager,
    }

    mock_nixl_agent = MagicMock(name="nixl_agent")
    mock_nixl_agent_config = MagicMock(name="nixl_agent_config")

    nixl_api_attrs = {
        "nixl_agent": mock_nixl_agent,
        "nixl_agent_config": mock_nixl_agent_config,
    }

    mock_init_logger = MagicMock(return_value=MagicMock())

    modules = {
        "vllm": _make_mod("vllm"),
        "vllm.v1": _make_mod("vllm.v1"),
        "vllm.v1.kv_offload": _make_mod("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": _make_mod(
            "vllm.v1.kv_offload.base", vllm_base_attrs
        ),
        "vllm.v1.kv_offload.tiering": _make_mod("vllm.v1.kv_offload.tiering"),
        "vllm.v1.kv_offload.tiering.base": _make_mod(
            "vllm.v1.kv_offload.tiering.base", tiering_base_attrs
        ),
        "vllm.logger": _make_mod("vllm.logger", {"init_logger": mock_init_logger}),
    }

    # Only mock nixl if not installed
    try:
        import nixl  # noqa: F401
    except ImportError:
        modules["nixl"] = _make_mod("nixl")
        modules["nixl._api"] = _make_mod("nixl._api", nixl_api_attrs)
        modules["nixl.logging"] = _make_mod(
            "nixl.logging", {"get_logger": MagicMock(return_value=MagicMock())}
        )

    for name, mod in modules.items():
        sys.modules[name] = mod

    return mock_nixl_agent, mock_nixl_agent_config


# Install mocks before any test imports
_mock_nixl_agent_cls, _mock_nixl_agent_config_cls = _install_mock_modules()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nixl_agent():
    """Return a fresh mock NIXL agent instance."""
    agent = MagicMock(name="nixl_agent_instance")
    agent.register_memory.return_value = MagicMock(name="mem_desc")
    agent.get_xfer_descs.return_value = MagicMock(name="xfer_descs")
    agent.initialize_xfer.return_value = MagicMock(name="xfer_handle")
    agent.transfer.return_value = "PROC"
    agent.check_xfer_state.return_value = "DONE"
    agent.get_plugin_list.return_value = ["UCX"]
    agent.get_plugin_mem_types.return_value = ["DRAM"]
    agent.get_plugin_params.return_value = {}
    agent.get_backend_mem_types.return_value = ["DRAM"]

    from unittest.mock import patch

    with (
        patch("llmd_pd_backend.transport.nixl_agent", return_value=agent),
        patch("nixl._api.nixl_agent", return_value=agent),
    ):
        yield agent


@pytest.fixture
def primary_kv_buf() -> bytearray:
    """A 4KB CPU buffer simulating the primary tier."""
    return bytearray(4096)


@pytest.fixture
def primary_kv_view(primary_kv_buf) -> memoryview:
    return memoryview(primary_kv_buf)


@pytest.fixture
def mock_offloading_spec():
    return MagicMock(name="OffloadingSpec")


@pytest.fixture
def sample_keys() -> list[OffloadKey]:
    return [make_offload_key(f"block_{i}".encode(), 0) for i in range(4)]


@pytest.fixture
def req_context() -> ReqContext:
    return ReqContext(req_id="test-req-1")


@pytest.fixture
def pd_req_context() -> ReqContext:
    """ReqContext with remote_prefiller params for decode-side loads."""
    return ReqContext(
        req_id="decode-req-1",
        kv_transfer_params={
            "remote_prefiller": {
                "kv_request_id": "kv-123",
                "remote_host": "127.0.0.1",
                "remote_port": 5710,
            }
        },
    )
