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

import logging
import struct
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import msgpack
import pytest

pytestmark = pytest.mark.no_cuda_required

CONNECTOR_ROOT = Path(__file__).resolve().parents[2]


class PrepareStoreOutput:
    def __init__(self, block_hashes_to_store, store_spec, block_hashes_evicted):
        self.block_hashes_to_store = block_hashes_to_store
        self.store_spec = store_spec
        self.block_hashes_evicted = block_hashes_evicted


class SharedStorageLoadStoreSpec:
    def __init__(self, block_hashes):
        self.block_hashes = list(block_hashes)


class NixlLookup:
    def __init__(self, _cfg):
        pass

    def exists(self, _key):
        return False


@contextmanager
def temporary_modules(modules):
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def module(name, **attrs):
    mod = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(mod, attr_name, value)
    return mod


def package(name):
    mod = module(name)
    mod.__path__ = []
    return mod


def load_module(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_storage_event_modules():
    llmd_fs_backend_pkg = package("llmd_fs_backend")
    llmd_nixl_pkg = package("llmd_nixl")
    loaded_names = [
        "storage_event_publisher_under_test",
        "llmd_fs_backend.event_publisher",
        "llmd_fs_backend.manager",
        "storage_nixl_manager_under_test",
    ]
    sentinel = object()
    previous_loaded = {name: sys.modules.get(name, sentinel) for name in loaded_names}

    stubs = {
        "llmd_fs_backend": llmd_fs_backend_pkg,
        "llmd_fs_backend.file_mapper": module(
            "llmd_fs_backend.file_mapper", FileMapper=object
        ),
        "llmd_fs_backend.mediums": module(
            "llmd_fs_backend.mediums",
            SharedStorageLoadStoreSpec=SharedStorageLoadStoreSpec,
        ),
        "llmd_nixl": llmd_nixl_pkg,
        "llmd_nixl.nixl_lookup": module("llmd_nixl.nixl_lookup", NixlLookup=NixlLookup),
        "vllm": package("vllm"),
        "vllm.v1": package("vllm.v1"),
        "vllm.v1.kv_offload": package("vllm.v1.kv_offload"),
        "vllm.logger": module(
            "vllm.logger", init_logger=lambda name: logging.getLogger(name)
        ),
        "vllm.v1.kv_offload.base": module(
            "vllm.v1.kv_offload.base",
            LoadStoreSpec=object,
            OffloadingManager=object,
            OffloadKey=object,
            PrepareStoreOutput=PrepareStoreOutput,
            ReqContext=object,
            RequestOffloadingContext=object,
            get_offload_block_hash=lambda key: key,
        ),
    }

    with temporary_modules(stubs):
        event_publisher = load_module(
            "storage_event_publisher_under_test",
            CONNECTOR_ROOT / "llmd_fs_backend" / "event_publisher.py",
        )
        sys.modules["llmd_fs_backend.event_publisher"] = event_publisher
        manager = load_module(
            "llmd_fs_backend.manager",
            CONNECTOR_ROOT / "llmd_fs_backend" / "manager.py",
        )
        nixl_manager = load_module(
            "storage_nixl_manager_under_test",
            CONNECTOR_ROOT / "llmd_nixl" / "manager.py",
        )

    for name, previous in previous_loaded.items():
        if previous is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    return event_publisher, manager, nixl_manager


event_publisher_module, manager_module, nixl_manager_module = (
    load_storage_event_modules()
)

StorageEventPublisher = event_publisher_module.StorageEventPublisher
StorageMedium = event_publisher_module.StorageMedium
_hash_to_uint64 = event_publisher_module._hash_to_uint64
SharedStorageOffloadingManager = manager_module.SharedStorageOffloadingManager
LOOKUP_MODE_DICT = nixl_manager_module.LOOKUP_MODE_DICT
NixlStorageOffloadingManager = nixl_manager_module.NixlStorageOffloadingManager


class FakeZMQSocket:
    def __init__(self):
        self.bound_endpoint = None
        self.closed = 0
        self.options = []
        self.sent = []

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def bind(self, endpoint):
        self.bound_endpoint = endpoint

    def send_multipart(self, frames):
        self.sent.append(frames)

    def close(self):
        self.closed += 1


class FakeZMQContext:
    def __init__(self):
        self.socket_instance = FakeZMQSocket()
        self.socket_types = []
        self.terminated = 0

    def socket(self, socket_type):
        self.socket_types.append(socket_type)
        return self.socket_instance

    def term(self):
        self.terminated += 1


class RecordingPublisher:
    def __init__(self):
        self.stored_calls = []
        self.removed_calls = []

    def publish_blocks_stored(self, block_hashes):
        self.stored_calls.append(list(block_hashes))

    def publish_blocks_removed(self, block_hashes, model_name=None):
        self.removed_calls.append((model_name, list(block_hashes)))


class ExplodingPublisher:
    def publish_blocks_stored(self, _block_hashes):
        raise RuntimeError("publish failed")


class FakeFileMapper:
    def get_file_name(self, block_hash):
        return f"file-{block_hash}"


def _publisher_with_fake_zmq(monkeypatch):
    ctx = FakeZMQContext()
    monkeypatch.setattr(event_publisher_module.zmq, "Context", lambda: ctx)

    publisher = StorageEventPublisher("tcp://*:5559", "test-model", 100_000)
    return publisher, ctx


def test_hash_to_uint64_matches_file_mapper_lower_64_bits():
    """Verify _hash_to_uint64 masks ints to lower 64 bits
    and converts bytes big-endian."""
    assert _hash_to_uint64(1) == 1
    assert _hash_to_uint64((1 << 72) + 5) == 5
    assert _hash_to_uint64(b"\x01\x02") == 0x0102


def test_storage_event_publisher_emits_go_compatible_three_frame_message(monkeypatch):
    """Verify publish_blocks_stored produces the 3-frame ZMQ
    message (topic, sequence, payload) with msgpack events
    matching the Go VLLMAdapter positional array format."""
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_stored([0xABCDEF0123456789, (1 << 72) + 7, b"\x01\x02"])

    assert ctx.socket_instance.bound_endpoint == "tcp://*:5559"
    assert len(ctx.socket_instance.sent) == 1

    topic, seq, payload = ctx.socket_instance.sent[0]
    assert topic == b"kv@SHARED_STORAGE@test-model"
    assert struct.unpack(">Q", seq)[0] == 1

    timestamp, raw_events = msgpack.unpackb(payload, raw=False)
    assert isinstance(timestamp, float)
    assert len(raw_events) == 1

    decoded = msgpack.unpackb(raw_events[0], raw=False)
    assert decoded == [
        "BlockStored",
        [0xABCDEF0123456789, 7, 0x0102],
        0,
        [],
        0,
        None,
        "SHARED_STORAGE",
    ]


def test_storage_event_publisher_sequence_and_close_are_idempotent(monkeypatch):
    """Verify sequence numbers are monotonically increasing,
    close() is idempotent, and no sends occur after close."""
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_stored([1])
    publisher.publish_blocks_stored([2])
    assert [
        struct.unpack(">Q", frames[1])[0] for frames in ctx.socket_instance.sent
    ] == [
        1,
        2,
    ]

    publisher.close()
    publisher.close()
    assert ctx.socket_instance.closed == 1
    assert ctx.terminated == 1

    publisher.publish_blocks_stored([3])
    assert len(ctx.socket_instance.sent) == 2


def test_shared_storage_manager_publishes_successful_stores_only():
    """Verify complete_store publishes events only when
    success=True and skips on failure."""
    publisher = RecordingPublisher()
    manager = SharedStorageOffloadingManager(
        FakeFileMapper(), event_publisher=publisher
    )

    manager.complete_store(iter([11, 22]), None, success=True)
    manager.complete_store([33], None, success=False)

    assert publisher.stored_calls == [[11, 22]]


def test_shared_storage_manager_publish_errors_are_fail_open():
    """Verify a publisher exception during complete_store
    is caught and does not propagate."""
    manager = SharedStorageOffloadingManager(
        FakeFileMapper(), event_publisher=ExplodingPublisher()
    )

    manager.complete_store([11], None, success=True)


def test_nixl_manager_publishes_and_preserves_dict_lookup_bookkeeping():
    """Verify NIXL complete_store publishes events and
    updates _stored_keys for dict lookup mode."""
    publisher = RecordingPublisher()
    manager = NixlStorageOffloadingManager(
        FakeFileMapper(),
        extra_config={"lookup_mode": LOOKUP_MODE_DICT},
        event_publisher=publisher,
    )

    manager.complete_store(iter([11, 22]), None, success=True)

    assert publisher.stored_calls == [[11, 22]]
    assert manager._stored_keys == {"file-11", "file-22"}


def test_nixl_manager_does_not_publish_or_record_failed_stores():
    """Verify NIXL complete_store emits no events and does
    not update _stored_keys when success=False."""
    publisher = RecordingPublisher()
    manager = NixlStorageOffloadingManager(
        FakeFileMapper(),
        extra_config={"lookup_mode": LOOKUP_MODE_DICT},
        event_publisher=publisher,
    )

    manager.complete_store([11], None, success=False)

    assert publisher.stored_calls == []
    assert manager._stored_keys == set()


def test_publish_blocks_removed_emits_correct_event_format(monkeypatch):
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_removed([0xABCDEF0123456789, 0x1234])

    assert len(ctx.socket_instance.sent) == 1

    topic, seq, payload = ctx.socket_instance.sent[0]
    assert topic == b"kv@SHARED_STORAGE@test-model"
    assert struct.unpack(">Q", seq)[0] == 1

    timestamp, raw_events = msgpack.unpackb(payload, raw=False)
    assert isinstance(timestamp, float)
    assert len(raw_events) == 1

    event = msgpack.unpackb(raw_events[0], raw=False)
    assert event == [
        "BlockRemoved",
        [0xABCDEF0123456789, 0x1234],
        "SHARED_STORAGE",
    ]


def test_publish_blocks_removed_with_model_name_override(monkeypatch):
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_removed([0x1234], model_name="other-model")

    topic, _, _ = ctx.socket_instance.sent[0]
    assert topic == b"kv@SHARED_STORAGE@other-model"


def test_publish_blocks_removed_empty_hashes_is_noop(monkeypatch):
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_removed([])

    assert len(ctx.socket_instance.sent) == 0


def test_publisher_without_model_name(monkeypatch):
    ctx = FakeZMQContext()
    monkeypatch.setattr(event_publisher_module.zmq, "Context", lambda: ctx)

    publisher = StorageEventPublisher("tcp://*:5559")
    assert publisher._topic is None

    publisher.publish_blocks_removed([0x1234], model_name="some-model")

    topic, _, _ = ctx.socket_instance.sent[0]
    assert topic == b"kv@SHARED_STORAGE@some-model"


def test_publish_blocks_removed_uses_instance_topic_when_no_override(monkeypatch):
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_removed([0x1234])

    topic, _, _ = ctx.socket_instance.sent[0]
    assert topic == b"kv@SHARED_STORAGE@test-model"


def test_publish_blocks_removed_masks_to_uint64(monkeypatch):
    publisher, ctx = _publisher_with_fake_zmq(monkeypatch)

    publisher.publish_blocks_removed([(1 << 72) + 5, b"\x01\x02"])

    _, _, payload = ctx.socket_instance.sent[0]
    _, raw_events = msgpack.unpackb(payload, raw=False)
    event = msgpack.unpackb(raw_events[0], raw=False)
    assert event[1] == [5, 0x0102]
