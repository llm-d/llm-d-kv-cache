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

"""Every protocol message survives msgspec's msgpack encoder and the decoder of the
side that receives it."""

import msgspec
import pytest
from vllm.distributed.kv_events import BlockStored

from kvpoold.api.v1 import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompatDescriptor,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromServer,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    NoSpace,
    PrepareLoad,
    PrepareStore,
    Refused,
    RegionDescriptor,
    ReleaseHits,
    RequestFinished,
    ShmFileRegion,
    Stats,
    StoreSpec,
    ToServer,
    UnknownClient,
)

KEY_A = b"\x01" * 32 + b"\x00\x00\x00\x00"
KEY_B = b"\x02" * 32 + b"\x00\x00\x00\x01"

EVENTS = [
    BlockStored(
        block_hashes=[b"\x01" * 32, 12345678901234567890],
        parent_block_hash=None,
        token_ids=list(range(32)),
        block_size=16,
        lora_id=None,
        medium=None,
        lora_name=None,
        group_idx=0,
    ),
    BlockStored(
        block_hashes=[b"\x02" * 32],
        parent_block_hash=b"\x01" * 32,
        token_ids=[],
        block_size=16,
        lora_id=7,
        medium=None,
        lora_name="adapter",
        extra_keys=[("adapter", ["mm-id", 4]), None],
        group_idx=1,
        kv_cache_spec_kind="mamba",
        kv_cache_spec_sliding_window=128,
        lookahead_token_ids=[[1, 2, 3]],
    ),
]

REGION = ShmFileRegion(
    region_id=b"region-1",
    num_chunks=4,
    kv_bytes_per_chunk=2 << 20,
    segments=[
        ("/host-shm/kvpoold.mmap", 0, 4 << 20),
        ("/host-shm/kvpoold.mmap.1", 4 << 20, 4 << 20),
    ],
)

TO_SERVER = [
    Attach(
        call_id=1,
        version=PROTOCOL_VERSION,
        client_id="engine-0",
        rank=3,
        compat=CompatDescriptor(layout=2 << 20, namespace=b"\xaa" * 8, hash_function="sha256", hash_seed=None),
    ),
    Attach(
        call_id=2,
        version=PROTOCOL_VERSION,
        client_id="engine-0",
        rank=None,
        compat=CompatDescriptor(layout=2 << 20, namespace=b"\xaa" * 8, hash_function="sha256", hash_seed="0"),
    ),
    Attach(
        call_id=7,
        version=PROTOCOL_VERSION,
        client_id="engine-1",
        rank=0,
        compat=CompatDescriptor(layout=1, namespace=b"\x00" * 8, hash_function="sha256_cbor", hash_seed="0"),
    ),
    Lookup(call_id=3, req_id="req-1", key=KEY_A),
    PrepareLoad(call_id=4, req_id="req-1", keys=[KEY_A, KEY_B], positions=[16, None]),
    PrepareStore(call_id=5, req_id="req-2", keys=[KEY_A, KEY_B], positions=[16, None]),
    GetStats(call_id=6),
    CompleteLoad(keys=[KEY_A]),
    CompleteStore(keys=[KEY_B], success=False),
    CompleteStore(keys=[KEY_A, KEY_B], success=True, events=EVENTS),
    RequestFinished(req_id="req-1"),
    ReleaseHits(hits=[(KEY_A, 2), (KEY_B, 1)]),
    Leave(),
]

FROM_SERVER = [
    AttachOk(call_id=1, region=REGION),
    Refused(call_id=2, reason="kv_bytes_per_chunk differs from the pool"),
    LookupOk(call_id=3, result="hit_pending"),
    LoadSpec(call_id=4, chunk_ids=[0, 3]),
    StoreSpec(call_id=5, keys_to_store=[KEY_B], chunk_ids=[2]),
    NoSpace(call_id=6),
    Stats(call_id=7, values={"num_chunks": 4, "free": 1}),
    Draining(call_id=8),
    UnknownClient(call_id=9),
]

ENCODER = msgspec.msgpack.Encoder()
SERVER = msgspec.msgpack.Decoder(ToServer)
CLIENT = msgspec.msgpack.Decoder(FromServer)


@pytest.mark.parametrize("message", TO_SERVER, ids=lambda m: type(m).__name__)
def test_to_server_round_trip(message):
    decoded = SERVER.decode(ENCODER.encode(message))
    assert type(decoded) is type(message)
    assert decoded == message


@pytest.mark.parametrize("message", FROM_SERVER, ids=lambda m: type(m).__name__)
def test_from_server_round_trip(message):
    decoded = CLIENT.decode(ENCODER.encode(message))
    assert type(decoded) is type(message)
    assert decoded == message


def test_complete_store_without_events_has_none():
    decoded = SERVER.decode(ENCODER.encode(CompleteStore(keys=[KEY_A], success=True)))
    assert decoded.events is None


def test_events_decode_as_block_stored_with_hash_types_kept():
    message = CompleteStore(keys=[KEY_A, KEY_B], success=True, events=EVENTS)
    decoded = SERVER.decode(ENCODER.encode(message))
    first, second = decoded.events
    assert all(type(event) is BlockStored for event in decoded.events)
    assert isinstance(first.block_hashes[0], bytes)
    assert isinstance(first.block_hashes[1], int)
    assert second.parent_block_hash == b"\x01" * 32
    assert second.lookahead_token_ids == [[1, 2, 3]]


def test_events_reencode_to_the_bytes_the_engine_sent():
    """The pool server publishes the decoded events, so a decode and
    encode must not change them, even though nested ``extra_keys`` values decode
    as lists where vLLM builds tuples."""
    decoded = SERVER.decode(ENCODER.encode(CompleteStore(keys=[KEY_A, KEY_B], success=True, events=EVENTS)))
    assert decoded.events[1].extra_keys == [("adapter", ["mm-id", 4]), None]
    for sent, received in zip(EVENTS, decoded.events):
        assert ENCODER.encode(received) == ENCODER.encode(sent)


def test_complete_store_decodes_without_the_trailing_events_field():
    """A CompleteStore array that stops after ``success`` decodes with no events,
    and one with extra trailing elements decodes with them ignored."""
    tag = type(CompleteStore(keys=[], success=True)).__struct_config__.tag
    short = msgspec.msgpack.encode([tag, [KEY_A], True])
    assert SERVER.decode(short) == CompleteStore(keys=[KEY_A], success=True)
    long = msgspec.msgpack.encode([tag, [KEY_A], True, None, "later"])
    assert SERVER.decode(long) == CompleteStore(keys=[KEY_A], success=True)


def test_region_subclass_survives_the_wire():
    decoded = CLIENT.decode(ENCODER.encode(AttachOk(call_id=1, region=REGION)))
    assert isinstance(decoded.region, ShmFileRegion)
    assert decoded.region.segments == REGION.segments


def test_region_descriptor_is_abstract():
    with pytest.raises(TypeError):
        RegionDescriptor(region_id=b"r", num_chunks=1, kv_bytes_per_chunk=4096)


def test_each_side_rejects_the_other_sides_messages():
    with pytest.raises(msgspec.ValidationError):
        SERVER.decode(ENCODER.encode(LookupOk(call_id=1, result="hit")))
    with pytest.raises(msgspec.ValidationError):
        CLIENT.decode(ENCODER.encode(Leave()))
