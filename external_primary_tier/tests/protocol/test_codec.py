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

"""Every protocol message survives an encode and decode on the side that receives it."""

import msgspec
import pytest
from external_primary_tier.protocol import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompleteLoad,
    CompleteStore,
    Draining,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    PrepareLoad,
    PrepareStore,
    Refused,
    RegionDescriptor,
    ReleaseHits,
    RequestFinished,
    ShmFileRegion,
    Stats,
    StoreSpec,
    UnknownClient,
    client_codec,
    tier_codec,
)

KEY_A = b"\x01" * 32 + b"\x00\x00\x00\x00"
KEY_B = b"\x02" * 32 + b"\x00\x00\x00\x01"

REGION = ShmFileRegion(
    region_id=b"region-1",
    num_chunks=4,
    kv_bytes_per_chunk=2 << 20,
    segments=[
        ("/host-shm/external-primary-tier.mmap", 0, 4 << 20),
        ("/host-shm/external-primary-tier.mmap.1", 4 << 20, 4 << 20),
    ],
)

TO_TIER = [
    Attach(
        call_id=1,
        version=PROTOCOL_VERSION,
        client_id="engine-0@pod-a",
        rank=3,
        kv_bytes_per_chunk=2 << 20,
        hash_function="sha256",
        hash_seed=None,
        compat_id=b"\xaa" * 8,
    ),
    Attach(
        call_id=2,
        version=PROTOCOL_VERSION,
        client_id="engine-0@pod-a",
        rank=None,
        kv_bytes_per_chunk=2 << 20,
        hash_function="sha256",
        hash_seed="0",
        compat_id=b"\xaa" * 8,
    ),
    Lookup(call_id=3, req_id="req-1", key=KEY_A),
    PrepareLoad(call_id=4, req_id="req-1", keys=[KEY_A, KEY_B]),
    PrepareStore(call_id=5, req_id="req-2", keys=[KEY_B]),
    GetStats(call_id=6),
    CompleteLoad(req_id="req-1", keys=[KEY_A]),
    CompleteStore(req_id="req-2", keys=[KEY_B], success=False),
    RequestFinished(req_id="req-1", key_groups=[[KEY_A, KEY_B], []]),
    ReleaseHits(hits=[(KEY_A, 2), (KEY_B, 1)]),
    Leave(),
]

FROM_TIER = [
    AttachOk(call_id=1, region=REGION),
    Refused(call_id=2, reason="kv_bytes_per_chunk differs from the pool"),
    LookupOk(call_id=3, result="hit_pending"),
    LoadSpec(call_id=4, chunk_ids=[0, 3]),
    StoreSpec(call_id=5, keys_to_store=[KEY_B], chunk_ids=[2], evicted_keys=[KEY_A]),
    Stats(call_id=6, values={"num_chunks": 4, "free": 1}),
    Draining(call_id=7),
    UnknownClient(call_id=8),
]


@pytest.mark.parametrize("message", TO_TIER, ids=lambda m: type(m).__name__)
def test_to_tier_round_trip(message):
    codec = tier_codec()
    decoded = codec.decode(client_codec().encode(message))
    assert type(decoded) is type(message)
    assert decoded == message


@pytest.mark.parametrize("message", FROM_TIER, ids=lambda m: type(m).__name__)
def test_from_tier_round_trip(message):
    decoded = client_codec().decode(tier_codec().encode(message))
    assert type(decoded) is type(message)
    assert decoded == message


def test_encoded_message_is_one_frame():
    assert len(client_codec().encode(TO_TIER[0])) == 1


def test_region_subclass_survives_the_wire():
    decoded = client_codec().decode(tier_codec().encode(AttachOk(call_id=1, region=REGION)))
    assert isinstance(decoded.region, ShmFileRegion)
    assert decoded.region.segments == REGION.segments


def test_region_descriptor_is_abstract():
    with pytest.raises(TypeError):
        RegionDescriptor(region_id=b"r", num_chunks=1, kv_bytes_per_chunk=4096)


def test_each_side_rejects_the_other_sides_messages():
    with pytest.raises(msgspec.ValidationError):
        tier_codec().decode(tier_codec().encode(LookupOk(call_id=1, result="hit")))
    with pytest.raises(msgspec.ValidationError):
        client_codec().decode(client_codec().encode(Leave()))
