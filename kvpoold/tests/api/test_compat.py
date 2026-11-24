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

"""``CompatDescriptor`` on the wire, inside ``Attach``."""

import msgspec
import pytest

from kvpoold.api.v1 import PROTOCOL_VERSION, Attach, CompatDescriptor, ToServer

DESCRIPTORS = [
    CompatDescriptor(layout=2 << 20, namespace=b"\xaa" * 8, hash_function="sha256", hash_seed=None),
    CompatDescriptor(layout=1, namespace=b"\x00" * 8, hash_function="sha256_cbor", hash_seed="0"),
]


@pytest.mark.parametrize("compat", DESCRIPTORS)
def test_a_descriptor_round_trips(compat):
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(compat), type=CompatDescriptor) == compat


@pytest.mark.parametrize("compat", DESCRIPTORS)
def test_an_attach_round_trips_with_its_descriptor(compat):
    attach = Attach(call_id=1, version=PROTOCOL_VERSION, client_id="engine-0", rank=None, compat=compat)
    assert msgspec.msgpack.Decoder(ToServer).decode(msgspec.msgpack.encode(attach)) == attach


def test_an_attach_without_a_descriptor_does_not_decode():
    encoded = msgspec.msgpack.encode(["Attach", 1, PROTOCOL_VERSION, "engine-0", None])
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.Decoder(ToServer).decode(encoded)


def test_the_protocol_starts_at_version_1():
    assert PROTOCOL_VERSION == 1
