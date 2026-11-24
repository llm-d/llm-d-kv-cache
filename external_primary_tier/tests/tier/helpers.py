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

"""Calls a fake engine's scheduler makes, for the end-to-end tests."""

from external_primary_tier.protocol import (
    CompleteStore,
    GetStats,
    Lookup,
    PrepareStore,
    Stats,
    StoreSpec,
)

from .fake_engine import FakeEngineProcess, key


def stats(process: FakeEngineProcess) -> dict[str, int]:
    reply = process.call(GetStats)
    assert isinstance(reply, Stats)
    return reply.values


def store(process: FakeEngineProcess, req_id: str, ids: list[int]) -> StoreSpec:
    """Store the keys and send their completion."""
    reply = process.call(PrepareStore, req_id=req_id, keys=[key(i) for i in ids], positions=[None] * len(ids))
    assert isinstance(reply, StoreSpec)
    process.send(CompleteStore(keys=reply.keys_to_store, success=True))
    # ZMQ orders messages only within one socket. Waiting for a reply on this
    # socket means the tier has handled the CompleteStore, so another engine's
    # lookup that follows sees the chunk as stored, not pending.
    process.call(GetStats)
    return reply


def lookup(process: FakeEngineProcess, req_id: str, i: int) -> str:
    return process.call(Lookup, req_id=req_id, key=key(i)).result
