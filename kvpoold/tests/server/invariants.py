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


"""Checks that the pool's bookkeeping agrees with vLLM's manager."""

from collections import Counter

from kvpoold.server.pool import Pool


def resident_chunks(pool: Pool) -> dict[bytes, object]:
    """Pool key to its chunk, for every chunk the cache policy holds."""
    policy = pool.manager._policy
    if hasattr(policy, "chunks"):  # LRU
        return dict(policy.chunks)
    return {**policy.t1, **policy.t2}  # ARC


def check(pool: Pool) -> None:
    """Assert that the manager's counters and chunk pool, and the clients' pins
    and writes, are consistent with each other."""
    manager = pool.manager
    chunks = resident_chunks(pool)

    # Every chunk id is held by exactly one chunk, or is free.
    held = sorted(chunk.chunk_id for chunk in chunks.values())
    free = sorted(manager._free_list)
    assert sorted(held + free) == list(range(manager._num_allocated_chunks))
    assert manager._get_num_free_chunks() + len(chunks) == manager._num_chunks

    # The manager's counters match the chunks' states.
    assert manager._num_evictable_cache_chunks == sum(chunk.ref_cnt == 0 for chunk in chunks.values())
    assert manager._num_write_pending_chunks == sum(chunk.ref_cnt == -1 for chunk in chunks.values())

    # A chunk's ref_cnt is the clients' pins on it, or -1 for a write.
    pins: Counter[bytes] = Counter()
    writes: Counter[bytes] = Counter()
    for client in pool.clients.values():
        for pool_key, deadlines in client.pins.items():
            assert deadlines, "an empty pin list is deleted"
            pins[pool_key] += len(deadlines)
        writes.update(client.writes.keys())
    for pool_key, chunk in chunks.items():
        if chunk.ref_cnt == -1:
            assert writes[pool_key] == 1 and pins[pool_key] == 0, pool_key
        else:
            assert chunk.ref_cnt == pins[pool_key] and writes[pool_key] == 0, pool_key
    assert set(pins) | set(writes) <= set(chunks)
