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

"""KV events end to end: engines store and evict over ZMQ, the server publishes them
with vLLM's ZMQ publisher, and a subscriber decodes them as vLLM's ``KVEventBatch``."""

import os
import time

import msgspec
import pytest
import zmq
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)

from kvpoold.api.v1 import CompleteStore, PrepareStore, StoreSpec
from kvpoold.server.app import Server
from kvpoold.server.cli import parse_args
from kvpoold.server.snapshot import SNAPSHOT_FILE_NAME, take_snapshot, write_snapshot

from ..recording_publisher import TOPIC, RecordingPublisher
from . import helpers
from .conftest import POOL_CHUNKS, server_config
from .fake_engine import FakeEngineProcess, key
from .helpers import attach_scheduler, stats
from .subscriber import Subscriber


def block_stored(i: int) -> BlockStored:
    """An engine's event for the chunk of ``key(i)``, as ``build_stored_events`` makes it."""
    return BlockStored(
        block_hashes=[1000 + i],
        parent_block_hash=None,
        token_ids=[i],
        block_size=16,
        lora_id=None,
        medium=None,
        lora_name=None,
        group_idx=0,
    )


def stored(i: int) -> BlockStored:
    """The event the server publishes for it."""
    return msgspec.structs.replace(block_stored(i), medium=MEDIUM_CPU)


def removed(i: int) -> BlockRemoved:
    return BlockRemoved(block_hashes=[1000 + i], medium=MEDIUM_CPU, group_idx=0)


def store(process: FakeEngineProcess, req_id: str, ids: list[int]) -> StoreSpec:
    """Store the keys, with the event of each key stored."""
    return helpers.store(process, req_id, ids, events=block_stored)


def receive_events(subscriber: Subscriber, count: int) -> list:
    """The events of the next batches, in order, until there are ``count``; each
    batch is on ``TOPIC``."""
    events: list = []
    while len(events) < count:
        topic, batch = subscriber.receive()
        assert topic == TOPIC
        events += batch.events
    assert len(events) == count
    return events


@pytest.fixture
def publishing_tier(start_tier, tmp_path):
    """A server with vLLM's ZMQ publisher on ``TOPIC``."""
    return start_tier(
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=True, endpoint=f"ipc://{tmp_path}/events.sock", topic=TOPIC
        )
    )


@pytest.fixture
def subscriber(tmp_path, publishing_tier):
    subscriber = Subscriber(f"ipc://{tmp_path}/events.sock", TOPIC)
    subscriber.wait_until_subscribed(publishing_tier.server.event_publisher)
    yield subscriber
    publishing_tier.stop()
    subscriber.close()


@pytest.fixture
def engine(publishing_tier, processes):
    """Attaches an engine's GPU worker and then its scheduler to the publishing
    server; returns the scheduler's process."""
    return lambda: attach_scheduler(processes, publishing_tier.address, with_worker=True)


# --- Publishing --------------------------------------------------------------------


def test_a_completed_store_is_published_on_the_configured_topic(engine, subscriber):
    store(engine(), "r1", [0, 1])
    topic, batch = subscriber.receive()
    assert topic == TOPIC
    assert batch.events == [stored(0), stored(1)]


def test_each_store_is_published_in_order(engine, subscriber):
    scheduler = engine()
    store(scheduler, "r1", [0])
    store(scheduler, "r2", [1])
    assert receive_events(subscriber, 2) == [stored(0), stored(1)]


def test_an_eviction_is_published_as_a_removal_with_the_stored_hashes(engine, subscriber):
    scheduler = engine()
    store(scheduler, "r1", list(range(POOL_CHUNKS)))
    assert receive_events(subscriber, POOL_CHUNKS) == [stored(i) for i in range(POOL_CHUNKS)]
    store(scheduler, "r2", [POOL_CHUNKS])  # evicts the oldest chunk: the keys carry no positions
    assert receive_events(subscriber, 2) == [removed(0), stored(POOL_CHUNKS)]


def test_a_store_without_events_publishes_nothing(engine, subscriber):
    scheduler = engine()
    reply = scheduler.call(PrepareStore, req_id="r1", keys=[key(0)], positions=[None])
    scheduler.send(CompleteStore(keys=reply.keys_to_store, success=True))
    store(scheduler, "r2", [1])  # the server handles an engine's messages in order
    assert receive_events(subscriber, 1) == [stored(1)]


def test_a_failed_store_publishes_nothing(engine, subscriber):
    scheduler = engine()
    reply = scheduler.call(PrepareStore, req_id="r1", keys=[key(0)], positions=[None])
    scheduler.send(CompleteStore(keys=reply.keys_to_store, success=False, events=[block_stored(0)]))
    store(scheduler, "r2", [1])
    assert receive_events(subscriber, 1) == [stored(1)]


# --- Stop ------------------------------------------------------------------------------


def test_the_clear_comes_first_and_the_publisher_shuts_down_after_the_last_events(
    start_tier, processes, recording_config
):
    tier = start_tier(kv_events_config=recording_config)
    store(attach_scheduler(processes, tier.address, with_worker=True), "r1", [0])
    tier.stop()
    assert tier.server.event_publisher.calls == [
        ("publish", [AllBlocksCleared()]),
        ("publish", [stored(0)]),
        ("shutdown", []),
    ]


def test_a_stopped_vllm_publisher_refuses_more_batches(publishing_tier):
    publisher = publishing_tier.server.event_publisher
    publishing_tier.stop()
    with pytest.raises(RuntimeError):
        publisher.publish(KVEventBatch(ts=0.0, events=[stored(0)]))


def test_the_server_creates_the_publisher_the_config_names_from_the_config_fields(start_tier, recording_config):
    publisher = start_tier(kv_events_config=recording_config).server.event_publisher
    assert isinstance(publisher, RecordingPublisher)
    assert publisher.config_fields["topic"] == TOPIC


# --- Events cannot be off ------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        KVEventsConfig(enable_kv_cache_events=False, endpoint="ipc:///nowhere/events.sock", topic=TOPIC),
        KVEventsConfig(enable_kv_cache_events=True, publisher="null", topic=TOPIC),
    ],
    ids=["events disabled", "null publisher"],
)
def test_the_server_refuses_a_config_that_does_not_publish_events(tmp_path, config):
    with pytest.raises(ValueError, match="must enable KV cache events"):
        Server(server_config(tmp_path, kv_events_config=config))


# --- Start and restart -------------------------------------------------------------------


def run_and_stop_with_chunks(start_tier, make, ids: list[int], config):
    """An instance that stores the chunks and stops cleanly; returns the pool's region."""
    tier = start_tier(kv_events_config=config)
    region = make(tier.address).attach("engine-a", rank=0).region
    store(attach_scheduler(make, tier.address), "r1", ids)
    tier.stop()
    return region


def publishes(publisher: RecordingPublisher) -> list[list]:
    """The events of each batch the publisher was given."""
    return [events for call, events in publisher.calls if call == "publish"]


def test_a_clean_restart_publishes_nothing_at_start(start_tier, processes, recording_config):
    run_and_stop_with_chunks(start_tier, processes, [0, 1, 2], recording_config)
    publisher = start_tier(kv_events_config=recording_config).server.event_publisher
    assert publishes(publisher) == []


def test_a_restored_chunk_is_removed_with_the_hashes_it_was_stored_with(start_tier, processes, recording_config):
    run_and_stop_with_chunks(start_tier, processes, list(range(POOL_CHUNKS)), recording_config)
    tier = start_tier(kv_events_config=recording_config)
    store(attach_scheduler(processes, tier.address), "r2", [POOL_CHUNKS])  # evicts the oldest restored chunk
    tier.stop()
    assert tier.server.event_publisher.calls == [
        ("publish", [removed(0)]),  # the eviction made by prepare_store
        ("publish", [stored(POOL_CHUNKS)]),
        ("shutdown", []),
    ]


def test_a_chunk_that_is_not_readable_when_the_server_stops_is_not_restored_with_an_event(
    start_tier, processes, recording_config
):
    tier = start_tier(kv_events_config=recording_config)
    scheduler = attach_scheduler(processes, tier.address, with_worker=True)
    store(scheduler, "r1", [0])
    scheduler.call(PrepareStore, req_id="r2", keys=[key(1)], positions=[None])  # a write that never completes
    tier.stop()

    restarted = start_tier(kv_events_config=recording_config)
    assert publishes(restarted.server.event_publisher) == []
    assert list(restarted.server.pool.block_stored.values()) == [stored(0)]


def test_a_write_pending_at_the_stop_publishes_its_store_when_it_completes_after_the_restart(
    start_tier, processes, recording_config
):
    tier = start_tier(kv_events_config=recording_config)
    scheduler = attach_scheduler(processes, tier.address, with_worker=True)
    reply = scheduler.call(PrepareStore, req_id="r1", keys=[key(1)], positions=[None])
    tier.stop()

    tier = start_tier(kv_events_config=recording_config)
    again = attach_scheduler(processes, tier.address)
    again.send(CompleteStore(keys=reply.keys_to_store, success=True, events=[block_stored(1)]))
    stats(again)
    tier.stop()
    assert tier.server.event_publisher.calls == [("publish", [stored(1)]), ("shutdown", [])]


def remove_the_snapshot(region_dir, region):
    os.unlink(os.path.join(region_dir, SNAPSHOT_FILE_NAME))


def garble_the_snapshot(region_dir, region):
    with open(os.path.join(region_dir, SNAPSHOT_FILE_NAME), "wb") as f:
        f.write(b"not a snapshot")


def move_the_snapshot_to_another_region(region_dir, region):
    snapshot = take_snapshot(region_dir)
    write_snapshot(region_dir, msgspec.structs.replace(snapshot, region_id=b"another region"))


def damage_the_region_files(region_dir, region):
    os.truncate(region.segments[0][0], 0)


@pytest.mark.parametrize(
    "spoil",
    [remove_the_snapshot, garble_the_snapshot, move_the_snapshot_to_another_region, damage_the_region_files],
)
def test_a_start_that_does_not_use_the_snapshot_publishes_only_a_clear(
    start_tier, processes, recording_config, tmp_path, spoil
):
    region = run_and_stop_with_chunks(start_tier, processes, [0, 1], recording_config)
    spoil(str(tmp_path), region)
    publisher = start_tier(kv_events_config=recording_config).server.event_publisher
    assert publishes(publisher) == [[AllBlocksCleared()]]


def test_a_start_with_a_cache_policy_that_is_not_snapshotted_publishes_only_a_clear(
    start_tier, processes, recording_config
):
    run_and_stop_with_chunks(start_tier, processes, [0, 1], recording_config)  # LRU
    publisher = start_tier(kv_events_config=recording_config, cache_policy="arc").server.event_publisher
    assert publishes(publisher) == [[AllBlocksCleared()]]


def test_a_fresh_start_publishes_only_a_clear(start_tier, recording_config):
    publisher = start_tier(kv_events_config=recording_config).server.event_publisher
    assert publishes(publisher) == [[AllBlocksCleared()]]


def replay(endpoint: str, expected_events: int, timeout_ms: int = 5_000) -> list:
    """Ask the replay endpoint for every batch from sequence number 0, again until
    it returns ``expected_events`` events; return them."""
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    end = (-1).to_bytes(8, "big", signed=True)
    ctx = zmq.Context()
    try:
        for _ in range(100):
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(endpoint)
            dealer.send_multipart([b"", (0).to_bytes(8, "big")])
            events: list = []
            while dealer.poll(timeout_ms):
                _, topic, seq, payload = dealer.recv_multipart()
                if seq == end:
                    break
                assert topic.decode() == TOPIC
                events += decoder.decode(payload).events
            dealer.close(linger=0)
            if len(events) >= expected_events:
                return events
            time.sleep(0.05)  # the publisher thread has not sent the batch yet
        raise TimeoutError(f"replayed {len(events)} of {expected_events} events")
    finally:
        ctx.destroy(linger=0)


@pytest.fixture
def replaying_config(tmp_path) -> KVEventsConfig:
    """vLLM's ZMQ publisher on ``TOPIC``, with a replay endpoint."""
    return KVEventsConfig(
        enable_kv_cache_events=True,
        endpoint=f"ipc://{tmp_path}/events.sock",
        replay_endpoint=f"ipc://{tmp_path}/replay.sock",
        topic=TOPIC,
    )


def test_a_subscriber_that_replays_gets_the_clear_of_a_cold_start_over_zmq(start_tier, replaying_config):
    start_tier(kv_events_config=replaying_config)
    assert replay(replaying_config.replay_endpoint, 1) == [AllBlocksCleared()]


def test_a_subscriber_that_replays_sees_nothing_for_a_clean_restart_until_a_removal_over_zmq(
    start_tier, processes, replaying_config
):
    run_and_stop_with_chunks(start_tier, processes, list(range(POOL_CHUNKS)), replaying_config)
    tier = start_tier(kv_events_config=replaying_config)
    store(attach_scheduler(processes, tier.address), "r2", [POOL_CHUNKS])  # evicts the oldest restored chunk
    # The first run's batches are not in this instance's buffer: its first is the removal.
    assert replay(replaying_config.replay_endpoint, 2) == [removed(0), stored(POOL_CHUNKS)]


# --- Command line -----------------------------------------------------------------------


CLI_BASE = ["--pool-bytes", "1", "--address", "ipc:///nowhere", "--region-dir", "/nowhere"]
ENDPOINT = ["--kv-events-endpoint", "ipc:///nowhere/events.sock"]
TOPIC_FLAGS = ["--kv-events-topic", TOPIC]


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ENDPOINT,
        TOPIC_FLAGS,
        [*ENDPOINT, "--kv-events-topic", ""],
        ["--kv-events-endpoint", "", *TOPIC_FLAGS],
        ["--kv-events-replay-endpoint", "ipc:///nowhere/replay.sock"],
    ],
    ids=[
        "no events flags",
        "endpoint without topic",
        "topic without endpoint",
        "empty topic",
        "empty endpoint",
        "replay without endpoint and topic",
    ],
)
def test_the_cli_refuses_a_missing_or_empty_endpoint_or_topic(flags):
    with pytest.raises(SystemExit):
        parse_args([*CLI_BASE, *flags])


def test_the_events_flags_build_the_kv_events_config(tmp_path):
    parsed, _ = parse_args(
        [
            *CLI_BASE,
            "--kv-events-endpoint",
            f"ipc://{tmp_path}/events.sock",
            "--kv-events-topic",
            TOPIC,
            "--kv-events-replay-endpoint",
            f"ipc://{tmp_path}/replay.sock",
            "--kv-events-buffer-steps",
            "7",
        ]
    )
    config = parsed.kv_events_config
    assert (
        config.enable_kv_cache_events,
        config.endpoint,
        config.topic,
        config.replay_endpoint,
        config.buffer_steps,
    ) == (
        True,
        f"ipc://{tmp_path}/events.sock",
        TOPIC,
        f"ipc://{tmp_path}/replay.sock",
        7,
    )
