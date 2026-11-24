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

"""``Client`` against a scripted pool server: a ROUTER that answers each request
with what the test's handler returns and records every message it gets."""

import threading
import time
from collections.abc import Callable

import msgspec
import pytest
import zmq

from kvpoold.api.v1 import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompatDescriptor,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromServer,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    NoSpace,
    PrepareLoad,
    PrepareStore,
    Refused,
    ReleaseHits,
    Request,
    RequestFinished,
    ShmFileRegion,
    StoreSpec,
    ToServer,
    UnknownClient,
)
from kvpoold.client import Client

COMPAT = CompatDescriptor(layout=1 << 20, namespace=b"\x11" * 8, hash_function="sha256", hash_seed="0")
REGION = ShmFileRegion(region_id=b"r", num_chunks=4, kv_bytes_per_chunk=1 << 20, segments=[("/x", 0, 4 << 20)])
KEY_A = b"a" * 36
KEY_B = b"b" * 36


def serve_everything(message: Request) -> FromServer:
    """A server that accepts every attach, hits every lookup and stores every key."""
    match message:
        case Attach():
            return AttachOk(call_id=message.call_id, region=REGION)
        case Lookup():
            return LookupOk(call_id=message.call_id, result="hit")
        case PrepareLoad():
            return LoadSpec(call_id=message.call_id, chunk_ids=list(range(len(message.keys))))
        case PrepareStore():
            return StoreSpec(
                call_id=message.call_id, keys_to_store=message.keys, chunk_ids=list(range(len(message.keys)))
            )
    raise AssertionError(f"unexpected {message!r}")


class FakeServer:
    def __init__(self, address: str) -> None:
        self.handler: Callable[[Request], FromServer | None] = serve_everything
        """Returns the reply to a request, or None to leave it unanswered."""
        self.received: list[ToServer] = []
        self._ctx = zmq.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.bind(address)
        self._decoder = msgspec.msgpack.Decoder(ToServer)
        self._encoder = msgspec.msgpack.Encoder()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            if not self._router.poll(10, zmq.POLLIN):
                continue
            identity, _, payload = self._router.recv_multipart()
            message = self._decoder.decode(payload)
            self.received.append(message)
            if isinstance(message, Request):
                reply = self.handler(message)
                if reply is not None:
                    self._router.send_multipart([identity, b"", self._encoder.encode(reply)])

    def of_type(self, message_type: type, timeout_s: float = 5.0) -> list:
        """The messages of ``message_type`` received so far, waiting up to
        ``timeout_s`` for the first."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            found = [m for m in self.received if isinstance(m, message_type)]
            if found:
                return found
            time.sleep(0.01)
        return []

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._router.close()
        self._ctx.term()


@pytest.fixture
def address(tmp_path) -> str:
    return f"ipc://{tmp_path}/pool.sock"


@pytest.fixture
def server(address):
    server = FakeServer(address)
    yield server
    server.close()


@pytest.fixture
def make_client(address):
    made: list[Client] = []

    def make(**kwargs) -> Client:
        kwargs.setdefault("attach_timeout_ms", 2000)
        kwargs.setdefault("call_timeout_ms", 200)
        client = Client(address, "engine-a", COMPAT, **kwargs)
        made.append(client)
        return client

    yield make
    for client in made:
        client.close()


@pytest.fixture
def scheduler(server, make_client) -> Client:
    client = make_client()
    assert client.attach(rank=None) == REGION
    return client


# --- Attach -----------------------------------------------------------------------


def test_attach_sends_the_engine_id_rank_version_and_compat_descriptor(server, make_client):
    make_client().attach(rank=3)
    [attach] = server.of_type(Attach)
    assert (attach.client_id, attach.rank, attach.version, attach.compat) == ("engine-a", 3, PROTOCOL_VERSION, COMPAT)


def test_a_worker_gets_the_region_and_is_not_an_attached_scheduler(server, make_client):
    client = make_client()
    assert client.attach(rank=0) == REGION
    assert not client.attached


def test_a_scheduler_is_attached_after_its_attach_is_accepted(scheduler):
    assert scheduler.attached


def test_a_refused_attach_returns_none_and_leaves_the_client_detached(server, make_client):
    server.handler = lambda m: Refused(call_id=m.call_id, reason="no")
    client = make_client()
    assert client.attach(rank=None) is None
    assert not client.attached


def test_an_attach_with_no_server_returns_none_without_waiting_for_the_timeout(make_client):
    client = make_client(attach_timeout_ms=200)
    start = time.monotonic()
    assert client.attach(rank=None) is None
    assert time.monotonic() - start < 1.0


# --- Loads ------------------------------------------------------------------------


def test_a_detached_client_misses_without_asking_the_server(server, make_client):
    assert make_client().lookup("req-1", KEY_A) == "miss"
    assert server.of_type(Lookup, timeout_s=0.2) == []


def test_hits_not_loaded_are_released_at_the_end_of_the_step(server, scheduler):
    assert scheduler.lookup("req-1", KEY_A) == "hit"
    assert scheduler.lookup("req-2", KEY_A) == "hit"
    assert scheduler.lookup("req-1", KEY_B) == "hit"
    assert scheduler.prepare_load("req-1", [KEY_B], [16]) == [0]
    scheduler.end_step()
    [release] = server.of_type(ReleaseHits)
    assert release.hits == [(KEY_A, 2)]


def test_a_step_with_every_hit_loaded_releases_nothing(server, scheduler):
    scheduler.lookup("req-1", KEY_A)
    scheduler.prepare_load("req-1", [KEY_A], [None])
    scheduler.end_step()
    scheduler.complete_load([KEY_A])
    server.of_type(CompleteLoad)
    assert server.of_type(ReleaseHits, timeout_s=0.1) == []


def test_a_load_of_a_key_that_was_not_a_hit_this_step_is_a_bug(scheduler):
    with pytest.raises(AssertionError):
        scheduler.prepare_load("req-1", [KEY_A], [None])


def test_a_load_the_server_does_not_prepare_raises(server, scheduler):
    scheduler.lookup("req-1", KEY_A)
    server.handler = lambda m: Refused(call_id=m.call_id, reason="not pinned")
    with pytest.raises(RuntimeError, match="did not prepare the load"):
        scheduler.prepare_load("req-1", [KEY_A], [None])


def test_prepare_load_sends_the_keys_with_their_positions(server, scheduler):
    scheduler.lookup("req-1", KEY_A)
    scheduler.lookup("req-1", KEY_B)
    scheduler.prepare_load("req-1", [KEY_A, KEY_B], [16, None])
    [load] = server.of_type(PrepareLoad)
    assert (load.req_id, load.keys, load.positions) == ("req-1", [KEY_A, KEY_B], [16, None])


# --- Stores -----------------------------------------------------------------------


def test_prepare_store_returns_the_keys_to_store_and_their_chunks(scheduler):
    assert scheduler.prepare_store("req-1", [KEY_A, KEY_B], [16, 32]) == ([KEY_A, KEY_B], [0, 1])


def test_a_store_without_space_is_skipped(server, scheduler):
    server.handler = lambda m: NoSpace(call_id=m.call_id)
    assert scheduler.prepare_store("req-1", [KEY_A], [16]) is None
    assert scheduler.attached


def test_a_detached_client_skips_every_store(server, make_client):
    assert make_client().prepare_store("req-1", [KEY_A], [16]) is None
    assert server.of_type(PrepareStore, timeout_s=0.2) == []


def test_complete_store_and_request_finished_are_notices(server, scheduler):
    scheduler.complete_store([KEY_A], success=False)
    scheduler.request_finished("req-1")
    [store] = server.of_type(CompleteStore)
    assert (store.keys, store.success, store.events) == ([KEY_A], False, None)
    [finished] = server.of_type(RequestFinished)
    assert finished.req_id == "req-1"


# --- Detaching and attaching again ------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [lambda m: UnknownClient(call_id=m.call_id), lambda m: Draining(call_id=m.call_id), lambda m: None],
    ids=["unknown client", "draining", "no answer"],
)
def test_a_server_that_lost_the_engine_detaches_the_client(server, scheduler, answer):
    server.handler = answer
    assert scheduler.lookup("req-1", KEY_A) == "miss"
    assert not scheduler.attached


def test_a_detached_client_attaches_again_at_the_end_of_a_step(server, scheduler):
    server.handler = lambda m: UnknownClient(call_id=m.call_id)
    scheduler.lookup("req-1", KEY_A)
    server.handler = serve_everything
    scheduler.end_step()
    assert scheduler.attached
    assert len(server.of_type(Attach)) == 2


# --- Close ------------------------------------------------------------------------


def test_closing_an_attached_scheduler_sends_leave(server, make_client):
    client = make_client()
    client.attach(rank=None)
    client.close()
    assert len(server.of_type(Leave)) == 1


def test_closing_a_worker_sends_no_leave(server, make_client):
    client = make_client()
    client.attach(rank=0)
    client.close()
    assert server.of_type(Leave, timeout_s=0.3) == []
