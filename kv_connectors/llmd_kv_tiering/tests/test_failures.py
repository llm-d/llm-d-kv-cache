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
"""Failure handling: timeouts, blocked sends, reconnect races, namespaces."""

from __future__ import annotations

import dataclasses
import os
import signal
import time

import pytest
from conftest import BLOCK, make_offloading_spec
from llmd_kv_tiering import vllm_tier
from llmd_kv_tiering.client import DaemonClient, DaemonError
from test_vllm_tier import CTX, job, key, make_tier, primary, run, wait_for


def connect(daemon, timeout_s=0.5) -> DaemonClient:
    return DaemonClient(
        str(daemon.socket_path),
        "ns",
        BLOCK,
        "test",
        on_event=lambda _m: None,
        on_disconnect=lambda: None,
        timeout_s=timeout_s,
    )


@pytest.fixture
def paused():
    """Pause a daemon process; resume it when the test ends."""
    stopped = []

    def pause(daemon):
        os.kill(daemon.proc.pid, signal.SIGSTOP)
        stopped.append(daemon)

    yield pause
    for d in stopped:
        os.kill(d.proc.pid, signal.SIGCONT)


def test_timed_out_pin_closes_the_connection(daemon_factory, paused):
    d = daemon_factory()
    c = connect(d)
    paused(d)
    # The daemon may apply the pin and answer late. The client cannot know,
    # so it closes the connection and the daemon drops the session's pins.
    with pytest.raises(DaemonError, match="timed out"):
        c.call("pin", keys=[b"k" * 12])
    assert not c.connected


def test_timed_out_snapshot_keeps_the_connection(daemon_factory, paused):
    d = daemon_factory()
    c = connect(d)
    paused(d)
    with pytest.raises(DaemonError, match="timed out"):
        c.call("snapshot")
    assert c.connected
    c.close()


def test_blocked_send_times_out(daemon_factory, paused):
    d = daemon_factory()
    c = connect(d)
    paused(d)
    # Far more than a Unix socket buffers, so the send blocks.
    keys = [i.to_bytes(12, "big") for i in range(400_000)]
    start = time.monotonic()
    with pytest.raises(DaemonError, match="send failed"):
        c.call("pin", keys=keys)
    assert time.monotonic() - start < 5
    assert not c.connected


def test_connection_lost_while_connecting(daemon_factory, tiers, monkeypatch):
    d = daemon_factory()
    real_call = DaemonClient.call
    dropped = []

    def call(self, op, **kw):
        resp = real_call(self, op, **kw)
        if op == "snapshot" and not dropped:
            # Drop the first connection right after its snapshot, before
            # the tier installs it.
            dropped.append(True)
            self._fail(DaemonError("injected"))
        return resp

    monkeypatch.setattr(DaemonClient, "call", call)
    arr = primary()
    tier = make_tier(d, arr)
    tiers.append(tier)
    assert dropped
    # The dead connection was not installed, so the tier reconnects.
    assert wait_for(lambda: tier.connected)
    assert run(tier, job(1, [key(1)], [0]), store=True).success


def test_namespace_covers_the_kv_layout():
    a = make_offloading_spec()
    b = make_offloading_spec()
    b.config = dataclasses.replace(b.config, kv_cache_layout="HND")
    c = make_offloading_spec()
    c.config = dataclasses.replace(c.config, kv_cache_layout="NHD")
    ns = [vllm_tier.derive_namespace(s, BLOCK) for s in (a, b, c)]
    assert len(set(ns)) == 3
    assert vllm_tier.derive_namespace(a, BLOCK) == ns[0]
    assert vllm_tier.derive_namespace(a, BLOCK, salt="rev-2") != ns[0]


def test_lookups_follow_events_for_own_stores(daemon_factory, tiers):
    d = daemon_factory(slots=1)
    arr = primary()
    tier = make_tier(d, arr)
    tiers.append(tier)
    assert run(tier, job(1, [key(1)], [0]), store=True).success
    assert key(1) in tier._mirror
    # Storing k2 evicts k1. The tier learns both from the daemon's events.
    assert run(tier, job(2, [key(2)], [1]), store=True).success
    assert wait_for(lambda: key(1) not in tier._mirror and key(2) in tier._mirror)
    events = list(tier.take_events())
    stored = [e for e in events if not e.removed]
    assert [list(e.keys) for e in stored] == [[key(1)], [key(2)]]
    assert all(e.req_context is CTX for e in stored)
    assert [list(e.keys) for e in events if e.removed] == [[key(1)]]
