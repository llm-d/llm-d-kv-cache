# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client for the kvtierd Unix socket.

Frames are a 4-byte big-endian length followed by one msgpack map. The
daemon passes its pool's file descriptor with SCM_RIGHTS on the hello
response; the client maps it so that block copies need no socket traffic.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import mmap
import os
import select
import socket
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import msgspec

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
MAX_FRAME = 256 << 20

# Per-key outcome of an alloc request.
ALLOCATED = 0
PRESENT = 1
NO_SPACE = 2

# Requests that only read. A timeout of any other request leaves the daemon's
# state unknown: it may have applied the request and sent a reply that came
# too late. The client then closes the connection, so the daemon drops
# whatever the session held, after its release grace period.
READ_ONLY_OPS = frozenset({"snapshot", "stats"})

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(dict)


class DaemonError(RuntimeError):
    """The daemon rejected a request, or the connection failed."""


class Arena:
    """A mapping of the daemon's shared-memory pool."""

    def __init__(self, fd: int, size: int, slot_bytes: int) -> None:
        try:
            # MAP_POPULATE builds the page tables now. Without it, the first
            # copy through every 4 KiB page takes a fault, which halves copy
            # bandwidth on a fresh connection.
            flags = mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0)
            self._mm = mmap.mmap(fd, size, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self.size = size
        self.slot_bytes = slot_bytes
        # Keep the ctypes view alive: it pins the mapping's address.
        self._anchor = (ctypes.c_char * size).from_buffer(self._mm)
        self.base = ctypes.addressof(self._anchor)

    def slot_address(self, slot: int) -> int:
        return self.base + slot * self.slot_bytes


class DaemonClient:
    """One connection to kvtierd.

    `call` is thread-safe. A reader thread routes responses to callers and
    hands pushed events to `on_event`. When the connection drops, pending
    calls fail and `on_disconnect` runs once.
    """

    def __init__(
        self,
        socket_path: str,
        namespace: str,
        slot_bytes: int,
        client_name: str,
        on_event: Callable[[dict[str, Any]], None],
        on_disconnect: Callable[[], None],
        timeout_s: float,
        hello_timeout_s: float = 120.0,
    ) -> None:
        self._timeout = timeout_s
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self._ids = itertools.count(2)
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._closed = False
        self._rx = 0

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # The first hello can wait while the daemon creates and prefaults its
        # pool, which takes seconds for a large pool.
        sock.settimeout(max(timeout_s, hello_timeout_s))
        try:
            sock.connect(socket_path)
            self._sock = sock
            hello, leftover = self._hello(namespace, slot_bytes, client_name)
        except BaseException:
            sock.close()
            raise
        sock.settimeout(None)
        self.session = hello["session"]
        self.arena: Arena = hello["arena"]
        self.num_slots: int = hello["num_slots"]
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(leftover,),
            name="llmd-kv-tiering-reader",
            daemon=True,
        )
        self._reader.start()

    def _hello(self, namespace: str, slot_bytes: int, client_name: str) -> tuple[dict[str, Any], bytes]:
        self._send(
            {
                "id": 1,
                "op": "hello",
                "version": PROTOCOL_VERSION,
                "ns": namespace,
                "slot_bytes": slot_bytes,
                "client": client_name,
            }
        )
        buf = bytearray()
        fds: list[int] = []
        while True:
            if len(buf) >= 4:
                (n,) = struct.unpack(">I", buf[:4])
                if len(buf) >= 4 + n:
                    break
            data, got_fds, _, _ = socket.recv_fds(self._sock, 1 << 20, 4)
            if not data:
                raise DaemonError("daemon closed the connection during hello")
            buf += data
            fds += got_fds
        (n,) = struct.unpack(">I", buf[:4])
        resp = _decoder.decode(bytes(buf[4 : 4 + n]))
        if resp.get("error"):
            for fd in fds:
                os.close(fd)
            raise DaemonError(resp["error"])
        if len(fds) != 1:
            for fd in fds:
                os.close(fd)
            raise DaemonError(f"expected 1 pool fd from hello, got {len(fds)}")
        resp["arena"] = Arena(fds[0], resp["arena_bytes"], resp["slot_bytes"])
        return resp, bytes(buf[4 + n :])

    def _send(self, msg: dict[str, Any], timeout_s: float | None = None) -> None:
        """Send one frame. Waiting for the send lock and for socket buffer
        space share one deadline. A frame that is cut off part way leaves
        the stream unusable, so callers close the connection on any error."""
        body = _encoder.encode(msg)
        data = memoryview(struct.pack(">I", len(body)) + body)
        deadline = time.monotonic() + (self._timeout if timeout_s is None else timeout_s)
        if not self._send_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError("timed out waiting to send")
        try:
            poller = select.poll()
            poller.register(self._sock, select.POLLOUT)
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out sending")
                if not poller.poll(remaining * 1000):
                    continue
                try:
                    n = self._sock.send(data, socket.MSG_DONTWAIT)
                except BlockingIOError:
                    continue
                data = data[n:]
        finally:
            self._send_lock.release()

    def call(self, op: str, **fields: Any) -> dict[str, Any]:
        """Send a request and wait for its response."""
        fut: Future = Future()
        with self._lock:
            if self._closed:
                raise DaemonError("not connected")
            req_id = next(self._ids)
            self._pending[req_id] = fut
        try:
            self._send({"id": req_id, "op": op, **fields})
        except (OSError, TimeoutError) as e:
            with self._lock:
                self._pending.pop(req_id, None)
            self._fail(e)
            raise DaemonError(f"send failed: {e}") from e
        try:
            resp = fut.result(timeout=self._timeout)
        except TimeoutError as e:
            with self._lock:
                self._pending.pop(req_id, None)
            if op not in READ_ONLY_OPS:
                self._fail(DaemonError(f"{op} timed out"))
            raise DaemonError(f"{op} timed out") from e
        if resp.get("error"):
            raise DaemonError(resp["error"])
        return resp

    def _read_loop(self, leftover: bytes) -> None:
        buf = bytearray(leftover)
        try:
            while True:
                while len(buf) >= 4:
                    (n,) = struct.unpack(">I", buf[:4])
                    if n > MAX_FRAME:
                        raise DaemonError(f"frame of {n} bytes exceeds limit")
                    if len(buf) < 4 + n:
                        break
                    msg = _decoder.decode(bytes(buf[4 : 4 + n]))
                    del buf[: 4 + n]
                    self._route(msg)
                data = self._sock.recv(1 << 20)
                if not data:
                    raise DaemonError("daemon closed the connection")
                buf += data
        except Exception as e:  # noqa: BLE001 - any failure ends the session
            self._fail(e)

    def _route(self, msg: dict[str, Any]) -> None:
        # The receive order of every event and reply, so a caller can tell
        # which events came after a given reply.
        self._rx += 1
        msg["_rx"] = self._rx
        req_id = msg.get("id", 0)
        if req_id == 0:
            try:
                self._on_event(msg)
            except Exception:
                logger.exception("event handler failed")
            return
        with self._lock:
            fut = self._pending.pop(req_id, None)
        if fut is not None:
            fut.set_result(msg)

    def _fail(self, err: BaseException) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            fut.set_exception(DaemonError(f"connection lost: {err}"))
        with contextlib.suppress(OSError):
            self._sock.close()
        self._on_disconnect()

    @property
    def connected(self) -> bool:
        return not self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._reader.join(timeout=self._timeout)
