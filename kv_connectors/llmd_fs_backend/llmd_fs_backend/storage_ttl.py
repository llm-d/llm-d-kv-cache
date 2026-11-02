# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-only fixed lifetime protocol; no vLLM or Torch imports.

A policy-versioned namespace fixes TTL independently of reader atime updates.
Readers pin the inode with an open descriptor; a same-container native worker
reads through /proc/<scheduler-pid>/fd/<fd>. Unlink/replace cannot break an
accepted load. Writers and deleters serialize pathname changes with stable
bucket POSIX record locks. Locks are never removed by directory cleanup.
"""

import fcntl
import os
import sys
import time
from pathlib import Path

PREFIX = ".llmd-ttl-v1-"
NANOSECONDS = 1_000_000_000


def validate_seconds(value):
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 2_147_483_647:
        raise ValueError("storage_ttl_seconds must be an integer in [1, 2147483647]")
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        raise ValueError("storage TTL requires Linux procfs descriptor access")
    return value


def policy_for_path(path):
    path = Path(path)
    for parent in path.parents:
        if parent.name.startswith(PREFIX):
            suffix = parent.name[len(PREFIX) :]
            if not suffix.isdecimal() or not 1 <= int(suffix) <= 2_147_483_647:
                raise ValueError("Invalid reserved TTL namespace")
            return parent, int(suffix)
    return None


def lock_path(path):
    policy = policy_for_path(path)
    if policy is None:
        raise ValueError("Not a TTL namespace")
    folder = policy[0] / ".locks"
    folder.mkdir(exist_ok=True)
    bucket = Path(path).stem[:3]
    if len(bucket) != 3 or any(c not in "0123456789abcdef" for c in bucket):
        raise ValueError("Invalid TTL block filename")
    return folder / (bucket + ".lock")


def try_delete(path, *, pressure, idle_seconds, dry_run=False):
    """Return deleted stat, or None if not eligible/busy/missing.

    Deadline is rechecked under the same exclusive lock as publication. An
    admitted reader's FD keeps old data valid after this pathname is removed.
    """
    policy = policy_for_path(path)
    if policy is None:
        return None
    descriptor = os.open(lock_path(path), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            stat = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return None
        now = time.time_ns()
        expired = now >= stat.st_mtime_ns + policy[1] * NANOSECONDS
        if not expired and not (
            pressure and now - stat.st_atime_ns >= idle_seconds * NANOSECONDS
        ):
            return None
        if not dry_run:
            os.unlink(path)
        return stat
    finally:
        os.close(descriptor)


class InodePins:
    """Request lifetime pins, including speculative lookups and late completions."""

    def __init__(self, seconds):
        self.seconds = validate_seconds(seconds)
        self.requests = {}
        self.counters = dict(opened=0, expired=0, misses=0)

    def lookup(self, request_id, key, path):
        state = self.requests.setdefault(
            request_id, dict(current={}, fds=[], loads=0, finished=False)
        )
        old = state["current"].get(key)
        now = time.time_ns()
        if old is not None and now < old[1] + self.seconds * NANOSECONDS:
            return True
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            self.counters["misses"] += 1
            return False
        try:
            stat = os.fstat(fd)
            if time.time_ns() >= stat.st_mtime_ns + self.seconds * NANOSECONDS:
                self.counters["expired"] += 1
                os.close(fd)
                return False
            state["current"][key] = (fd, stat.st_mtime_ns)
            state["fds"].append(fd)
            self.counters["opened"] += 1
            return True
        except BaseException:
            os.close(fd)
            raise

    def prepare_load(self, request_id, key_paths):
        state = self.requests[request_id]
        if state["finished"]:
            raise RuntimeError("Load submitted after request finished")
        result = {}
        for key, path in key_paths:
            fd = state["current"][key][0]
            result[path] = f"/proc/{os.getpid()}/fd/{fd}"
        state["loads"] += 1
        return result

    def complete_load(self, request_id):
        state = self.requests[request_id]
        if state["loads"] <= 0:
            raise RuntimeError("Unexpected TTL load completion")
        state["loads"] -= 1
        if state["finished"] and state["loads"] == 0:
            self._close(request_id)

    def finish(self, request_id):
        state = self.requests.get(request_id)
        if state is not None:
            state["finished"] = True
            if state["loads"] == 0:
                self._close(request_id)

    def _close(self, request_id):
        state = self.requests.pop(request_id)
        for fd in state["fds"]:
            os.close(fd)

    def close(self):
        for request_id in list(self.requests):
            self._close(request_id)
