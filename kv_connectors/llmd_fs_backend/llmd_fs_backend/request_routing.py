# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Package-local request-route lifecycle kernel for optional TTL namespaces.

Opaque generation tokens must travel with accepted jobs. Disjoint TTL namespaces
avoid cross-policy renewal. This owns no files, quota, native I/O or deployment.
"""

from dataclasses import dataclass, field
from itertools import count
from threading import RLock

MISSING = object()


@dataclass(frozen=True)
class Route:
    request_id: str
    generation: int
    seconds: int | None  # None means decline storage, not infinite lifetime.

    @property
    def namespace(self):
        return None if self.seconds is None else f".llmd-ttl-v1-{self.seconds}"


@dataclass
class _State:
    route: Route
    finished: bool = False
    jobs: set = field(default_factory=set)


class RequestPolicyRoutes:
    def __init__(self, allowed, default):
        allowed = tuple(allowed)
        if not allowed or any(
            type(x) is not int or not 0 < x <= 2147483647 for x in allowed
        ):
            raise ValueError("Explicit positive integer lifetime allowlist required")
        if type(default) is not int or default not in allowed:
            raise ValueError("Default must belong to allowlist")
        self.allowed, self.default = frozenset(allowed), default
        self._states = {}
        self._sequence = count(1)
        self._mutex = RLock()

    def start(self, request_id, hint=MISSING):
        if type(request_id) is not str or not request_id:
            raise ValueError("Nonempty server request ID required")
        with self._mutex:
            if request_id in self._states:
                raise RuntimeError("Prior route must fully drain before ID reuse")
            seconds = self.default if hint is MISSING else hint
            if type(seconds) is not int or seconds not in self.allowed:
                seconds = None  # Invalid client hints decline storage, not serving.
            route = Route(request_id, next(self._sequence), seconds)
            self._states[request_id] = _State(route)
            return route

    def _state(self, route):
        state = self._states.get(route.request_id)
        if state is None or state.route != route:
            raise RuntimeError("Stale or foreign request generation")
        return state

    def policy(self, route):
        with self._mutex:
            return self._state(route).route.seconds

    def accept(self, route, kind, job_id):
        if kind not in ("load", "store"):
            raise ValueError("Explicit load or store job required")
        with self._mutex:
            state = self._state(route)
            if state.finished or route.seconds is None:
                raise RuntimeError("Cannot accept storage after finish or decline")
            job = (kind, job_id)
            if job in state.jobs:
                raise RuntimeError("Duplicate accepted job")
            state.jobs.add(job)

    def complete(self, route, kind, job_id):
        with self._mutex:
            state = self._state(route)
            job = (kind, job_id)
            if job not in state.jobs:
                raise RuntimeError("Unmatched or duplicate job completion")
            state.jobs.remove(job)
            self._retire(state)

    def finish(self, route):
        with self._mutex:
            if route.request_id not in self._states:
                return  # Idempotent finish after complete retirement.
            state = self._state(route)
            state.finished = True
            self._retire(state)

    def _retire(self, state):
        if state.finished and not state.jobs:
            del self._states[state.route.request_id]

    def reset(self):
        with self._mutex:
            if self._states:
                raise RuntimeError("Requests and all accepted jobs must drain")

    def snapshot(self):
        with self._mutex:
            return dict(
                requests=len(self._states),
                jobs=sum(len(s.jobs) for s in self._states.values()),
                finished_waiting=sum(s.finished for s in self._states.values()),
            )


def validate_request_policies(allowed, default):
    """Validate the server allowlist; client hints never extend this set."""
    if not isinstance(allowed, (list, tuple)):
        raise ValueError("storage_ttl_allowlist must be a nonempty integer list")
    routes = RequestPolicyRoutes(allowed, default)
    if len(routes.allowed) != len(allowed):
        raise ValueError("storage_ttl_allowlist must not contain duplicates")
    return tuple(allowed)
