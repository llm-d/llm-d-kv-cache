# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-request TTL routing; capacity control stays with the PVC evictor.

No private quota, LRU, prediction, or shared-file max-deadline policy is added.
Each allowed lifetime owns independent copies. Internal path aliases use only
trusted server mappers; a request route remains until all accepted jobs drain.
"""

from collections import Counter, defaultdict, deque
from itertools import count

from vllm.v1.kv_offload.base import OffloadingManager, RequestOffloadingContext

from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.request_routing import MISSING, RequestPolicyRoutes

_ROUTE = "_llmd_storage_ttl_route"


class RequestStorageTTLManager(OffloadingManager):
    def __init__(self, worker_mapper, policy_mappers, extra_config, default):
        self.worker_mapper = worker_mapper
        self.routes = RequestPolicyRoutes(policy_mappers, default)
        self._tokens = {}
        self._jobs = defaultdict(lambda: defaultdict(lambda: defaultdict(deque)))
        self._sequence = count(1)
        self._managers = {
            seconds: SharedStorageOffloadingManager(
                mapper, dict(extra_config, storage_ttl_seconds=seconds)
            )
            for seconds, mapper in policy_mappers.items()
        }

    def on_new_request(self, req_context):
        token = self.routes.start(
            req_context.req_id,
            (req_context.kv_transfer_params or {}).get(
                "llmd_storage_ttl_seconds", MISSING
            ),
        )
        self._tokens[req_context.req_id] = token
        # ReqContext is scheduler-owned and kept for late completions in vLLM.
        # Bind the generation here so an old context cannot complete a reused ID.
        setattr(req_context, _ROUTE, token)
        if token.seconds is None:
            return RequestOffloadingContext()
        return self._managers[token.seconds].on_new_request(req_context)

    def _route(self, ctx):
        token = getattr(ctx, _ROUTE, None)
        if token is None or self._tokens.get(ctx.req_id) != token:
            raise RuntimeError("Stale or unregistered TTL request context")
        self.routes.policy(token)
        return token, self._managers.get(token.seconds)

    def _retire(self, token):
        try:
            self.routes.policy(token)
        except RuntimeError:
            self._tokens.pop(token.request_id, None)
            self._jobs.pop(token.request_id, None)

    def lookup(self, key, req_context):
        _, manager = self._route(req_context)
        return False if manager is None else manager.lookup(key, req_context)

    def prepare_load(self, keys, req_context):
        keys = list(keys)
        token, manager = self._route(req_context)
        if manager is None:
            raise RuntimeError("Storage was declined for this request")
        self.routes.accept(token, "load", job := next(self._sequence))
        try:
            spec = manager.prepare_load(keys, req_context)
        except BaseException:
            self.routes.complete(token, "load", job)
            raise
        self._jobs[req_context.req_id]["load"][frozenset(keys)].append(job)
        return SharedStorageLoadStoreSpec(
            keys,
            source_paths={
                self.worker_mapper.get_file_name(k): spec.source_paths[
                    manager.file_mapper.get_file_name(k)
                ]
                for k in keys
            },
        )

    def complete_load(self, keys, req_context):
        keys = list(keys)
        token, manager = self._route(req_context)
        pending = self._jobs[req_context.req_id]["load"].get(frozenset(keys))
        if not pending:
            raise RuntimeError("Unmatched TTL load completion")
        manager.complete_load(keys, req_context)
        self.routes.complete(token, "load", pending.popleft())
        self._retire(token)

    def prepare_store(self, keys, req_context):
        token, manager = self._route(req_context)
        if manager is None:
            return None
        # Reject after-finish work before delegating, without altering any mapper.
        self.routes.accept(token, "store", guard := next(self._sequence))
        try:
            answer = manager.prepare_store(keys, req_context)
            if answer is not None:
                for key in answer.keys_to_store:
                    job = next(self._sequence)
                    self.routes.accept(token, "store", job)
                    self._jobs[req_context.req_id]["store"][key].append(job)
                answer.store_spec.source_paths = {
                    self.worker_mapper.get_file_name(
                        k
                    ): manager.file_mapper.get_file_name(k)
                    for k in answer.keys_to_store
                }
            return answer
        finally:
            self.routes.complete(token, "store", guard)

    def complete_store(self, keys, req_context, success=True):
        keys = list(keys)
        token, manager = self._route(req_context)
        pending = self._jobs[req_context.req_id]["store"]
        if any(len(pending.get(k, ())) < n for k, n in Counter(keys).items()):
            raise RuntimeError("Unmatched TTL store completion")
        manager.complete_store(keys, req_context, success=success)
        for key in keys:
            self.routes.complete(token, "store", pending[key].popleft())
        self._retire(token)

    def touch(self, keys, req_context):
        _, manager = self._route(req_context)
        if manager is not None:
            manager.touch(keys, req_context)

    def on_request_finished(self, req_context):
        if req_context.req_id not in self._tokens:
            return
        token, manager = self._route(req_context)
        if manager is not None:
            manager.on_request_finished(req_context)
        self.routes.finish(token)
        self._retire(token)

    def take_events(self):
        for manager in self._managers.values():
            yield from manager.take_events()

    def reset_cache(self):
        self.routes.reset()
        for manager in self._managers.values():
            manager.reset_cache()

    def shutdown(self):
        self.routes.reset()
        for manager in self._managers.values():
            manager.shutdown()
