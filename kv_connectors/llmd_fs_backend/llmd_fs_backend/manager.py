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

import os
from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from zmq import ZMQError

from llmd_fs_backend.event_publisher import StorageMedium
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.storage_ttl import InodePins, validate_seconds

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        extra_config: dict | None = None,
        event_publisher=None,
    ) -> None:
        self.file_mapper: FileMapper = file_mapper
        self._declined_ttl_requests: set[str] = set()
        self.storage_ttl_seconds = validate_seconds(
            (extra_config or {}).get("storage_ttl_seconds")
        )
        self._ttl_pins = (
            InodePins(self.storage_ttl_seconds) if self.storage_ttl_seconds else None
        )
        self._event_publisher = (
            event_publisher
            if event_publisher is not None
            else self._create_event_publisher(
                self.file_mapper.model_name, extra_config or {}
            )
        )

    @staticmethod
    def _create_event_publisher(model_name: str, extra_config: dict):
        """Create a StorageEventPublisher if events are enabled in *extra_config*."""
        if not extra_config.get("enable_events", False):
            return None

        endpoint = extra_config.get("storage_events_endpoint")
        if not endpoint:
            return None

        kwargs = {}
        if "storage_medium" in extra_config:
            kwargs["medium"] = StorageMedium(extra_config["storage_medium"])
        if "storage_events_hwm" in extra_config:
            kwargs["sndhwm"] = int(extra_config["storage_events_hwm"])

        try:
            from llmd_fs_backend.event_publisher import StorageEventPublisher

            return StorageEventPublisher(
                endpoint=endpoint,
                model_name=model_name,
                **kwargs,
            )
        except ZMQError:
            logger.warning(
                "failed to create storage event publisher for %s",
                endpoint,
                exc_info=True,
            )
            return None

    def _publish_blocks_stored(self, keys: Collection[OffloadKey]) -> None:
        if self._event_publisher is None:
            return
        try:
            block_hashes = [get_offload_block_hash(k) for k in keys]
            self._event_publisher.publish_blocks_stored(block_hashes)
        except Exception:
            logger.warning("failed to publish storage event", exc_info=True)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # This first version uses one fixed deadline policy per namespace.
        # A request may confirm that policy but cannot renew or change it.
        params = req_context.kv_transfer_params or {}
        hint = params.get("llmd_storage_ttl_seconds")
        if hint is not None:
            try:
                matches = validate_seconds(hint) == self.storage_ttl_seconds
            except ValueError:
                matches = False
            if not matches:
                # A backend callback exception can terminate vLLM's scheduler.
                # Decline this request's storage reuse/persistence instead.
                self._declined_ttl_requests.add(req_context.req_id)
                logger.warning(
                    "Declining storage for request %s: "
                    "TTL hint does not match namespace policy",
                    req_context.req_id,
                )
        return RequestOffloadingContext()

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready to be read.
        """
        if req_context.req_id in self._declined_ttl_requests:
            return False
        file_path = self.file_mapper.get_file_name(key)
        if self._ttl_pins is not None:
            return self._ttl_pins.lookup(req_context.req_id, key, file_path)
        return os.path.exists(file_path)

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        paths = None
        if self._ttl_pins is not None:
            paths = self._ttl_pins.prepare_load(
                req_context.req_id,
                [(k, self.file_mapper.get_file_name(k)) for k in keys],
            )
        return SharedStorageLoadStoreSpec(keys, source_paths=paths)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Release accepted loads after worker completion, including late callbacks."""
        if self._ttl_pins is not None:
            self._ttl_pins.complete_load(req_context.req_id)

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        if req_context.req_id in self._declined_ttl_requests:
            return None
        keys_to_store = list(keys)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],  # no eviction needed
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        For shared storage, storing is stateless but we emit events for stored blocks.
        """
        if success:
            self._publish_blocks_stored(keys)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._declined_ttl_requests.discard(req_context.req_id)
        if self._ttl_pins is not None:
            self._ttl_pins.finish(req_context.req_id)

    def reset_cache(self) -> None:
        if self._ttl_pins is not None:
            # vLLM discards old completion callbacks after reset, before the
            # asynchronous worker flush has necessarily run. Closing accepted
            # read FDs here would break them; keep reset explicit and safe.
            if any(s["loads"] for s in self._ttl_pins.requests.values()):
                raise RuntimeError("Wait for accepted TTL loads before cache reset")
            self._ttl_pins.close()

    def shutdown(self) -> None:
        if self._ttl_pins is not None:
            self._ttl_pins.close()
        if self._event_publisher is not None:
            self._event_publisher.close()
