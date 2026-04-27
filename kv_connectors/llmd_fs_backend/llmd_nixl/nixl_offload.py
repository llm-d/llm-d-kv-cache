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

"""Storage Offload Engine for managing asynchronous GPU-Storage transfers."""

import time
from abc import ABC, abstractmethod
from typing import List, NamedTuple
import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlBackendError
from nixl.logging import get_logger
from vllm.v1.kv_offload.worker.worker import TransferResult
from collections import deque
from typing import ClassVar, Literal


class TransferEntry(NamedTuple):
    job_id: int
    xfer_handle: object
    files_desc: object
    fd_list: list | None        # None for OBJ backend (no real FDs)
    stagings: list | None       # None for GDS backends
    read_block_ids: list | None  # None for WRITE; block_ids for READ


# ------------------------------------------------------------------ #
# Abstract base class                                                   #
# ------------------------------------------------------------------ #

class StorageOffloadEngine(ABC):
    """
    Abstract base class for GPU-Storage transfer engines.

    Subclasses implement the handful of backend-specific behaviours
    (opening files, building NIXL descriptors, staging GPU data).
    This class owns the NIXL agent, transfer queue, and all polling logic.
    """


    NixlMem = Literal["DRAM", "VRAM", "OBJ", "FILE"]
    nixl_source: ClassVar[NixlMem]
    nixl_dest:   ClassVar[NixlMem]

    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: List[torch.Tensor],
        backend: str,
    ):
        self.io_threads = io_threads
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.tensors = tensors
        self.backend = backend
        self.logger = get_logger(__name__)

        self._transfers: deque[TransferEntry] = deque()
        self._pending_results: list[TransferResult] = []
        self._block_size = tensors[0].stride(0) * tensors[0].element_size()

        agent_config = nixl_agent_config(backends=[])
        self.agent = nixl_agent("StorageOffloadEngine", agent_config)

        plugin_list = self.agent.get_plugin_list()
        if backend not in plugin_list:
            raise RuntimeError(f"{backend} plugin not available in NIXL")

        self.logger.info(
            "%s Plugin parameters:\n%s\n%s",
            backend,
            self.agent.get_plugin_mem_types(backend),
            self.agent.get_plugin_params(backend),
        )
        self.agent.create_backend(backend, self._backend_params())
        self.logger.info(
            "%s Backend parameters:\n%s",
            backend,
            self.agent.get_backend_mem_types(backend),
        )
        self.logger.info("Tensor[0] shape: %s", tensors[0].shape)

    # ------------------------------------------------------------------ #
    # Abstract interface - one small method per varying behaviour          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _backend_params(self) -> dict:
        """Return params dict for agent.create_backend()."""

    @abstractmethod
    def _get_staging_and_copy(self, block_ids: List) -> tuple:
        """Return (tensors, stagings) ready for a WRITE transfer."""

    @abstractmethod
    def _get_staging(self, block_ids: List) -> tuple:
        """Return (tensors, stagings) ready for a READ transfer."""

    @abstractmethod
    def _get_blocks_data(self, tensors: List[torch.Tensor], block_ids: List) -> list:
        """Return list of (addr, size, device_id) for NIXL xfer_descs."""

    @abstractmethod
    def _open_files(self, files: List[str]) -> list:
        """Open files; return fd_list (ints for file backends, strings for OBJ)."""

    @abstractmethod
    def _build_nixl_file_entry(
        self, fd_list: list, file_idx: int, intra_offset: int
    ) -> tuple:
        """Return one NIXL file descriptor tuple for a block.

        Args:
            fd_list:      file descriptors / S3 keys, one per file.
            file_idx:     index into fd_list identifying the file.
            intra_offset: position of this block within the file
                          (always 0 for OBJ since block size == GPU block size).
        """

    @abstractmethod
    def _close_fds(self, fd_list: list) -> None:
        """Close descriptors opened by _open_files."""

    @abstractmethod
    def _complete_read(self, stagings: list, block_ids: List) -> None:
        """Copy completed READ data from stagings to GPU (no-op for GDS)."""

    @abstractmethod
    def _shutdown_backend(self) -> None:
        """Release backend-specific resources."""

    def _sync_before_transfer(self) -> None:
        """Wait for any async D2H copies before posting to NIXL. No-op for GDS."""

    def _on_submit_error(self, _stagings) -> None:
        """Release staging resources on submit failure. No-op for non-staged backends."""

    # ------------------------------------------------------------------ #
    # Common transfer logic - no backend conditionals                      #
    # ------------------------------------------------------------------ #

    def _submit_transfer(
        self,
        job_id: int,
        tensors: List[torch.Tensor],
        stagings,
        files: List[str],
        block_ids: List,
        op: str,
    ) -> bool:
        fd_list = self._open_files(files)
        blocks_data = self._get_blocks_data(tensors, block_ids)
        assert blocks_data

        # Build one NIXL file entry per block, grouped by file.
        # file_idx indexes fd_list; intra_offset is the block's position within that file.
        nixl_files = []
        for file_idx, block_list in enumerate(block_ids):
            for intra_offset, _ in enumerate(block_list):
                nixl_files.append(self._build_nixl_file_entry(fd_list, file_idx, intra_offset))

        assert len(blocks_data) == len(nixl_files)
        xfer_desc = self.agent.get_xfer_descs(blocks_data, self.nixl_source)
        assert xfer_desc is not None

        files_desc = self.agent.register_memory(nixl_files, self.nixl_dest)
        assert files_desc is not None
        xfer_files = files_desc.trim()

        xfer_handle = self.agent.initialize_xfer(op, xfer_desc, xfer_files, "StorageOffloadEngine")
        if not xfer_handle:
            self.logger.error("initialize_xfer failed")
            self.agent.deregister_memory(files_desc)
            self._close_fds(fd_list)
            self._on_submit_error(stagings)
            return False

        self._sync_before_transfer()

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            self.logger.error("agent.transfer failed")
            self.agent.deregister_memory(files_desc)
            self._close_fds(fd_list)
            self._on_submit_error(stagings)
            return False

        self._transfers.append(TransferEntry(
            job_id=job_id,
            xfer_handle=xfer_handle,
            files_desc=files_desc,
            fd_list=fd_list,
            stagings=stagings,
            read_block_ids=block_ids if op == "READ" else None,
        ))
        return True

    def async_store_gpu_blocks(self, job_id: int, files: List[str], block_ids: List) -> bool:
        """ Store gpu kv cache blocks into storage (obj, posix, gds, whatever) """
        self.logger.debug("async_store_gpu_blocks in_flight=%d", len(self._transfers))
        tensors, stagings = self._get_staging_and_copy(block_ids)
        return self._submit_transfer(job_id, tensors, stagings, files, block_ids, "WRITE")

    def async_load_gpu_blocks(self, job_id: int, files: List[str], block_ids: List) -> bool:
        """ Load kv cache blocks from storage into gpu """
        self.logger.debug("async_load_gpu_blocks in_flight=%d", len(self._transfers))
        tensors, stagings = self._get_staging(block_ids)
        return self._submit_transfer(job_id, tensors, stagings, files, block_ids, "READ")

    def _complete_transfer(self, entry: TransferEntry) -> None:
        """Finalise a completed transfer: close FDs, release NIXL resources."""
        self._close_fds(entry.fd_list or [])
        self.agent.deregister_memory(entry.files_desc)
        self.agent.release_xfer_handle(entry.xfer_handle)

    def get_finished(self) -> List[TransferResult]:
        """Poll all in-flight transfers and return a list of completed (job_id, success) pairs."""
        results, self._pending_results = self._pending_results, [] # tuple swap
        to_remove = []

        for entry in self._transfers:
            state = self.agent.check_xfer_state(entry.xfer_handle)
            if state == "DONE":
                self.logger.debug("DONE job_id=%d", entry.job_id)
                self._complete_transfer(entry)
                results.append((entry.job_id, True))
                to_remove.append(entry)
            elif state == "PROC":
                continue
            else:
                self.logger.error("transfer failed job %d state=%s", entry.job_id, state)
                self._complete_transfer(entry)
                results.append((entry.job_id, False))
                to_remove.append(entry)

        for entry in to_remove:
            self._transfers.remove(entry)
        return results

    def wait_job(self, job_id: int) -> None:
        """Block until the specified job completes."""
        entry = next((e for e in self._transfers if e.job_id == job_id), None)
        if entry is None:
            self.logger.warning("wait_job: job %d not found (already completed?)", job_id)
            return
        i = 0
        while True:
            state = self.agent.check_xfer_state(entry.xfer_handle)
            if state == "DONE":
                break
            elif state == "PROC":
                i += 1
                if i % 10 == 0:
                    self.logger.debug("wait_job %d iterations=%d", job_id, i)
                time.sleep(0.1)
            else:
                self.logger.error("wait_job: error state=%s job=%d", state, job_id)
                break

    def shutdown(self) -> None:
        self._shutdown_backend()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
