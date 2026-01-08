/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "storage_offload.hpp"

namespace py = pybind11;
// Pybind11 bindings exposing the C++ StorageOffloadEngine for
// asynchronous KV-cache transfers between GPU memory and shared storage.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<StorageOffloadEngine>(
      m,
      "StorageOffloadEngine",
      "Engine for asynchronous KV-cache offloading between GPU memory "
      "and shared storage using background I/O threads.")
      .def(py::init<int, int, std::vector<torch::Tensor>&>(),
           py::arg("io_threads"),
           py::arg("gpu_blocks_per_file"),
           py::arg("tensors"),
           "Create a StorageOffloadEngine instance for asynchronous KV-cache "
           "transfers "
           "between GPU memory and shared storage. "
           "This initializes background I/O threads, per-thread CPU staging "
           "buffers, "
           "and dedicated CUDA streams for async copy operations. "
           "The current CUDA device must already be set by the caller, and all "
           "tensors "
           "are expected to reside on that device. "
           "CPU staging memory is NUMA-aware and prefers the NUMA node local "
           "to the GPU "
           "to improve data locality and performance.\n\n"
           "Args:\n"
           "  io_threads: Number of background I/O worker threads.\n"
           "  gpu_blocks_per_file: Number of GPU KV-cache blocks per file.\n"
           "  tensors: List of GPU tensors backing the KV-cache.")

      .def("get_finished",
           &StorageOffloadEngine::get_finished,
           "Return a list of finished job IDs and their success status.\n\n"
           "Each entry is a (job_id, success) tuple.")

      .def("async_store_gpu_blocks",
           &StorageOffloadEngine::async_store_gpu_blocks,
           py::arg("job_id"),
           py::arg("dst_files"),
           py::arg("all_block_ids"),
           "Asynchronously store GPU KV-cache blocks to shared storage.\n\n"
           "Args:\n"
           "  job_id: Identifier for the async job.\n"
           "  dst_files: Destination file paths.\n"
           "  all_block_ids: KV-cache block IDs per file.")

      .def("async_load_gpu_blocks",
           &StorageOffloadEngine::async_load_gpu_blocks,
           py::arg("job_id"),
           py::arg("src_files"),
           py::arg("all_block_ids"),
           "Asynchronously load KV-cache blocks from shared storage into "
           "GPU.\n\n"
           "Args:\n"
           "  job_id: Identifier for the async job.\n"
           "  src_files: Source file paths.\n"
           "  all_block_ids: KV-cache block IDs per file.")

      .def("wait_job",
           &StorageOffloadEngine::wait_job,
           py::arg("job_id"),
           "Block until all tasks for the given job ID have completed.");
}
