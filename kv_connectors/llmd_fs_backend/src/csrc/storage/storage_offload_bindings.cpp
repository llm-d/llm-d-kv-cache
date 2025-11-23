/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License,
 * Version 2.0 (the "License");
 * you may not use this file except in
 * compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by
 * applicable law or agreed to in writing, software
 * distributed under the
 * License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the
 * specific language governing permissions and
 * limitations under the
 * License.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "storage_offload.hpp"

namespace py = pybind11;
// Pybind11 bindings exposing the C++ StorageOffloadEngine for
// asynchronous KV-cache transfers between GPU memory and shared storage.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<StorageOffloadEngine>(m, "StorageOffloadEngine")
      .def(py::init<int,
                    size_t,
                    size_t,
                    int,
                    int,
                    std::vector<torch::Tensor>&,
                    bool,
                    bool,
                    int>(),
           py::arg("io_threads"),
           py::arg("staging_buffer_size_mb"),
           py::arg("max_staging_memory_gb"),
           py::arg("tp_rank"),
           py::arg("gpu_blocks_per_file"),
           py::arg("tensors"),
           py::arg("kv_before_blocks"),
           py::arg("layers_before_blocks"),
           py::arg("num_blocks_dimension"))

      .def("get_finished", &StorageOffloadEngine::get_finished)

      .def("transfer_async_put",
           &StorageOffloadEngine::transfer_async_put,
           py::arg("job_id"),
           py::arg("dst_files"),
           py::arg("src_tensors"),
           py::arg("all_block_ids"))

      .def("transfer_async_get",
           &StorageOffloadEngine::transfer_async_get,
           py::arg("job_id"),
           py::arg("src_files"),
           py::arg("all_block_ids"),
           py::arg("dst_tensors"))
      .def("wait_job", &StorageOffloadEngine::wait_job, py::arg("job_id"));
}
