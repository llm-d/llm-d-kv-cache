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

#pragma once

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include "cfg.hpp"

// Copy selected GPU blocks into a staging CPU tensor.
// Returns a staging CPU tensor containing raw K/V block bytes.
bool copy_gpu_tensors_to_cpu_tensor(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    torch::Tensor& cpu_tensor,
    const c10::cuda::CUDAStream& stream,
    const ConnectorConfig& cfg);

// Copy data from a staging CPU buffer back into GPU tensors
bool copy_cpu_tensor_to_gpu_tensors(
    torch::Tensor& cpu_tensor,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<torch::Tensor>& dst_tensors,
    const c10::cuda::CUDAStream& stream,
    const ConnectorConfig& cfg);

// Kernel-based copy implementation (used when enabled in config)
void copy_via_kernel(uint8_t* cpu_base,
                     const std::vector<torch::Tensor>& gpu_tensors,
                     const std::vector<int64_t>& block_ids_list,
                     const c10::cuda::CUDAStream& stream,
                     bool is_put,
                     const ConnectorConfig& cfg);
