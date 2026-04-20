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

class TensorCopier {
 public:
  TensorCopier(std::vector<torch::Tensor>& tensors,
               std::vector<std::vector<int64_t>> group_tensor_indices,
               int gpu_blocks_per_files);

  // Main transfer function - dispatches to kernel or memcpy path
  // group_idx selects the subset of tensors used for this transfer.
  void copy_blocks(uint8_t* cpu_base,
                   const std::vector<int64_t>& block_ids_list,
                   int group_idx,
                   bool is_store);

  // Return number of tensors used by the given KV cache group
  size_t num_tensors_for_group(int group_idx) const {
    return m_group_tensor_indices[group_idx].size();
  }

 private:
  // GPU tensor list (flat, canonical)
  std::vector<torch::Tensor> m_gpu_tensors;
  // Per-group tensor indices (into m_gpu_tensors). Each sub-list contains the
  // tensor indices that belong to that KV cache group.
  std::vector<std::vector<int64_t>> m_group_tensor_indices;
  // Number of GPU blocks stored per file
  int m_gpu_blocks_per_file;
  // Size in bytes of one KV block
  size_t m_tensor_block_size;
  // Use kernel-based copy for put operations
  bool m_use_kernel_copy_write;
  // Use kernel-based copy for get operations
  bool m_use_kernel_copy_read;

  // Performs block transfers using cudaMemcpyAsync (DMA-based copy)
  void copy_blocks_via_cuda_memcpy(uint8_t* cpu_base,
                                   const std::vector<int64_t>& block_ids_list,
                                   int group_idx,
                                   bool is_store);

  // Performs block transfers using a custom CUDA kernel
  void copy_blocks_via_kernels(uint8_t* cpu_base,
                               const std::vector<int64_t>& block_ids_list,
                               int group_idx,
                               bool is_store);
};