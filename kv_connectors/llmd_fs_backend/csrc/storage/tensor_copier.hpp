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

  // Main transfer function - dispatches to kernel or memcpy path.
  // group_idx selects the subset of tensors used for this transfer.
  // head_offset = staging slot (in GPU blocks) where this group's data sits.
  void copy_blocks(uint8_t* cpu_base,
                   const std::vector<int64_t>& block_ids_list,
                   int group_idx,
                   int head_offset,
                   bool is_store);

  // Accessor methods for GDS direct access
  const std::vector<torch::Tensor>& get_tensors() const {
    return m_gpu_tensors;
  }
  // Tensor indices belonging to a given KV cache group.
  const std::vector<int64_t>& get_group_tensor_indices(int group_idx) const {
    return m_group_tensor_indices[group_idx];
  }
  // Returns the size in bytes of a single KV block across all tensor layers
  size_t get_block_size() const { return m_tensor_block_size; }
  // Return number of tensors used by the given KV cache group
  size_t num_tensors_for_group(int group_idx) const {
    return m_group_tensor_indices[group_idx].size();
  }
  // Number of GPU blocks bundled into a single file.
  int gpu_blocks_per_file() const { return m_gpu_blocks_per_file; }
  // Total bytes of actual KV data for one block of the given group
  // (sums per-tensor block bytes across all tensors in the group).
  size_t bytes_per_block_for_group(int group_idx) const {
    return num_tensors_for_group(group_idx) * m_tensor_block_size;
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
  // Use cudaMemcpyBatchAsync (CUDA 12.8+) for put operations
  bool m_use_batch_memcpy_write;
  // Use cudaMemcpyBatchAsync (CUDA 12.8+) for get operations
  bool m_use_batch_memcpy_read;

  // Performs block transfers using cudaMemcpyAsync (DMA-based copy)
  void copy_blocks_via_cuda_memcpy(uint8_t* cpu_base,
                                   const std::vector<int64_t>& block_ids_list,
                                   int group_idx,
                                   int head_offset,
                                   bool is_store);

  // Performs block transfers using a custom CUDA kernel
  void copy_blocks_via_kernels(uint8_t* cpu_base,
                               const std::vector<int64_t>& block_ids_list,
                               int group_idx,
                               int head_offset,
                               bool is_store);

  // Single cudaMemcpyBatchAsync call (CUDA 12.8+) submitting all
  // (block, layer) copies — removes per-call dispatch overhead.
  void copy_blocks_via_batch_memcpy(uint8_t* cpu_base,
                                    const std::vector<int64_t>& block_ids_list,
                                    int group_idx,
                                    int head_offset,
                                    bool is_store);
};
