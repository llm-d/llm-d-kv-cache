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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "tensor_copier.hpp"
#include "thread_pool.hpp"
#include "logger.hpp"
#include <cstdlib>
#include <string>

// Constructor - initializes configuration
TensorCopier::TensorCopier(std::vector<torch::Tensor>& tensors,
                           int gpu_blocks_per_file,
                           int sub_blocks_per_gpu_block,
                           int kernel_blocks_per_canonical_block)
    : m_gpu_blocks_per_file(gpu_blocks_per_file),
      m_sub_blocks_per_gpu_block(sub_blocks_per_gpu_block),
      m_kernel_blocks_per_canonical_block(kernel_blocks_per_canonical_block),
      m_gpu_tensors(tensors) {
  TORCH_CHECK(!m_gpu_tensors.empty(), "TensorCopier: tensors is empty");
  TORCH_CHECK(m_gpu_blocks_per_file > 0,
              "TensorCopier: gpu_blocks_per_file must be > 0");
  TORCH_CHECK(m_sub_blocks_per_gpu_block > 0,
              "TensorCopier: sub_blocks_per_gpu_block must be > 0");
  TORCH_CHECK(m_kernel_blocks_per_canonical_block >= 1,
              "TensorCopier: kernel_blocks_per_canonical_block must be >= 1");
  m_tensor_block_size = tensors[0].stride(0) * tensors[0].element_size();
  TORCH_CHECK(m_tensor_block_size > 0,
              "TensorCopier: tensor block size must be > 0");
  TORCH_CHECK(m_tensor_block_size % m_sub_blocks_per_gpu_block == 0,
              "TensorCopier: tensor block size must be divisible by "
              "sub_blocks_per_gpu_block");
  m_tensor_sub_block_size = m_tensor_block_size / m_sub_blocks_per_gpu_block;
  for (size_t i = 1; i < tensors.size(); ++i) {
    const size_t tensor_block_size =
        tensors[i].stride(0) * tensors[i].element_size();
    TORCH_CHECK(tensor_block_size == m_tensor_block_size,
                "All KV-cache tensors must have the same block byte size.");
  }
  // Env flags
  m_use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
  m_use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
  FS_LOG_DEBUG("TensorCopier: use_kernel_copy_read="
               << m_use_kernel_copy_read
               << ", use_kernel_copy_write=" << m_use_kernel_copy_write
               << ", m_gpu_blocks_per_file=" << m_gpu_blocks_per_file
               << ", kb_per_canonical=" << m_kernel_blocks_per_canonical_block);
}

// Performs block transfers using cudaMemcpyAsync (DMA-based copy)
void TensorCopier::copy_blocks_via_cuda_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<int64_t>* block_offsets_list,
    const std::vector<int64_t>* block_counts_list,
    bool is_store) {
  uint8_t** src;
  uint8_t** dst;
  uint8_t* gpu_blk_ptr;
  uint8_t* cpu_blk_ptr;
  cudaMemcpyKind kind;

  // Determine source and destination based on direction
  if (is_store) {
    kind = cudaMemcpyDeviceToHost;
    src = &gpu_blk_ptr;
    dst = &cpu_blk_ptr;
  } else {
    kind = cudaMemcpyHostToDevice;
    src = &cpu_blk_ptr;
    dst = &gpu_blk_ptr;
  }

  // Get current CUDA stream
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_partial_ranges =
      block_offsets_list != nullptr && block_counts_list != nullptr;
  TORCH_CHECK(
      (block_offsets_list == nullptr) == (block_counts_list == nullptr),
      "TensorCopier: block offsets and counts must either both be set or "
      "both be unset");

  size_t total_sub_blocks = 0;
  if (use_partial_ranges) {
    TORCH_CHECK(block_offsets_list->size() == block_ids_list.size(),
                "TensorCopier: block_offsets must match block_ids length");
    TORCH_CHECK(block_counts_list->size() == block_ids_list.size(),
                "TensorCopier: block_counts must match block_ids length");
    for (size_t i = 0; i < block_ids_list.size(); ++i) {
      const int64_t block_offset = (*block_offsets_list)[i];
      const int64_t block_count = (*block_counts_list)[i];
      TORCH_CHECK(block_offset >= 0, "TensorCopier: block_offset must be >= 0");
      TORCH_CHECK(block_count >= 0, "TensorCopier: block_count must be >= 0");
      TORCH_CHECK(block_offset + block_count <= m_sub_blocks_per_gpu_block,
                  "TensorCopier: block_offset + block_count exceeds "
                  "sub_blocks_per_gpu_block");
      total_sub_blocks += static_cast<size_t>(block_count);
    }
  } else {
    total_sub_blocks =
        block_ids_list.size() * static_cast<size_t>(m_sub_blocks_per_gpu_block);
  }
  TORCH_CHECK(total_sub_blocks <=
                  static_cast<size_t>(m_gpu_blocks_per_file) *
                      static_cast<size_t>(m_sub_blocks_per_gpu_block),
              "TensorCopier: transfer exceeds configured file capacity");

  // Compute CPU staging offset. Each staging file stores all tensors
  // sequentially per logical sub-block entry and remains right-aligned for
  // short first files to preserve the existing fixed-width file layout.
  cpu_blk_ptr = cpu_base +
                (static_cast<size_t>(m_gpu_blocks_per_file) *
                     static_cast<size_t>(m_sub_blocks_per_gpu_block) -
                 total_sub_blocks) *
                    m_gpu_tensors.size() * m_tensor_sub_block_size;

  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    const int64_t canonical_block_idx = block_ids_list[bi];
    const int64_t block_offset =
        use_partial_ranges ? (*block_offsets_list)[bi] : 0;
    const int64_t block_count =
        use_partial_ranges ? (*block_counts_list)[bi] : m_sub_blocks_per_gpu_block;
    const size_t block_copy_size =
        static_cast<size_t>(block_count) * m_tensor_sub_block_size;

    // When kernel_blocks_per_canonical_block > 1, each canonical
    // block_id maps to multiple consecutive kernel blocks in GPU
    // memory.  The staging buffer (and thus file) stores them
    // contiguously per layer, making the on-disk layout
    // independent of the runtime kernel block size.
    for (int kb = 0; kb < m_kernel_blocks_per_canonical_block; ++kb) {
      const int64_t gpu_block_idx =
          canonical_block_idx * m_kernel_blocks_per_canonical_block + kb;
      for (const auto& tensor : m_gpu_tensors) {
        gpu_blk_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr()) +
                      gpu_block_idx * m_tensor_block_size +
                      static_cast<size_t>(block_offset) *
                          m_tensor_sub_block_size;
        cudaError_t err = cudaMemcpyAsync(*dst,
                                          *src,
                                          block_copy_size,
                                          kind,
                                          stream.stream());
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
        cpu_blk_ptr += block_copy_size;
      }
    }
  }
}

// Main transfer function - dispatches to kernel or memcpy path
void TensorCopier::copy_blocks(uint8_t* cpu_base,
                               const std::vector<int64_t>& block_ids_list,
                               const std::vector<int64_t>* block_offsets_list,
                               const std::vector<int64_t>* block_counts_list,
                               bool is_store) {
  bool use_kernel = is_store ? m_use_kernel_copy_write : m_use_kernel_copy_read;
  if (block_offsets_list != nullptr || block_counts_list != nullptr) {
    use_kernel = false;
  }
  if (use_kernel) {
    copy_blocks_via_kernels(
        cpu_base, block_ids_list, block_offsets_list, block_counts_list, is_store);
  } else {
    copy_blocks_via_cuda_memcpy(
        cpu_base, block_ids_list, block_offsets_list, block_counts_list, is_store);
  }
}
