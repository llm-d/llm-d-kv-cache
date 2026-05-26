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
TensorCopier::TensorCopier(
    std::vector<torch::Tensor>& tensors,
    std::vector<std::vector<int64_t>> group_tensor_indices,
    int gpu_blocks_per_file)
    : m_gpu_blocks_per_file(gpu_blocks_per_file),
      m_gpu_tensors(tensors),
      m_group_tensor_indices(std::move(group_tensor_indices)) {
  TORCH_CHECK(!m_gpu_tensors.empty(), "TensorCopier: tensors is empty");
  TORCH_CHECK(!m_group_tensor_indices.empty(),
              "TensorCopier: group_tensor_indices is empty");
  TORCH_CHECK(m_gpu_blocks_per_file > 0,
              "TensorCopier: gpu_blocks_per_file must be > 0");
  TORCH_CHECK(tensors[0].is_contiguous(), "GPU tensor must be contiguous");

  m_tensor_block_size = tensors[0].stride(0) * tensors[0].element_size();
  // Env flags
  m_use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
  m_use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
  // Batched DMA is the default fast path on CUDA 12.8+; the per-call
  // cudaMemcpyAsync loop remains as a fallback when these flags are
  // explicitly set to 0 (older toolkits, debugging, A/B comparison).
  // cudaMemcpyBatchAsync was introduced in CUDA 12.8 — default off below that.
#if CUDA_VERSION >= 12080
  constexpr bool kBatchDefault = true;
#else
  constexpr bool kBatchDefault = false;
#endif
  m_use_batch_memcpy_read =
      get_env_flag("USE_BATCH_MEMCPY_READ", kBatchDefault);
  m_use_batch_memcpy_write =
      get_env_flag("USE_BATCH_MEMCPY_WRITE", kBatchDefault);
  FS_LOG_INFO("TensorCopier: use_kernel_copy_read="
              << m_use_kernel_copy_read
              << ", use_kernel_copy_write=" << m_use_kernel_copy_write
              << ", use_batch_memcpy_read=" << m_use_batch_memcpy_read
              << ", use_batch_memcpy_write=" << m_use_batch_memcpy_write
              << ", m_gpu_blocks_per_file=" << m_gpu_blocks_per_file
              << ", num_groups=" << m_group_tensor_indices.size());
}

// Performs block transfers using cudaMemcpyAsync (DMA-based copy)
void TensorCopier::copy_blocks_via_cuda_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    int group_idx,
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

  const auto& tensor_indices = m_group_tensor_indices[group_idx];
  const size_t num_tensors = tensor_indices.size();

  // Get current CUDA stream
  const auto stream = at::cuda::getCurrentCUDAStream();
  //  Compute CPU block offset, Each block in CPU memory stores all layers
  //  sequentially: [layer0_data, layer1_data, ..., layerN_data]
  cpu_blk_ptr = cpu_base + (m_gpu_blocks_per_file - block_ids_list.size()) *
                               num_tensors * m_tensor_block_size;

  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];
    // Process only tensors belonging to this group
    for (int64_t tidx : tensor_indices) {
      const auto& tensor = m_gpu_tensors[tidx];
      gpu_blk_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr()) +
                    gpu_block_idx * m_tensor_block_size;
      // Perform async copy - returns immediately, transfers in background
      cudaError_t err = cudaMemcpyAsync(*dst,
                                        *src,
                                        m_tensor_block_size,
                                        kind,
                                        stream.stream());
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");

      // increment CPU block pointer to next block
      cpu_blk_ptr += m_tensor_block_size;
    }
  }
}

// Batched DMA path: one cudaMemcpyBatchAsync covers all per-(block, layer)
// copies for the blocks in this file (num_blocks * num_tensors).
// The batch executes in stream order; ordering within the batch is unspecified.
void TensorCopier::copy_blocks_via_batch_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    bool is_store) {
  const size_t num_tensors = m_gpu_tensors.size();
  const size_t num_blocks = block_ids_list.size();
  const size_t total = num_tensors * num_blocks;
  if (total == 0) return;

  // Thread-local scratch arrays — avoid alloc on every transfer. Capacities
  // grow monotonically with the largest call we've seen.
  thread_local std::vector<void*> dsts;
  thread_local std::vector<void*> srcs;
  thread_local std::vector<size_t> sizes;
  dsts.resize(total);
  srcs.resize(total);
  sizes.resize(total);

  //  Compute CPU block offset, Each block in CPU memory stores all layers
  //  sequentially: [layer0_data, layer1_data, ..., layerN_data]
  uint8_t* cpu_blk_ptr = cpu_base + (m_gpu_blocks_per_file - num_blocks) *
                                        num_tensors * m_tensor_block_size;

  // Build one (dst, src, size) descriptor per (block, layer) copy.
  size_t idx = 0;
  for (size_t bi = 0; bi < num_blocks; ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];
    for (const auto& tensor : m_gpu_tensors) {
      uint8_t* gpu_blk_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr()) +
                             gpu_block_idx * m_tensor_block_size;
      if (is_store) {
        srcs[idx] = gpu_blk_ptr;
        dsts[idx] = cpu_blk_ptr;
      } else {
        srcs[idx] = cpu_blk_ptr;
        dsts[idx] = gpu_blk_ptr;
      }
      sizes[idx] = m_tensor_block_size;
      cpu_blk_ptr += m_tensor_block_size;
      ++idx;
    }
  }

#if CUDA_VERSION >= 12080
  // Set attributes with srcAccessOrder=ANY (cudaMemcpySrcAccessOrderAny)
  // for malloc'd host staging buffer. Same as vLLM's cuda_mem_ops.py.
  // static (not thread_local): never mutated, no per-thread duplication needed.
  // Not const: CUDA's C API takes non-const pointers.
  static cudaMemcpyAttributes attrs = [] {
    cudaMemcpyAttributes a{};
    a.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    return a;
  }();
  static size_t attrs_idx = 0;

  // Get current CUDA stream
  const auto stream = at::cuda::getCurrentCUDAStream();

  // CUDA 13 dropped the failIdx out-param; CUDA 12.8/12.9 still requires it.
  #if CUDA_VERSION >= 13000
  cudaError_t err = cudaMemcpyBatchAsync(dsts.data(),
                                         srcs.data(),
                                         sizes.data(),
                                         total,
                                         &attrs,
                                         &attrs_idx,
                                         /*numAttrs=*/1,
                                         stream.stream());
  #else
  static thread_local size_t fail_idx;
  cudaError_t err = cudaMemcpyBatchAsync(dsts.data(),
                                         srcs.data(),
                                         sizes.data(),
                                         total,
                                         &attrs,
                                         &attrs_idx,
                                         /*numAttrs=*/1,
                                         &fail_idx,
                                         stream.stream());
  #endif
  TORCH_CHECK(err == cudaSuccess,
              "cudaMemcpyBatchAsync failed err=",
              cudaGetErrorString(err));
#else
  // CUDA < 12.8: cudaMemcpyBatchAsync is not available — fall back.
  copy_blocks_via_cuda_memcpy(cpu_base, block_ids_list, is_store);
#endif
}

// Dispatches to one of three paths (priority: batch > kernel > memcpy):
//   - batch memcpy: one cudaMemcpyBatchAsync (CUDA 12.8+) for all
//     per-(block, layer) copies in this file.
//   - kernel copy:  custom CUDA kernel doing the copies.
//   - memcpy loop:  one cudaMemcpyAsync per (block, layer) (fallback).
void TensorCopier::copy_blocks(uint8_t* cpu_base,
                               const std::vector<int64_t>& block_ids_list,
                               int group_idx,
                               bool is_store) {
  TORCH_CHECK(group_idx >= 0 && static_cast<size_t>(group_idx) <
                                    m_group_tensor_indices.size(),
              "TensorCopier: invalid group_idx=",
              group_idx,
              " num_groups=",
              m_group_tensor_indices.size());
  bool use_batch =
      is_store ? m_use_batch_memcpy_write : m_use_batch_memcpy_read;
  bool use_kernel = is_store ? m_use_kernel_copy_write : m_use_kernel_copy_read;
  if (use_batch) {
    copy_blocks_via_batch_memcpy(cpu_base, block_ids_list, is_store);
  } else if (use_kernel) {
    copy_blocks_via_kernels(cpu_base, block_ids_list, group_idx, is_store);
  } else {
    copy_blocks_via_cuda_memcpy(cpu_base, block_ids_list, group_idx, is_store);
  }
}

// Batched DMA path: one cudaMemcpyBatchAsync covers all per-(block, layer)
// copies for the blocks in this file (num_blocks * num_tensors_in_group).
// The batch executes in stream order; ordering within the batch is unspecified.
void TensorCopier::copy_blocks_via_batch_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    int group_idx,
    bool is_store) {
  const auto& tensor_indices = m_group_tensor_indices[group_idx];
  const size_t num_tensors = tensor_indices.size();
  const size_t num_blocks = block_ids_list.size();
  const size_t total = num_tensors * num_blocks;
  if (total == 0) return;

  // Thread-local scratch arrays — avoid alloc on every transfer. Capacities
  // grow monotonically with the largest call we've seen.
  thread_local std::vector<void*> dsts;
  thread_local std::vector<void*> srcs;
  thread_local std::vector<size_t> sizes;
  dsts.resize(total);
  srcs.resize(total);
  sizes.resize(total);

  //  Compute CPU block offset, Each block in CPU memory stores all layers
  //  sequentially: [layer0_data, layer1_data, ..., layerN_data]
  uint8_t* cpu_blk_ptr = cpu_base + (m_gpu_blocks_per_file - num_blocks) *
                                        num_tensors * m_tensor_block_size;

  // Build one (dst, src, size) descriptor per (block, layer) copy.
  size_t idx = 0;
  for (size_t bi = 0; bi < num_blocks; ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];
    for (int64_t tidx : tensor_indices) {
      uint8_t* gpu_blk_ptr =
          reinterpret_cast<uint8_t*>(m_gpu_tensors[tidx].data_ptr()) +
          gpu_block_idx * m_tensor_block_size;
      if (is_store) {
        srcs[idx] = gpu_blk_ptr;
        dsts[idx] = cpu_blk_ptr;
      } else {
        srcs[idx] = cpu_blk_ptr;
        dsts[idx] = gpu_blk_ptr;
      }
      sizes[idx] = m_tensor_block_size;
      cpu_blk_ptr += m_tensor_block_size;
      ++idx;
    }
  }

  // Set attributes with srcAccessOrder=ANY (cudaMemcpySrcAccessOrderAny)
  // for malloc'd host staging buffer. Same as vLLM's cuda_mem_ops.py.
  thread_local cudaMemcpyAttributes attrs = [] {
    cudaMemcpyAttributes a{};
    a.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    return a;
  }();
  thread_local size_t attrs_idx = 0;

  // Get current CUDA stream
  const auto stream = at::cuda::getCurrentCUDAStream();

  // CUDA 13 dropped the failIdx out-param; CUDA 12.8/12.9 still requires it.
#if CUDA_VERSION >= 13000
  cudaError_t err = cudaMemcpyBatchAsync(dsts.data(),
                                         srcs.data(),
                                         sizes.data(),
                                         total,
                                         &attrs,
                                         &attrs_idx,
                                         /*numAttrs=*/1,
                                         stream.stream());
#else
  static thread_local size_t fail_idx;
  cudaError_t err = cudaMemcpyBatchAsync(dsts.data(),
                                         srcs.data(),
                                         sizes.data(),
                                         total,
                                         &attrs,
                                         &attrs_idx,
                                         /*numAttrs=*/1,
                                         &fail_idx,
                                         stream.stream());
#endif
  TORCH_CHECK(err == cudaSuccess,
              "cudaMemcpyBatchAsync failed err=",
              cudaGetErrorString(err));
}
