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
#include "debug_utils.hpp"
#include <cstdlib>
#include <string>

// Constructor - initializes configuration
TensorCopier::TensorCopier(std::vector<torch::Tensor>& tensors,
                           int gpu_blocks_per_file)
    : m_gpu_blocks_per_file(gpu_blocks_per_file), m_gpu_tensors(tensors) {
  TORCH_CHECK(!m_gpu_tensors.empty(), "TensorCopier: tensors is empty");
  TORCH_CHECK(m_gpu_blocks_per_file > 0,
              "TensorCopier: gpu_blocks_per_file must be > 0");
  TORCH_CHECK(tensors[0].is_contiguous(), "GPU tensor must be contiguous");

  m_tensor_block_size = tensors[0].stride(0) * tensors[0].element_size();
  // Env flags
  m_use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
  m_use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
  std::cout << "[INFO] TensorCopier: use_kernel_copy_read="
            << m_use_kernel_copy_read
            << ", use_kernel_copy_write=" << m_use_kernel_copy_write
            << ", m_gpu_blocks_per_file=" << m_gpu_blocks_per_file << std::endl;
}

// Performs block transfers using cudaMemcpyAsync (DMA-based copy)
void TensorCopier::copy_blocks_via_cuda_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
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
  //  Compute CPU block offset, Each block in CPU memory stores all layers
  //  sequentially: [layer0_data, layer1_data, ..., layerN_data]
  cpu_blk_ptr = cpu_base + (m_gpu_blocks_per_file - block_ids_list.size()) *
                               m_gpu_tensors.size() * m_tensor_block_size;

  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];
    // Process all layers for this block (for cross-layer layout is just one
    // layer)
    for (const auto& tensor : m_gpu_tensors) {
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

// Main transfer function - dispatches to kernel or memcpy path
void TensorCopier::copy_blocks(uint8_t* cpu_base,
                               const std::vector<int64_t>& block_ids_list,
                               bool is_store) {
  bool use_kernel = is_store ? m_use_kernel_copy_write : m_use_kernel_copy_read;
  if (use_kernel) {
    copy_blocks_via_kernels(cpu_base, block_ids_list, is_store);
  } else {
    copy_blocks_via_cuda_memcpy(cpu_base, block_ids_list, is_store);
  }
}
