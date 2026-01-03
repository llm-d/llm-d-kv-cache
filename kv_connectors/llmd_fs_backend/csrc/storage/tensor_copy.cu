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

#include "tensor_copy.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"
#include <cstdlib>
#include <string>

// Constructor - initializes configuration
TensorCopy::TensorCopy(std::vector<torch::Tensor>& tensors,
                       int gpu_blocks_per_file)
    : m_gpu_blocks_per_file(gpu_blocks_per_file), m_gpu_tensors(tensors) {
  TORCH_CHECK(!m_gpu_tensors.empty(), "TensorCopy: tensors is empty");
  TORCH_CHECK(m_gpu_blocks_per_file > 0,
              "TensorCopy: gpu_blocks_per_file must be > 0");
  TORCH_CHECK(tensors[0].is_contiguous(), "GPU tensor must be contiguous");

  m_tensor_block_size = tensors[0].stride(0) * tensors[0].element_size();
  // Env flags
  m_use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
  m_use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
  std::cout << "[INFO] TensorCopy: use_kernel_copy_read="
            << m_use_kernel_copy_read
            << ", use_kernel_copy_write=" << m_use_kernel_copy_write
            << ", m_gpu_blocks_per_file=" << m_gpu_blocks_per_file << std::endl;
}

// Performs block transfers using cudaMemcpyAsync (DMA-based copy)
void TensorCopy::copy_blocks_via_cuda_memcpy(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream,
    bool is_put) {
  uint8_t** src;
  uint8_t** dst;
  uint8_t* gpu_layer_offset;
  uint8_t* cpu_layer_offset;
  cudaMemcpyKind kind;

  // Determine source and destination based on direction
  if (is_put) {
    kind = cudaMemcpyDeviceToHost;
    src = &gpu_layer_offset;
    dst = &cpu_layer_offset;
  } else {
    kind = cudaMemcpyHostToDevice;
    src = &cpu_layer_offset;
    dst = &gpu_layer_offset;
  }

  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];

    // Compute CPU block offset
    // Each block in CPU memory stores all layers sequentially:
    // [layer0_data, layer1_data, ..., layerN_data]
    uint8_t* cpu_block_offset =
        cpu_base + (gpu_block_idx % m_gpu_blocks_per_file) *
                       m_gpu_tensors.size() * m_tensor_block_size;

    // Process all layers for this block (for cross-layer layout is just one
    // layer)
    size_t li = 0;
    for (const auto& tensor : m_gpu_tensors) {
      gpu_layer_offset = reinterpret_cast<uint8_t*>(tensor.data_ptr()) +
                         gpu_block_idx * m_tensor_block_size;
      cpu_layer_offset = cpu_block_offset + li * m_tensor_block_size;

      // Perform async copy - returns immediately, transfers in background
      cudaError_t err = cudaMemcpyAsync(*dst,
                                        *src,
                                        m_tensor_block_size,
                                        kind,
                                        stream.stream());
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
      li++;
    }
  }
}

// Main transfer function - dispatches to kernel or memcpy path
void TensorCopy::copy_blocks(uint8_t* cpu_base,
                             const std::vector<int64_t>& block_ids_list,
                             const c10::cuda::CUDAStream& stream,
                             bool is_put) {
  bool use_kernel = is_put ? m_use_kernel_copy_write : m_use_kernel_copy_read;
  if (use_kernel) {
    copy_blocks_via_kernels(cpu_base, block_ids_list, stream, is_put);
  } else {
    copy_blocks_via_cuda_memcpy(cpu_base, block_ids_list, stream, is_put);
  }
}
