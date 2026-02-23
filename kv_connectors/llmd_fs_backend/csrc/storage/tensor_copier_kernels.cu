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

//----------------------------------------------------------------------
// Helper Structures and Functions
//----------------------------------------------------------------------
// Helper to wrap CPU arrays as GPU tensors for kernel access
template <typename T>
inline torch::Tensor to_gpu_tensor(const std::vector<T>& data) {
  return torch::from_blob(const_cast<T*>(data.data()),
                          {static_cast<int64_t>(data.size())},
                          torch::dtype(torch::kInt64))
      .to(torch::kCUDA, /*non_blocking=*/true);
}

// Thread configuration constant
constexpr int COPY_THREADS =
    512;  // TODO: check optimal thread count (256 or 512)

// Error checking helper
inline void check_cuda_error(cudaError_t err, const char* msg) {
  TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

//----------------------------------------------------------------------
// CUDA Kernel
//----------------------------------------------------------------------
// Kernel copies one block of one tensor(layer).
// Each thread cooperates to copy bytes from src to dst.
__global__ void copy_blocks_kernel(
    uint8_t* __restrict__ cpu_base,         // CPU staging buffer
    uint8_t* __restrict__ gpu_base,         // GPU tensor base pointer
    const int64_t* __restrict__ block_ids,  // Global block IDs to copy
    const size_t tensor_block_size,         // Bytes per block per layer
    const int num_blocks,                   // Number of blocks to copy
    const int layer_idx,                    // Current layer being processed
    const int num_layers,                   // Total number of layers
    const int gpu_blocks_per_file,  // Blocks per file for modulo calculation
    const bool is_store)            // Direction flag (true=PUT, false=GET)
{
  const int bi = blockIdx.x;  // block index
  const int tid = threadIdx.x;

  if (bi >= num_blocks) return;

  const int64_t gpu_block_idx = block_ids[bi];

  // Compute GPU offset: direct block indexing
  const size_t gpu_offset = gpu_block_idx * tensor_block_size;

  // Compute CPU block offset
  // Each block in CPU memory stores all layers sequentially:
  // [layer0_data, layer1_data, ..., layerN_data]
  const size_t cpu_block_base =
      (gpu_block_idx % gpu_blocks_per_file) * num_layers * tensor_block_size;
  const size_t cpu_offset = cpu_block_base + layer_idx * tensor_block_size;

  // Determine source and destination based on direction
  const uint8_t* src =
      is_store ? (gpu_base + gpu_offset) : (cpu_base + cpu_offset);
  uint8_t* dst = is_store ? (cpu_base + cpu_offset) : (gpu_base + gpu_offset);

  // Copy cooperatively across threads
  for (size_t i = tid; i < tensor_block_size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Performs block transfers using a custom CUDA kernel
void TensorCopier::copy_blocks_via_kernels(
    uint8_t* cpu_base,
    const std::vector<int64_t>& block_ids_list,
    bool is_store) {
  const int num_layers = static_cast<int>(m_gpu_tensors.size());

  // Wrap block IDs in tensor and copy to GPU for kernel access
  torch::Tensor block_ids_tensor = to_gpu_tensor(block_ids_list);

  // Map CPU memory to device pointer (required for GPU kernel to access
  // host memory - zero-copy)
  uint8_t* cpu_base_dev;
  check_cuda_error(cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0),
                   "cudaHostGetDevicePointer failed");

  // Configure grid dimensions
  // grid.x = number of blocks to copy
  // grid.y = 1 (process one layer at a time)
  const dim3 grid(block_ids_list.size(), 1);
  constexpr dim3 block(COPY_THREADS);

  // Get current CUDA stream
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Launch copy kernel for each layer sequentially
  for (int layer = 0; layer < num_layers; ++layer) {
    uint8_t* gpu_base =
        reinterpret_cast<uint8_t*>(m_gpu_tensors[layer].data_ptr());

    copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
        cpu_base_dev,  // CPU staging buffer (device-mapped)
        gpu_base,      // GPU tensor base pointer for this layer
        block_ids_tensor.data_ptr<int64_t>(),
        m_tensor_block_size,
        block_ids_list.size(),
        layer,  // Pass current layer index
        num_layers,
        m_gpu_blocks_per_file,
        is_store);

    // Check for kernel launch errors immediately
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      FS_LOG_ERROR("Kernel launch failed for layer "
                   << layer << ": " << cudaGetErrorString(launch_err));
      check_cuda_error(launch_err, "Kernel launch failed");
    }
  }
  check_cuda_error(cudaGetLastError(), "Kernel launch failed");
}
