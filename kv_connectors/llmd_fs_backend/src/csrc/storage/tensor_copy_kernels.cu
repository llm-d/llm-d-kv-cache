#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "tensor_copy.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"

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
// Kernel copies one K or V plane of one block.
// Each thread cooperates to copy bytes from src to dst.
__global__ void copy_blocks_kernel(
    const uint8_t* __restrict__ src_base,  // Source (CPU for GET, GPU for PUT)
    uint8_t* __restrict__ dst_base,  // Destination (GPU for GET, CPU for PUT)
    const int64_t* __restrict__ block_ids,          // Global block IDs to copy
    const int64_t* __restrict__ src_block_offsets,  // Per-block source offsets
                                                    // (within file or buffer)
    const int num_blocks,                           // Number of blocks to copy
    const int layer,                                // Layer index
    const int64_t
        num_blocks_tot,  // Total blocks per tensor (used for offset math)
    const size_t bytes_per_plane,  // Bytes per K or V plane
    const size_t bytes_per_block,  // Total bytes per block (K+V)
    const bool kv_plane_first,  // use plane-based layout or full-block layout
    const bool layers_before_blocks,  // layers appear before block dimension
    const bool is_put)                // Add direction flag
{
  const int bi = blockIdx.x;  // block index
  const int plane_idx =
      blockIdx.y;  // 0=K, 1=V (only used if kv_plane_first==true)
  const int tid = threadIdx.x;

  if (bi >= num_blocks) return;

  // Global block ID
  const int64_t gpu_block_idx = block_ids[bi];
  // CPU offset for this block
  const size_t src_block_base = src_block_offsets[bi];

  // Compute src/dst pointers based on layout
  const uint8_t* src;
  uint8_t* dst;
  size_t copy_size;

  // ----------------------------------------------------------------------
  // Case 1: (DefaultKVFirst) KV-before-block layout
  // ----------------------------------------------------------------------
  if (kv_plane_first) {
    const size_t plane_offset = (plane_idx == 1 ? bytes_per_plane : 0);
    const size_t gpu_offset =
        (gpu_block_idx + plane_idx * num_blocks_tot) * bytes_per_plane;
    const size_t cpu_offset =
        src_block_base + layer * bytes_per_block + plane_offset;

    src = src_base + (is_put ? gpu_offset : cpu_offset);
    dst = dst_base + (is_put ? cpu_offset : gpu_offset);
    copy_size = bytes_per_plane;
  }
  // ----------------------------------------------------------------------
  // Case 2: (DefaultBlockFirst) Block-before-KV layout
  // ----------------------------------------------------------------------
  else if (layers_before_blocks) {
    const size_t gpu_offset = gpu_block_idx * bytes_per_block;
    const size_t cpu_offset = src_block_base + layer * bytes_per_block;

    src = src_base + (is_put ? gpu_offset : cpu_offset);
    dst = dst_base + (is_put ? cpu_offset : gpu_offset);
    copy_size = bytes_per_block;
  }
  // ----------------------------------------------------------------------
  // Case 3: Cross-layer layout
  // ----------------------------------------------------------------------
  else {
    const size_t gpu_offset = gpu_block_idx * bytes_per_block;

    src = src_base + (is_put ? gpu_offset : src_block_base);
    dst = dst_base + (is_put ? src_block_base : gpu_offset);
    copy_size = bytes_per_block;
  }

  // Copy cooperatively across threads
  for (size_t i = tid; i < copy_size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// GPU kernel-based copy path (uses CUDA threads for copying)
void copy_via_kernel(uint8_t* cpu_base,
                     const std::vector<torch::Tensor>& gpu_tensors,
                     const std::vector<int64_t>& block_ids_list,
                     const c10::cuda::CUDAStream& stream,
                     bool is_put,
                     const ConnectorConfig& cfg) {
  const int num_layers = static_cast<int>(gpu_tensors.size());
  const bool is_cross_layer = (cfg.layout.mode == LayoutMode::CrossLayer);
  const bool kv_plane_first = (cfg.layout.mode == LayoutMode::DefaultKVFirst);

  // Calculate CPU buffer offset for each block (maps global block ID to local
  // file offset)
  std::vector<int64_t> cpu_offsets(block_ids_list.size());
  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    cpu_offsets[bi] = (block_ids_list[bi] % cfg.gpu_blocks_per_file) *
                      gpu_tensors.size() * cfg.layout.bytes_per_block;
  }
  // Wrap block IDs in tensor and copy to GPU for kernel access
  torch::Tensor block_ids_tensor = to_gpu_tensor(block_ids_list);

  // Wrap CPU offsets in tensor and copy to GPU for kernel access
  torch::Tensor cpu_offsets_tensor = to_gpu_tensor(cpu_offsets);

  // Map CPU memory to device pointer (required for GPU kernel to write to
  // host memory - zero-copy)
  uint8_t* cpu_base_dev = cpu_base;
  if (is_put) {
    check_cuda_error(cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0),
                     "cudaHostGetDevicePointer failed");
  }

  // Configure grid dimensions
  const dim3 grid(block_ids_list.size(),
                  kv_plane_first ? 2 : 1);  // (blocks, K/V if kv_plane_first)
  constexpr dim3 block(COPY_THREADS);

  // Launch copy kernel for all the blocks on each layer
  // (Cross-layer layout: launch once with layer_count=1 and gpu_tensors[0])
  const int layer_count = is_cross_layer ? 1 : num_layers;

  for (int layer = 0; layer < layer_count; ++layer) {
    uint8_t* gpu_ptr = reinterpret_cast<uint8_t*>(
        gpu_tensors[is_cross_layer ? 0 : layer].data_ptr());

    copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
        is_put ? gpu_ptr : cpu_base,      // Source
        is_put ? cpu_base_dev : gpu_ptr,  // Destination
        block_ids_tensor.data_ptr<int64_t>(),
        cpu_offsets_tensor.data_ptr<int64_t>(),
        block_ids_list.size(),
        layer,
        cfg.layout.num_blocks,
        cfg.layout.kv_bytes_per_plane,
        cfg.layout.bytes_per_block,
        kv_plane_first,
        !is_cross_layer,
        is_put);
  }

  // Check for kernel launch errors
  check_cuda_error(cudaGetLastError(), "Kernel launch failed");
}
