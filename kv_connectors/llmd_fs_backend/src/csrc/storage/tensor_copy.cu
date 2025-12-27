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
// Copy Implementation Functions
//----------------------------------------------------------------------
// Standard cudaMemcpyAsync path (DMA-based copying)
void copy_via_cuda_memcpy(uint8_t* cpu_base,
                          const std::vector<torch::Tensor>& gpu_tensors,
                          const std::vector<int64_t>& block_ids_list,
                          const c10::cuda::CUDAStream& stream,
                          bool is_put,
                          const ConnectorConfig& cfg) {
  cudaMemcpyKind kind =
      is_put ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

  uint8_t* gpu_base = reinterpret_cast<uint8_t*>(gpu_tensors[0].data_ptr());
  // Direct pointer arithmetic - no indexing operations
  for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
    int64_t gpu_block_idx = block_ids_list[bi];
    // Compute block offset
    uint8_t* cpu_block_offset =
        cpu_base + (gpu_block_idx % cfg.gpu_blocks_per_file) *
                       gpu_tensors.size() * cfg.layout.bytes_per_block;

    // 1) Cross-layer: single memcpy for the whole block (all layers packed)
    if (cfg.layout.mode == LayoutMode::CrossLayer) {
      uint8_t* gpu_block_offset =
          gpu_base + gpu_block_idx * cfg.layout.bytes_per_block;
      const void* src = is_put ? gpu_block_offset : cpu_block_offset;
      void* dst = is_put ? cpu_block_offset : gpu_block_offset;
      cudaError_t err = cudaMemcpyAsync(dst,
                                        src,
                                        cfg.layout.bytes_per_block,
                                        kind,
                                        stream.stream());
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
      continue;
    }

    // Otherwise, it's one of the per-layer layouts
    for (size_t layer = 0; layer < gpu_tensors.size(); ++layer) {
      uint8_t* gpu_layer_offset =
          reinterpret_cast<uint8_t*>(gpu_tensors[layer].data_ptr());
      uint8_t* cpu_layer_offset =
          cpu_block_offset + layer * cfg.layout.bytes_per_block;

      // 2) KV-first: two plane copies (K then V)
      if (cfg.layout.mode == LayoutMode::DefaultKVFirst) {
        size_t plane = cfg.layout.kv_bytes_per_plane;

        size_t k_off = static_cast<size_t>(gpu_block_idx) * plane;
        size_t v_off =
            (static_cast<size_t>(gpu_block_idx) + cfg.layout.num_blocks) *
            plane;

        // Compute GPU and CPU offsets for K
        void* src_K = is_put ? (gpu_layer_offset + k_off) : (cpu_layer_offset);
        void* dst_K = is_put ? (cpu_layer_offset) : (gpu_layer_offset + k_off);

        cudaError_t err1 =
            cudaMemcpyAsync(dst_K, src_K, plane, kind, stream.stream());
        TORCH_CHECK(err1 == cudaSuccess, "cudaMemcpyAsync failed (K)");

        // Compute GPU and CPU offsets for V
        void* src_V =
            is_put ? (gpu_layer_offset + v_off) : (cpu_layer_offset + plane);
        void* dst_V =
            is_put ? (cpu_layer_offset + plane) : (gpu_layer_offset + v_off);
        cudaError_t err2 =
            cudaMemcpyAsync(dst_V, src_V, plane, kind, stream.stream());
        TORCH_CHECK(err2 == cudaSuccess, "cudaMemcpyAsync failed (V)");
        continue;
      }

      // 3) Block-first: one copy for K+V together
      TORCH_CHECK(cfg.layout.mode == LayoutMode::DefaultBlockFirst,
                  "Unexpected layout mode");
      uint8_t* gpu_block_offset =
          gpu_layer_offset + gpu_block_idx * cfg.layout.bytes_per_block;
      void* src = is_put ? (gpu_block_offset) : (cpu_layer_offset);
      void* dst = is_put ? (cpu_layer_offset) : (gpu_block_offset);
      cudaError_t err = cudaMemcpyAsync(dst,
                                        src,
                                        cfg.layout.bytes_per_block,
                                        kind,
                                        stream.stream());
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
    }
  }
}

// Main transfer function - dispatches to kernel or memcpy path
void transfer_kv_blocks(uint8_t* cpu_base,
                        const std::vector<torch::Tensor>& gpu_tensors,
                        const std::vector<int64_t>& block_ids_list,
                        const c10::cuda::CUDAStream& stream,
                        bool is_put,
                        const ConnectorConfig& cfg) {
  bool use_kernel =
      is_put ? cfg.use_kernel_copy_write : cfg.use_kernel_copy_read;
  if (use_kernel) {
    copy_via_kernel(cpu_base, gpu_tensors, block_ids_list, stream, is_put, cfg);
  } else {
    copy_via_cuda_memcpy(cpu_base,
                         gpu_tensors,
                         block_ids_list,
                         stream,
                         is_put,
                         cfg);
  }
}

//----------------------------------------------------------------------
// GPU -> Storage (PUT)
//----------------------------------------------------------------------
// Copy selected GPU K/V blocks into a single staging CPU tensor.
bool copy_gpu_tensors_to_cpu_tensor(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    torch::Tensor& cpu_tensor,
    const c10::cuda::CUDAStream& stream,
    const ConnectorConfig& cfg) {
  TORCH_CHECK(!src_tensors.empty(), "Source tensors list is empty");
  const auto& ref = src_tensors[0];
  TORCH_CHECK(ref.is_contiguous(), "src_tensors must be contiguous");

  // Fetch or allocate thread-local staging buffer
  StagingBufferInfo& buf =
      ThreadPool::tls_staging_buffer(cfg.staging_buffer_size_bytes);
  // Wrap staging buffer as tensor view (no copy)
  cpu_tensor = torch::from_blob(buf.ptr,
                                {cfg.staging_buffer_total_elements},
                                torch::TensorOptions()
                                    .dtype(cfg.layout.dtype)
                                    .device(torch::kCPU)
                                    .pinned_memory(true));

  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_put = true;

  // Execute the copy operation
  transfer_kv_blocks(cpu_base,
                     src_tensors,
                     block_ids_list,
                     stream,
                     is_put,
                     cfg);

  // Reinterpret bfloat16 tensor as uint16_t for safe raw byte access (I/O or
  // memcpy)
  if (cpu_tensor.dtype() == torch::kBFloat16) {
    cpu_tensor = cpu_tensor.view(torch::kUInt16);
  }

  return true;
}

//----------------------------------------------------------------------
// Storage -> GPU (GET)
//----------------------------------------------------------------------

bool copy_cpu_tensor_to_gpu_tensors(
    torch::Tensor& cpu_tensor,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<torch::Tensor>& dst_tensors,
    const c10::cuda::CUDAStream& stream,
    const ConnectorConfig& cfg) {
  TORCH_CHECK(!dst_tensors.empty(), "Destination tensors list is empty");

  const auto& ref = dst_tensors[0];
  TORCH_CHECK(ref.is_contiguous(), "dst_tensors must be contiguous");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "cpu buffer must be contiguous");
  TORCH_CHECK(cpu_tensor.is_pinned(),
              "cpu_tensor must be pinned memory for kernel-based copy");

  auto* cpu_base = cpu_tensor.data_ptr<uint8_t>();
  bool is_put = false;

  // Execute the copy operation
  transfer_kv_blocks(cpu_base,
                     dst_tensors,
                     block_ids_list,
                     stream,
                     is_put,
                     cfg);

  return true;
}
