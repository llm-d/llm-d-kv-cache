#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "tensor_copy.hpp"
#include "buffer.hpp"
#include "debug_utils.hpp"

//----------------------------------------------------------------------
// Helper Structures and Functions
//----------------------------------------------------------------------

// Encapsulates KV cache block layout and dimensions
struct CacheLayout {
    int64_t num_blocks;         // Number of blocks, from block_axis
    size_t bytes_per_block;     // Bytes for each block, calculated from
                                // stride[block_axis] * element_size
    size_t elem_size;           // Bytes per tensor element
    int block_axis;             // Which axis is the block index
    bool kv_before_blocks;      // True if K/V dimension is before block_axis
    bool layers_before_blocks;  // True if layer dimension is before block_axis
    size_t kv_bytes_per_plane;  // Bytes per K or V plane use only when
                                // kv_before_blocks is true

    // Factory method to extract layout from tensor shape
    static CacheLayout from_tensor(const torch::Tensor& ref,
                                   int block_axis,
                                   bool kv_before_blocks,
                                   bool layers_before_blocks) {
        TORCH_CHECK(block_axis >= 0 && block_axis < ref.dim(),
                    "block_axis out of bounds!");
        int64_t num_blocks = ref.size(block_axis);
        size_t elem_size = ref.element_size();
        size_t bytes_per_plane = ref.stride(block_axis) * elem_size;

        CacheLayout layout;
        layout.num_blocks = num_blocks;
        layout.bytes_per_block =
            kv_before_blocks ? bytes_per_plane * 2 : bytes_per_plane;
        layout.elem_size = elem_size;
        layout.block_axis = block_axis;
        layout.kv_before_blocks = kv_before_blocks;
        layout.layers_before_blocks = layers_before_blocks;
        layout.kv_bytes_per_plane = bytes_per_plane;
        return layout;
    }
};

// Helper to wrap CPU arrays as GPU tensors for kernel access
template <typename T>
torch::Tensor to_gpu_tensor(const std::vector<T>& data) {
    return torch::from_blob(const_cast<T*>(data.data()),
                            {static_cast<int64_t>(data.size())},
                            torch::dtype(torch::kInt64))
        .to(torch::kCUDA, /*non_blocking=*/true);
}

// Helper for reading environment variable flags
inline bool get_env_flag(const char* name, bool default_val) {
    const char* env = std::getenv(name);
    if (!env) return default_val;

    std::string v(env);
    if (v == "1" || v == "true" || v == "TRUE") return true;
    if (v == "0" || v == "false" || v == "FALSE") return false;

    return default_val;
}

inline int64_t compute_cpu_offset(size_t bi,
                                  bool is_put,
                                  int num_layers,
                                  size_t bytes_per_block,
                                  int64_t gpu_block_idx = 0,
                                  int num_blocks_in_file = 0) {
    if (is_put) {
        // PUT - all blocks are stored contiguously in buffer
        return bi * num_layers * bytes_per_block;
    } else {
        // GET - may have only partial blocks to load from the file
        int64_t lblock = gpu_block_idx % num_blocks_in_file;
        return lblock * num_layers * bytes_per_block;
    }
}

// Thread configuration constant
constexpr int COPY_THREADS =
    512;  // TODO: check optimal thread count (256 or 512)

//----------------------------------------------------------------------
// CUDA Kernel
//----------------------------------------------------------------------
// Kernel copies one K or V plane of one block.
// Each thread cooperates to copy bytes from src → dst.
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
    const bool kv_before_blocks,  // use plane-based layout or full-block layout
    const bool layers_before_blocks,  // layers appear before block dimension
    const bool is_put)                // Add direction flag
{
    const int bi = blockIdx.x;  // block index
    const int k_or_v =
        blockIdx.y;  // 0=K, 1=V (only used if kv_before_blocks==true)
    const int tid = threadIdx.x;

    if (bi >= num_blocks) return;

    // Global block ID
    const int64_t gpu_block_idx = block_ids[bi];
    // CPU offset for this block
    const size_t src_block_base = src_block_offsets[bi];
    size_t src_plane_offset, dst_plane_offset, src_block_offset,
        dst_block_offset;

    // ----------------------------------------------------------------------
    // Case 1: K/V planes appear before the block dimension
    // (kv_before_blocks == true AND layers_before_blocks == true)
    // ----------------------------------------------------------------------
    if (kv_before_blocks) {
        size_t plane_offset = (k_or_v == 1 ? bytes_per_plane : 0);

        if (is_put) {
            // PUT: GPU → CPU
            src_plane_offset =
                static_cast<size_t>(gpu_block_idx + k_or_v * num_blocks_tot) *
                bytes_per_plane;
            dst_plane_offset = src_block_base +
                               static_cast<size_t>(layer) * bytes_per_block +
                               plane_offset;
        } else {
            // GET: CPU → GPU
            src_plane_offset = src_block_base +
                               static_cast<size_t>(layer) * bytes_per_block +
                               plane_offset;
            dst_plane_offset =
                static_cast<size_t>(gpu_block_idx + k_or_v * num_blocks_tot) *
                bytes_per_plane;
        }

        const uint8_t* src = src_base + src_plane_offset;
        uint8_t* dst = dst_base + dst_plane_offset;

        // Copy cooperatively across threads
        for (size_t i = tid; i < bytes_per_plane; i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    // ----------------------------------------------------------------------
    // Case 2: layers_before_blocks == true (but kv_before_blocks == false)
    // Copy entire block for this layer
    // ----------------------------------------------------------------------
    if (layers_before_blocks) {
        if (is_put) {
            // PUT: GPU → CPU
            src_block_offset =
                static_cast<size_t>(gpu_block_idx) * bytes_per_block;
            dst_block_offset =
                src_block_base + static_cast<size_t>(layer) * bytes_per_block;
        } else {
            // GET: CPU → GPU
            src_block_offset =
                src_block_base + static_cast<size_t>(layer) * bytes_per_block;
            dst_block_offset =
                static_cast<size_t>(gpu_block_idx) * bytes_per_block;
        }

        const uint8_t* src = src_base + src_block_offset;
        uint8_t* dst = dst_base + dst_block_offset;

        // Copy cooperatively across threads
        for (size_t i = tid; i < bytes_per_block; i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    // ----------------------------------------------------------------------
    // Case 3: layers_before_blocks == false
    // Entire block contains all layers - copy block once
    // ----------------------------------------------------------------------
    if (is_put) {
        // PUT: GPU → CPU
        src_block_offset = static_cast<size_t>(gpu_block_idx) * bytes_per_block;
        dst_block_offset = src_block_base;
    } else {
        // GET: CPU → GPU
        src_block_offset = src_block_base;
        dst_block_offset = static_cast<size_t>(gpu_block_idx) * bytes_per_block;
    }

    const uint8_t* src = src_base + src_block_offset;
    uint8_t* dst = dst_base + dst_block_offset;

    // Copy cooperatively across threads
    for (size_t i = tid; i < bytes_per_block; i += blockDim.x) {
        dst[i] = src[i];
    }
}

//----------------------------------------------------------------------
// Copy Implementation Functions
//----------------------------------------------------------------------
// Standard cudaMemcpyAsync path (DMA-based copying)
void copy_via_cuda_memcpy(uint8_t* cpu_base,
                          const std::vector<torch::Tensor>& gpu_tensors,
                          const std::vector<int64_t>& block_ids_list,
                          const CacheLayout& layout,
                          const c10::cuda::CUDAStream& stream,
                          int num_blocks_in_file,
                          bool is_put) {
    cudaMemcpyKind kind =
        is_put ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    // Direct pointer arithmetic - no indexing operations
    for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
        int64_t gpu_block_idx = block_ids_list[bi];
        size_t cpu_block_base = compute_cpu_offset(bi,
                                                   is_put,
                                                   gpu_tensors.size(),
                                                   layout.bytes_per_block,
                                                   gpu_block_idx,
                                                   num_blocks_in_file);
        if (layout.layers_before_blocks) {  // Standard layout
            for (size_t layer = 0; layer < gpu_tensors.size(); ++layer) {
                uint8_t* gpu_base =
                    reinterpret_cast<uint8_t*>(gpu_tensors[layer].data_ptr());
                size_t block_offset = gpu_block_idx * layout.bytes_per_block;

                if (layout.kv_before_blocks) {
                    size_t plane = layout.kv_bytes_per_plane;
                    // Compute GPU and CPU offsets for K
                    void* src_K = is_put ? (gpu_base + block_offset)
                                         : (cpu_base + cpu_block_base);
                    void* dst_K = is_put ? (cpu_base + cpu_block_base)
                                         : (gpu_base + block_offset);
                    cudaError_t err1 = cudaMemcpyAsync(dst_K,
                                                       src_K,
                                                       plane,
                                                       kind,
                                                       stream.stream());
                    TORCH_CHECK(err1 == cudaSuccess,
                                "cudaMemcpyAsync failed (K)");

                    // Compute GPU and CPU offsets for V
                    void* src_V = is_put ? (gpu_base + block_offset + plane)
                                         : (cpu_base + cpu_block_base + plane);
                    void* dst_V = is_put ? (cpu_base + cpu_block_base + plane)
                                         : (gpu_base + block_offset + plane);
                    cudaError_t err2 = cudaMemcpyAsync(dst_V,
                                                       src_V,
                                                       plane,
                                                       kind,
                                                       stream.stream());
                    TORCH_CHECK(err2 == cudaSuccess,
                                "cudaMemcpyAsync failed (V)");
                } else {  // One copy for both K and V
                    void* src = is_put ? (gpu_base + block_offset)
                                       : (cpu_base + cpu_block_base);
                    void* dst = is_put ? (cpu_base + cpu_block_base)
                                       : (gpu_base + block_offset);
                    cudaError_t err = cudaMemcpyAsync(dst,
                                                      src,
                                                      layout.bytes_per_block,
                                                      kind,
                                                      stream.stream());
                    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
                }
            }
        } else {  // Cross-layer layout- one copy for all layers
            uint8_t* gpu_base =
                reinterpret_cast<uint8_t*>(gpu_tensors[0].data_ptr());
            size_t gpu_offset = gpu_block_idx * layout.bytes_per_block;
            uint8_t* cpu_ptr = cpu_base + cpu_block_base;

            const void* src = is_put ? (gpu_base + gpu_offset) : cpu_ptr;
            void* dst = is_put ? cpu_ptr : (gpu_base + gpu_offset);
            cudaError_t err = cudaMemcpyAsync(dst,
                                              src,
                                              layout.bytes_per_block,
                                              kind,
                                              stream.stream());
            TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
        }
    }
}

// GPU kernel-based copy path (uses CUDA threads for copying)
void copy_via_kernel(uint8_t* cpu_base,
                     const std::vector<torch::Tensor>& gpu_tensors,
                     const std::vector<int64_t>& block_ids_list,
                     const CacheLayout& layout,
                     const c10::cuda::CUDAStream& stream,
                     bool is_put,
                     int num_blocks_in_file) {
    const int num_layers = static_cast<int>(gpu_tensors.size());

    // Calculate CPU buffer offset for each block (maps global block ID to local
    // file offset)
    std::vector<int64_t> cpu_offsets(block_ids_list.size());
    for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
        cpu_offsets[bi] = compute_cpu_offset(bi,
                                             is_put,
                                             num_layers,
                                             layout.bytes_per_block,
                                             block_ids_list[bi],
                                             num_blocks_in_file);
    }

    // Wrap block IDs in tensor and copy to GPU for kernel access
    torch::Tensor block_ids_tensor = to_gpu_tensor(block_ids_list);

    // Wrap CPU offsets in tensor and copy to GPU for kernel access
    torch::Tensor cpu_offsets_tensor = to_gpu_tensor(cpu_offsets);

    // Map CPU memory to device pointer (required for GPU kernel to write to
    // host memory - zero-copy)
    uint8_t* cpu_base_dev = cpu_base;
    if (is_put) {
        cudaError_t map_err =
            cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0);
        TORCH_CHECK(map_err == cudaSuccess,
                    "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(map_err));
    }
    if (layout.layers_before_blocks) {  // standard layout
        dim3 grid(block_ids_list.size(),
                  layout.kv_before_blocks ? 2 : 1);  // (blocks, K/V)
        dim3 block(COPY_THREADS);

        // Launch copy kernel for all the blocks on each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            uint8_t* gpu_ptr =
                reinterpret_cast<uint8_t*>(gpu_tensors[layer].data_ptr());

            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                is_put ? gpu_ptr : cpu_base,      // Source
                is_put ? cpu_base_dev : gpu_ptr,  // Destination
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                layout.num_blocks,
                layout.kv_bytes_per_plane,
                layout.bytes_per_block,
                layout.kv_before_blocks,
                layout.layers_before_blocks,
                is_put);
        }

        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
                    "Kernel launch failed: ",
                    cudaGetErrorString(launch_err));

    } else {  // / Cross-layer layout (each block contains all layers)
        TORCH_CHECK(!layout.kv_before_blocks,
                    "Invalid layout: kv_before_blocks=true but "
                    "layers_before_blocks=false");

        dim3 grid(block_ids_list.size(), 1);
        dim3 block(COPY_THREADS);

        uint8_t* gpu_ptr =
            reinterpret_cast<uint8_t*>(gpu_tensors[0].data_ptr());
        // Launch the copy kernel once for all blocks across all layers
        copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
            is_put ? gpu_ptr : cpu_base,      // Source
            is_put ? cpu_base_dev : gpu_ptr,  // Destination
            block_ids_tensor.data_ptr<int64_t>(),
            cpu_offsets_tensor.data_ptr<int64_t>(),
            block_ids_list.size(),
            /*layer=*/0,  // Ignored in cross-layer layout
            layout.num_blocks,
            layout.kv_bytes_per_plane,
            layout.bytes_per_block,
            layout.kv_before_blocks,
            layout.layers_before_blocks,
            is_put);

        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
                    "Kernel launch failed: ",
                    cudaGetErrorString(launch_err));
    }
}

// Main transfer function - dispatches to kernel or memcpy path
void transfer_kv_blocks(
    uint8_t* cpu_base,
    const std::vector<torch::Tensor>& gpu_tensors,
    const std::vector<int64_t>& block_ids_list,
    const CacheLayout& layout,
    const c10::cuda::CUDAStream& stream,
    bool is_put,
    bool use_kernel,
    int num_blocks_in_file = 0  // default is 0, ignored for PUT
) {
    if (use_kernel) {
        copy_via_kernel(cpu_base,
                        gpu_tensors,
                        block_ids_list,
                        layout,
                        stream,
                        is_put,
                        num_blocks_in_file);
    } else {
        copy_via_cuda_memcpy(cpu_base,
                             gpu_tensors,
                             block_ids_list,
                             layout,
                             stream,
                             is_put,
                             num_blocks_in_file);
    }
}

//----------------------------------------------------------------------
// GPU → Storage (PUT)
//----------------------------------------------------------------------

// Copy selected GPU K/V blocks into a single staging CPU buffer.
// The staging buffer is returned as a CPU tensor (no copy, just a view).
torch::Tensor copy_gpu_tensors_to_buffer(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream) {
    TORCH_CHECK(!src_tensors.empty(), "Source tensors list is empty");
    const auto& ref = src_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "src_tensors must be contiguous");

    // Extract tensor geometry
    auto layout =
        CacheLayout::from_tensor(ref,
                                 g_connector_config.block_axis,
                                 g_connector_config.kv_before_blocks,
                                 g_connector_config.layers_before_blocks);
    const int num_layers = static_cast<int>(src_tensors.size());

    // Total required staging memory
    const size_t total_bytes =
        block_ids_list.size() * num_layers * layout.bytes_per_block;

    // Fetch or allocate thread-local staging buffer
    StagingBufferInfo buf = get_thread_local_staging_buffer(total_bytes);
    TORCH_CHECK(buf.size >= total_bytes,
                "staging buffer too small: need ",
                total_bytes,
                " got ",
                buf.size);

    auto dtype = ref.dtype();
    const int64_t total_elements =
        static_cast<int64_t>(total_bytes / layout.elem_size);

    // Wrap staging buffer as tensor view (no copy)
    torch::Tensor result_cpu = torch::from_blob(buf.ptr,
                                                {total_elements},
                                                torch::TensorOptions()
                                                    .dtype(dtype)
                                                    .device(torch::kCPU)
                                                    .pinned_memory(true));

    auto* cpu_base = static_cast<uint8_t*>(buf.ptr);

    // Check environment variable to determine copy method
    bool use_kernel = get_env_flag("USE_KERNEL_COPY_WRITE", false);
    bool is_put = true;

    // Execute the copy operation
    transfer_kv_blocks(cpu_base,
                       src_tensors,
                       block_ids_list,
                       layout,
                       stream,
                       is_put,
                       use_kernel);

    // Reinterpret bfloat16 tensor as uint16_t for safe raw byte access (I/O or
    // memcpy)
    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

//----------------------------------------------------------------------
// Storage → GPU (GET)
//----------------------------------------------------------------------

bool copy_buffer_to_gpu_tensors(torch::Tensor cpu_buf,
                                const std::vector<int64_t>& block_ids_list,
                                const std::vector<torch::Tensor>& dst_tensors,
                                int num_blocks_in_file,
                                const c10::cuda::CUDAStream& stream) {
    TORCH_CHECK(!dst_tensors.empty(), "Destination tensors list is empty");
    const auto& ref = dst_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "dst_tensors must be contiguous");
    TORCH_CHECK(cpu_buf.is_contiguous(), "cpu buffer must be contiguous");

    // Verify cpu_buf is pinned memory
    TORCH_CHECK(cpu_buf.is_pinned(),
                "cpu_buf must be pinned memory for kernel-based copy");

    // Extract tensor geometry
    auto layout =
        CacheLayout::from_tensor(ref,
                                 g_connector_config.block_axis,
                                 g_connector_config.kv_before_blocks,
                                 g_connector_config.layers_before_blocks);
    const int num_layers = static_cast<int>(dst_tensors.size());

    auto* cpu_base = cpu_buf.data_ptr<uint8_t>();

    // Check environment variable to determine copy method (default: kernel for
    // READ)
    bool use_kernel = get_env_flag("USE_KERNEL_COPY_READ", false);
    bool is_put = false;

    // Execute the copy operation
    transfer_kv_blocks(cpu_base,
                       dst_tensors,
                       block_ids_list,
                       layout,
                       stream,
                       is_put,
                       use_kernel,
                       num_blocks_in_file);

    return true;
}
