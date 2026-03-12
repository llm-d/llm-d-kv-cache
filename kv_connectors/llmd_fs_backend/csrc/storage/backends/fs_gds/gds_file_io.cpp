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

#include "gds_file_io.hpp"
#include "file_io.hpp"
#include "logger.hpp"

#include <torch/extension.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

// Constructor
GdsFileIO::GdsFileIO(const std::vector<std::pair<void*, size_t>>& gpu_buffers,
                     size_t block_size,
                     GdsMode gds_mode,
                     TensorCopier& tensor_copier)
    : m_gds_initialized(false),
      m_gds_mode(gds_mode),
      m_tensor_copier(tensor_copier),
      m_use_for_read(gds_mode == GdsMode::READ_ONLY ||
                     gds_mode == GdsMode::READ_WRITE ||
                     gds_mode == GdsMode::BB_READ_ONLY ||
                     gds_mode == GdsMode::BB_READ_WRITE),
      m_use_for_write(gds_mode == GdsMode::WRITE_ONLY ||
                      gds_mode == GdsMode::READ_WRITE ||
                      gds_mode == GdsMode::BB_WRITE_ONLY ||
                      gds_mode == GdsMode::BB_READ_WRITE) {
  if (!is_gds_supported()) {
    FS_LOG_INFO("GdsFileIO: GDS not supported, using CPU buffer staging");
    return;
  }

  if (initialize_gds()) {
    FS_LOG_INFO("GdsFileIO: GPUDirect Storage (GDS) enabled");

    // Register GPU buffers with optional per-block registration
    size_t total_bytes = 0;
    for (const auto& [ptr, size] : gpu_buffers) {
      if (!register_gpu_buffer(ptr, size, block_size)) {
        FS_LOG_WARN("GdsFileIO: Failed to register buffer " << ptr);
      } else {
        total_bytes += size;
      }
    }
    FS_LOG_INFO("GdsFileIO: Registered " << gpu_buffers.size() << " buffers ("
                                         << (total_bytes / (1024.0 * 1024.0))
                                         << " MB total)");
  } else {
    FS_LOG_WARN(
        "GdsFileIO: GDS initialization failed, using CPU "
        "buffer staging");
  }
}

// Destructor - cleanup all GDS resources
GdsFileIO::~GdsFileIO() {
#ifdef USE_CUFILE
  if (!m_gds_initialized) {
    return;
  }

  // Deregister all buffers
  for (const auto& [ptr, size] : m_registered_buffers) {
    cuFileBufDeregister(ptr);
  }
  m_registered_buffers.clear();

  // Close driver
  CUfileError_t status = cuFileDriverClose();
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_WARN(
        "GdsFileIO: cuFileDriverClose failed with error code: " << status.err);
  }

  m_gds_initialized = false;
#endif
}

// Helper to check if current mode uses Bounce Buffer
bool GdsFileIO::is_bb_mode() const {
  return m_gds_mode == GdsMode::BB_READ_ONLY ||
         m_gds_mode == GdsMode::BB_WRITE_ONLY ||
         m_gds_mode == GdsMode::BB_READ_WRITE;
}

// Static capability check
bool GdsFileIO::is_gds_supported() {
#ifdef USE_CUFILE
  // Check if cuFile library is available
  CUfileError_t status;

  // Try to get cuFile version as a simple availability check
  int version = 0;
  status = cuFileGetVersion(&version);

  if (status.err != CU_FILE_SUCCESS) {
    return false;
  }

  // Check if GPU supports GDS
  int device_id = 0;
  cudaError_t cuda_err = cudaGetDevice(&device_id);
  if (cuda_err != cudaSuccess) {
    return false;
  }

  // Check GPU capability
  int gds_supported = 0;
  cuda_err = cudaDeviceGetAttribute(&gds_supported,
                                    cudaDevAttrGPUDirectRDMASupported,
                                    device_id);

  if (cuda_err != cudaSuccess || gds_supported == 0) {
    return false;
  }

  return true;
#else
  return false;
#endif
}

// Initialize GDS driver - opens cuFile driver and queries capabilities
bool GdsFileIO::initialize_gds() {
#ifdef USE_CUFILE
  CUfileError_t status = cuFileDriverOpen();  // Initialize cuFile driver (must
                                              // be called once per process)

  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR(
        "GdsFileIO: cuFileDriverOpen failed with error code: " << status.err);
    return false;
  }

  m_gds_initialized = true;  // Mark as initialized for subsequent operations

  // Query and log driver capabilities (optional, for debugging/monitoring)
  CUfileDrvProps_t props;
  status = cuFileDriverGetProperties(&props);
  if (status.err == CU_FILE_SUCCESS) {
    FS_LOG_INFO("GdsFileIO: cuFile driver properties:\n"
                << "  - max_device_cache_size: " << props.max_device_cache_size
                << "\n"  // Device cache limit
                << "  - max_device_pinned_mem_size: "
                << props.max_device_pinned_mem_size);
  }

  return true;
#else
  return false;
#endif
}

// Register GPU buffer with cuFile for optimized DMA transfers
bool GdsFileIO::register_gpu_buffer(void* gpu_ptr,
                                    size_t size,
                                    size_t block_size) {
#ifdef USE_CUFILE
  if (!m_gds_initialized) {
    return false;
  }

  if (m_registered_buffers.find(gpu_ptr) != m_registered_buffers.end()) {
    return true;  // Skip if already registered
  }

  FS_LOG_DEBUG("cuFileBufRegister: gpu_ptr "
               << gpu_ptr << " size " << size << " bytes ("
               << (size / (1024.0 * 1024.0)) << " MB)"
               << (block_size > 0
                       ? " block_size " + std::to_string(block_size) + " bytes"
                       : ""));

  // Check if current GDS mode uses Bounce Buffer
  bool use_bb = is_bb_mode();

  // If block_size is 0 or BB mode is enabled, register entire buffer at once
  if (block_size == 0 || use_bb) {
    FS_LOG_DEBUG("GDS" << (use_bb ? " with BB mode" : "")
                       << ": Registering entire buffer"
                       << ": ptr " << gpu_ptr << " size "
                       << (size / (1024.0 * 1024.0)) << " MB");
    CUfileError_t status = cuFileBufRegister(gpu_ptr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
      FS_LOG_WARN("GdsFileIO: cuFileBufRegister failed with error code: "
                  << status.err);
      return false;
    }

    m_registered_buffers[gpu_ptr] = size;
    return true;
  }

  // The CHUNK_MULTIPLIER to group blocks into larger chunks and reduce
  // GDS registration table entries (e.g. CHUNK_MULTIPLIER = TARGET_CHUNK_SIZE /
  // block_size)
  const size_t CHUNK_MULTIPLIER = 1;  // currently register one block at a time
                                      // due to GDS driver limitations
  const size_t chunk_size = block_size * CHUNK_MULTIPLIER;
  size_t num_chunks = (size + chunk_size - 1) / chunk_size;

  FS_LOG_DEBUG("GDS mode: Registering " << num_chunks << " chunks of "
                                        << (chunk_size / (1024.0 * 1024.0))
                                        << " MB each (CHUNK_MULTIPLIER "
                                        << CHUNK_MULTIPLIER << ")");

  for (size_t i = 0; i < num_chunks; i++) {
    void* block_ptr = static_cast<uint8_t*>(gpu_ptr) + (i * chunk_size);

    CUfileError_t status =
        cuFileBufRegister(block_ptr, chunk_size, CU_FILE_RDMA_REGISTER);
    if (status.err != CU_FILE_SUCCESS) {
      FS_LOG_WARN("GdsFileIO: cuFileBufRegister failed for block "
                  << i << " with error code: " << status.err);
      return false;
    }

    m_registered_buffers[block_ptr] = chunk_size;
  }
  FS_LOG_DEBUG("GDS mode: Successfully registered "
               << num_chunks << " chunks (total " << (size / (1024.0 * 1024.0))
               << " MB)");
  return true;
#else
  return false;
#endif
}

// StorageHandler interface: Write blocks to file
bool GdsFileIO::write_blocks_to_file(const std::string& file_path,
                                     const std::vector<int64_t>& block_ids,
                                     cudaStream_t stream) {
  // Each ThreadPool thread has its own CUDA stream, but cuFileWrite is
  // synchronous and operates directly on the device — no stream needed.
  (void)stream;
  // Get tensors and block size from tensor copier
  const auto& tensors = m_tensor_copier.get_tensors();
  size_t block_size = m_tensor_copier.get_block_size();

#ifdef USE_CUFILE
  // Create parent directory if needed
  fs::path path(file_path);
  fs::path parent_dir = path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("GdsFileIO: Failed to create directories: " << e.what());
    return false;
  }

  // Open file once with O_RDWR and O_DIRECT for GDS
  int fd = open(file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
  if (fd < 0) {
    FS_LOG_ERROR("GdsFileIO: Failed to open file "
                 << file_path << ": " << std::strerror(errno)
                 << " (errno=" << errno << ")");
    return false;
  }

  // Register file descriptor with cuFile driver for DMA setup
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  // Configure the descriptor with our file descriptor
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // Register the file with cuFile driver
  CUfileHandle_t handle;
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR("GdsFileIO: cuFileHandleRegister failed with error code: "
                 << status.err);
    close(fd);
    return false;
  }

  // Write all blocks sequentially
  bool success = true;
  off_t file_offset = 0;

  for (size_t bi = 0; bi < block_ids.size() && success; ++bi) {
    int64_t gpu_block_idx = block_ids[bi];

    for (const auto& tensor : tensors) {
      // Calculate GPU pointer (base + offset)
      void* gpu_base_ptr = tensor.data_ptr();
      void* actual_gpu_ptr =
          static_cast<uint8_t*>(gpu_base_ptr) + (gpu_block_idx * block_size);

      // Write this block's data for this layer
      ssize_t bytes_written =
          cuFileWrite(handle, actual_gpu_ptr, block_size, file_offset, 0);

      if (bytes_written < 0) {
        FS_LOG_ERROR("GdsFileIO: cuFileWrite failed with error: "
                     << bytes_written << " at file_offset=" << file_offset);
        success = false;
        break;
      } else if (bytes_written != static_cast<ssize_t>(block_size)) {
        FS_LOG_ERROR("GdsFileIO: Incomplete write: "
                     << bytes_written << " / " << block_size
                     << " bytes at file_offset=" << file_offset);
        success = false;
        break;
      }

      file_offset += block_size;
    }
  }

  cuFileHandleDeregister(handle);
  close(fd);

  return success;
#else
  return false;
#endif
}

// StorageHandler interface: Read blocks from file
bool GdsFileIO::read_blocks_from_file(const std::string& file_path,
                                      const std::vector<int64_t>& block_ids,
                                      cudaStream_t stream) {
  // Each ThreadPool thread has its own CUDA stream, but cuFileRead is
  // synchronous and operates directly on the device — no stream needed.
  (void)stream;
  // Get tensors and block size from tensor copier
  const auto& tensors = m_tensor_copier.get_tensors();
  size_t block_size = m_tensor_copier.get_block_size();

#ifdef USE_CUFILE
  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    FS_LOG_ERROR("GdsFileIO: Failed to open file "
                 << file_path << ": " << std::strerror(errno)
                 << " (errno=" << errno << ")");
    return false;
  }

  // Register file descriptor with cuFile driver
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  // Configure the descriptor with our file descriptor
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // register the file with cuFile driver
  CUfileHandle_t handle;
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR("GdsFileIO: cuFileHandleRegister failed with error code: "
                 << status.err);
    close(fd);
    return false;
  }

  // Read all blocks sequentially
  bool success = true;
  off_t file_offset = 0;

  for (size_t bi = 0; bi < block_ids.size() && success; ++bi) {
    int64_t gpu_block_idx = block_ids[bi];

    for (const auto& tensor : tensors) {
      // Calculate GPU pointer (base + offset)
      void* gpu_base_ptr = tensor.data_ptr();
      void* actual_gpu_ptr =
          static_cast<uint8_t*>(gpu_base_ptr) + (gpu_block_idx * block_size);

      // Read this block's data for this layer
      ssize_t bytes_read =
          cuFileRead(handle, actual_gpu_ptr, block_size, file_offset, 0);

      if (bytes_read < 0) {
        FS_LOG_ERROR("GdsFileIO: cuFileRead failed with error: "
                     << bytes_read << " at file_offset=" << file_offset);
        success = false;
        break;
      } else if (bytes_read != static_cast<ssize_t>(block_size)) {
        FS_LOG_ERROR("GdsFileIO: Incomplete read: "
                     << bytes_read << " / " << block_size
                     << " bytes at file_offset=" << file_offset);
        success = false;
        break;
      }

      file_offset += block_size;
    }
  }

  cuFileHandleDeregister(handle);
  close(fd);

  return success;
#else
  return false;
#endif
}

// Helper function to parse GDS mode string
GdsMode parse_gds_mode(const std::string& gds_mode_str) {
  if (gds_mode_str == "read_only") {
    return GdsMode::READ_ONLY;
  } else if (gds_mode_str == "write_only") {
    return GdsMode::WRITE_ONLY;
  } else if (gds_mode_str == "read_write") {
    return GdsMode::READ_WRITE;
  } else if (gds_mode_str == "bb_read_only") {
    return GdsMode::BB_READ_ONLY;
  } else if (gds_mode_str == "bb_write_only") {
    return GdsMode::BB_WRITE_ONLY;
  } else if (gds_mode_str == "bb_read_write") {
    return GdsMode::BB_READ_WRITE;
  } else {
    // Default to DISABLED for any other value including "disabled"
    if (gds_mode_str != "disabled") {
      FS_LOG_WARN(
          "Unknown GDS mode '"
          << gds_mode_str
          << "', defaulting to 'disabled'. "
             "Valid options: disabled, read_only, write_only, read_write, "
             "bb_read_only, bb_write_only, bb_read_write");
    }
    return GdsMode::DISABLED;
  }
}
