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
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

// Forward declare cuFile types to avoid header dependency when GDS is not
// available
#ifdef USE_CUFILE
  #include <cufile.h>
#else
typedef void* CUfileHandle_t;
typedef struct CUfileDescr_st {
  int handle;
  int type;
} CUfileDescr_t;
#endif

#include "storage_types.hpp"
#include "storage_handler.hpp"
#include "tensor_copier.hpp"

// GDS File I/O class - unified interface for both storage modes
class GdsFileIO : public StorageHandler {
 public:
  // Constructor: registers gpu_buffers with GDS;
  // block_size controls per-block registration (0 = entire buffer)
  GdsFileIO(const std::vector<std::pair<void*, size_t>>& gpu_buffers,
            size_t block_size,
            GdsMode gds_mode,
            TensorCopier& tensor_copier);

  // Destructor: deregisters GPU buffers and releases GDS resources
  ~GdsFileIO();

  // Check if GDS is available and initialized
  bool is_gds_available() const { return m_gds_initialized; }

  // Static capability check - can be called before construction
  static bool is_gds_supported();

  // Helper to check if current mode uses Bounce Buffer
  bool is_bb_mode() const;

  // Returns true if GDS should be used for read operations
  bool use_for_read() const { return m_use_for_read; }
  // Returns true if GDS should be used for write operations
  bool use_for_write() const { return m_use_for_write; }

  // Write block_ids from GPU tensors directly to dst_file via GDS
  bool write_blocks_to_file(const std::string& dst_file,
                            const std::vector<int64_t>& block_ids,
                            cudaStream_t stream) override;

  // Read block_ids from src_file directly into GPU tensors via GDS
  bool read_blocks_from_file(const std::string& src_file,
                             const std::vector<int64_t>& block_ids,
                             cudaStream_t stream) override;

  // Returns StorageMode::GDS_DIRECT
  StorageMode get_mode() const override { return StorageMode::GDS_DIRECT; }

 private:
  // GDS initialization state
  bool m_gds_initialized;
  // GDS mode
  GdsMode m_gds_mode;
  // Reference to tensor copier
  TensorCopier& m_tensor_copier;
  // Computed flags for read/write usage based on mode
  bool m_use_for_read;
  bool m_use_for_write;

  // Registered GPU buffers (for GDS mode)
  std::unordered_map<void*, size_t> m_registered_buffers;

  // Initialize GDS driver
  bool initialize_gds();

  // Register GPU buffer for GDS; buffers are auto-deregistered in destructor
  // block_size: if > 0, register in blocks of this size
  bool register_gpu_buffer(void* gpu_ptr, size_t size, size_t block_size);
};

// Parse a GDS mode string (e.g. "read_write") into a GdsMode enum value.
// Returns GdsMode::DISABLED for unrecognized strings.
GdsMode parse_gds_mode(const std::string& gds_mode_str);
