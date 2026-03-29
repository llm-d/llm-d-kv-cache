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
#include <cstddef>

struct StagingBufferInfo {
  void* ptr = nullptr;
  size_t size = 0;
};

// Storage mode for file I/O operations
enum class StorageMode {
  CPU_BUFFER_STAGE,  // GPU → CPU buffer → File (traditional)
  GDS_DIRECT,        // GPU → File direct (GPUDirect Storage)
  GDS_BOUNCE_BUFFER  // GPU → GDS bounce buffer → File
};

// GDS operation mode - controls which operations use GDS
enum class GdsMode {
  DISABLED,       // GDS disabled, use CPU staging for both read and write
  READ_ONLY,      // GDS for reads only, CPU staging for writes
  WRITE_ONLY,     // GDS for writes only, CPU staging for reads
  READ_WRITE,     // GDS for both reads and writes
  BB_READ_ONLY,   // GDS with Bounce Buffer for reads only, CPU staging for
                  // writes
  BB_WRITE_ONLY,  // GDS with Bounce Buffer for writes only, CPU staging for
                  // reads
  BB_READ_WRITE   // GDS with Bounce Buffer for both reads and writes
};
