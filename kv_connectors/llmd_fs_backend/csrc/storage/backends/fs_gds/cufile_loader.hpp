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

#include <dlfcn.h>
#include <cstddef>
#include <cstdint>

// cuFile operation status codes
enum CUfileOpError { CU_FILE_SUCCESS = 0 };

// File handle types for cuFileHandleRegister
enum CUfileFileHandleType { CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1 };

// Buffer registration flag for RDMA-capable memory
#define CU_FILE_RDMA_REGISTER 1

// Return type for most cuFile API calls
struct CUfileError_t {
  CUfileOpError err;
};

// Forward declaration for filesystem operations table
struct CUfileFSOps_t;

// File descriptor passed to cuFileHandleRegister (layout must match cufile.h)
struct CUfileDescr_t {
  CUfileFileHandleType type;
  union {
    int fd;
    void* handle;
  } handle;
  const CUfileFSOps_t* fs_ops;
};

// Opaque handle returned by cuFileHandleRegister
typedef void* CUfileHandle_t;

// Driver properties returned by cuFileDriverGetProperties
struct CUfileDrvProps_t {
  size_t max_device_cache_size;
  size_t max_device_pinned_mem_size;
  char _reserved[256];
};

// CuFileApi is a runtime wrapper for the NVIDIA cuFile (GDS) library,
// so the same wheel works with or without GDS. Loads libcufile.so via dlopen.
// Function pointers are resolved via dlsym by symbol name (e.g. "cuFileRead").
// Singleton — library is loaded once and function pointers are reused.
class CuFileApi {
 public:
  static CuFileApi& instance() {
    static CuFileApi loader;
    return loader;
  }

  bool is_loaded() const { return m_handle != nullptr; }

  // Function signature types — define what each cuFile function looks like
  using FnDriverOpen = CUfileError_t (*)();
  using FnDriverClose = CUfileError_t (*)();
  using FnGetVersion = CUfileError_t (*)(int*);
  using FnDriverGetProperties = CUfileError_t (*)(CUfileDrvProps_t*);
  using FnBufRegister = CUfileError_t (*)(const void*, size_t, int);
  using FnBufDeregister = CUfileError_t (*)(const void*);
  using FnHandleRegister = CUfileError_t (*)(CUfileHandle_t*, CUfileDescr_t*);
  using FnHandleDeregister = void (*)(CUfileHandle_t);
  using FnRead = ssize_t (*)(CUfileHandle_t, void*, size_t, off_t, off_t);
  using FnWrite =
      ssize_t (*)(CUfileHandle_t, const void*, size_t, off_t, off_t);

  // Resolved function pointers — null if library not loaded, filled by
  // constructor via dlsym
  FnDriverOpen cuFileDriverOpen = nullptr;
  FnDriverClose cuFileDriverClose = nullptr;
  FnGetVersion cuFileGetVersion = nullptr;
  FnDriverGetProperties cuFileDriverGetProperties = nullptr;
  FnBufRegister cuFileBufRegister = nullptr;
  FnBufDeregister cuFileBufDeregister = nullptr;
  FnHandleRegister cuFileHandleRegister = nullptr;
  FnHandleDeregister cuFileHandleDeregister = nullptr;
  FnRead cuFileRead = nullptr;
  FnWrite cuFileWrite = nullptr;

 private:
  void* m_handle = nullptr;

  // Attempts to load libcufile.so and resolve all function symbols.
  // If the library or any symbol is missing, m_handle stays null.
  CuFileApi() {
    m_handle = dlopen("libcufile.so", RTLD_NOW);
    if (!m_handle) {
      // Try versioned name
      m_handle = dlopen("libcufile.so.0", RTLD_NOW);
    }
    if (!m_handle) return;

    cuFileDriverOpen =
        reinterpret_cast<FnDriverOpen>(dlsym(m_handle, "cuFileDriverOpen"));
    cuFileDriverClose =
        reinterpret_cast<FnDriverClose>(dlsym(m_handle, "cuFileDriverClose"));
    cuFileGetVersion =
        reinterpret_cast<FnGetVersion>(dlsym(m_handle, "cuFileGetVersion"));
    cuFileDriverGetProperties = reinterpret_cast<FnDriverGetProperties>(
        dlsym(m_handle, "cuFileDriverGetProperties"));
    cuFileBufRegister =
        reinterpret_cast<FnBufRegister>(dlsym(m_handle, "cuFileBufRegister"));
    cuFileBufDeregister = reinterpret_cast<FnBufDeregister>(
        dlsym(m_handle, "cuFileBufDeregister"));
    cuFileHandleRegister = reinterpret_cast<FnHandleRegister>(
        dlsym(m_handle, "cuFileHandleRegister"));
    cuFileHandleDeregister = reinterpret_cast<FnHandleDeregister>(
        dlsym(m_handle, "cuFileHandleDeregister"));
    cuFileRead = reinterpret_cast<FnRead>(dlsym(m_handle, "cuFileRead"));
    cuFileWrite = reinterpret_cast<FnWrite>(dlsym(m_handle, "cuFileWrite"));

    // Verify all symbols resolved
    if (!cuFileDriverOpen || !cuFileDriverClose || !cuFileGetVersion ||
        !cuFileBufRegister || !cuFileBufDeregister || !cuFileHandleRegister ||
        !cuFileHandleDeregister || !cuFileRead || !cuFileWrite) {
      dlclose(m_handle);
      m_handle = nullptr;
    }
  }

  // Unloads the library when the process exits
  ~CuFileApi() {
    if (m_handle) {
      dlclose(m_handle);
    }
  }
};
