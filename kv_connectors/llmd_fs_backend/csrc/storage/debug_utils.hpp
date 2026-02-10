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

#include "logger.hpp"
#include <chrono>
#include <sstream>

// -------------------------------------
// Debugging and timing macros
// -------------------------------------

// Timing macro - measures execution time when STORAGE_CONNECTOR_DEBUG  is set
// and not "0"
#define TIME_EXPR(label, expr, ...)                                     \
  ([&]() -> bool {                                                      \
    if (FSLogger::level() > LogLevel::DEBUG) {                          \
      return ((expr), true);                                            \
    }                                                                   \
    auto __t0 = std::chrono::high_resolution_clock::now();              \
    auto __ret = [&]() {                                                \
      if constexpr (std::is_void_v<decltype(expr)>) {                   \
        (expr);                                                         \
        return true;                                                    \
      } else {                                                          \
        return (expr);                                                  \
      }                                                                 \
    }();                                                                \
    auto __t1 = std::chrono::high_resolution_clock::now();              \
    double __ms =                                                       \
        std::chrono::duration<double, std::milli>(__t1 - __t0).count(); \
    std::ostringstream __fs_time_oss;                                           \
    __fs_time_oss << "[TIME] " << label << " took " << __ms << " ms";           \
    __VA_OPT__(__fs_time_oss << " | "; [&]<typename... Args>(Args&&... args) {  \
      ((__fs_time_oss << args), ...);                                           \
    }(__VA_ARGS__);)                                                            \
    FS_LOG_DEBUG(__fs_time_oss.str());                                          \
    return __ret;                                                       \
  })()

// Helper for reading environment variable flags
inline bool get_env_flag(const char* name, bool default_val) {
  const char* env = std::getenv(name);
  if (!env) return default_val;

  std::string v(env);
  if (v == "1" || v == "true" || v == "TRUE") return true;
  if (v == "0" || v == "false" || v == "FALSE") return false;

  return default_val;
}
