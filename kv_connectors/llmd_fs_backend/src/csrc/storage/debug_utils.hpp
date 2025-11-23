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

#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>

// -------------------------------------
// Debugging and timing macros
// -------------------------------------

// Debug print - enabled when STORAGE_CONNECTOR_DEBUG is set and not "0"
#define DEBUG_PRINT(msg)                                      \
  do {                                                        \
    const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG"); \
    if (env && std::string(env) != "0")                       \
      std::cout << "[DEBUG] " << msg << std::endl;            \
  } while (0)

// Timing macro - measures execution time when STORAGE_CONNECTOR_DEBUG  is set
// and not "0"
#define TIME_EXPR(label, expr, info_str)                                   \
  ([&]() {                                                                 \
    const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");              \
    if (!(env && std::string(env) != "0")) {                               \
      return (expr);                                                       \
    }                                                                      \
    auto __t0 = std::chrono::high_resolution_clock::now();                 \
    auto __ret = (expr);                                                   \
    auto __t1 = std::chrono::high_resolution_clock::now();                 \
    double __ms =                                                          \
        std::chrono::duration<double, std::milli>(__t1 - __t0).count();    \
    std::cout << "[DEBUG][TIME] " << label << " took " << __ms << " ms | " \
              << info_str << std::endl;                                    \
    return __ret;                                                          \
  })()
