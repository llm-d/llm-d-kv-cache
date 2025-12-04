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

#include <string>
#include <torch/extension.h>

// Write a tensor to disk using a temporary file and atomic rename
bool write_tensor_to_file(const torch::Tensor& host_buf,
                          const std::string& target_path);

// Read a file into a CPU tensor using the thread-local staging buffer
bool read_tensor_from_file(const std::string& path, torch::Tensor& host_buf);

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path);
