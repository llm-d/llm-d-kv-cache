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

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <vector>
#include <algorithm>
#include <sys/sysinfo.h>

#include "numa_utils.hpp"
#include "debug_utils.hpp"

// Return NUMA node associated with a given GPU
int get_gpu_numa_node(int device_id) {
    int numa_node = -1;

    cudaError_t err =
        cudaDeviceGetAttribute(&numa_node, cudaDevAttrHostNumaId, device_id);
    if (err != cudaSuccess) {
        std::cerr << "[WARN] Failed to query NUMA node for GPU " << device_id
                  << ": " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    return numa_node;
}

// Return list of CPU cores that belong to a NUMA node
std::vector<int> get_cpus_in_numa_node(int node) {
    std::vector<int> cpus;

    if (node < 0) {
        std::cerr << "[WARN] Requested NUMA node " << node
                  << " (negative index) – returning empty CPU list\n";
        return cpus;
    }

    // Sysfs cpulist path for this NUMA node
    std::string path =
        "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] NUMA node " << node
                  << " cpulist not found or not readable: " << path << "\n";
        return cpus;
    }

    // Read full cpulist line (format: "0-13,84-97")
    std::string line;
    if (!std::getline(f, line) || line.empty()) {
        std::cerr << "[WARN] NUMA node " << node
                  << " cpulist is empty or unreadable: " << path << "\n";
        return cpus;
    }

    // Parse comma-separated tokens (ranges like "0-13" or single "7")
    size_t start = 0;
    while (start < line.size()) {
        // Extract next comma-separated token
        size_t comma = line.find(',', start);
        std::string token = line.substr(
            start,
            (comma == std::string::npos ? std::string::npos : comma - start));

        if (!token.empty()) {
            size_t dash = token.find('-');
            try {
                if (dash != std::string::npos) {
                    // Range form: "a-b" → expand to [a,b]
                    int a = std::stoi(token.substr(0, dash));
                    int b = std::stoi(token.substr(dash + 1));
                    if (a <= b) {
                        for (int c = a; c <= b; ++c) {
                            cpus.push_back(c);
                        }
                    } else {
                        // Invalid range (start > end)
                        std::cerr << "[WARN] NUMA node " << node
                                  << " has invalid CPU range '" << token
                                  << "' (start > end) in " << path << "\n";
                    }
                } else {
                    // Single CPU: "7"
                    cpus.push_back(std::stoi(token));
                }
            } catch (const std::exception& e) {
                // Malformed token (non-numeric, out-of-range, etc.): ignore
                std::cerr << "[WARN] NUMA node " << node
                          << " has malformed cpulist token '" << token
                          << "' in " << path << ": " << e.what() << "\n";
            }
        }

        if (comma == std::string::npos) break;
        start = comma + 1;
    }

    return cpus;
}