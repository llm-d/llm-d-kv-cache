# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Source files
sources = [
    "csrc/storage/storage_offload.cpp",
    "csrc/storage/storage_offload_bindings.cpp",
    "csrc/storage/numa_utils.cpp",
    "csrc/storage/backends/fs_io/file_io.cpp",
    "csrc/storage/thread_pool.cpp",
    "csrc/storage/tensor_copier.cu",
    "csrc/storage/tensor_copier_kernels.cu",
    "csrc/storage/backends/fs_gds/gds_file_io.cpp",
]

# Include directories
base_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [
    os.path.join(base_dir, "csrc/storage"),
    os.path.join(base_dir, "csrc/storage/backends/fs_io"),
    os.path.join(base_dir, "csrc/storage/backends/fs_gds"),
]

# Libraries and compile flags
# GDS (libcufile) is loaded at runtime via dlopen — no build-time dependency
libraries = ["numa", "dl"]
cxx_args = ["-O3", "-std=c++17", "-fopenmp"]
nvcc_args = [
    "-O3",
    "-std=c++17",
    "-Xcompiler",
    "-std=c++17",
    "-Xcompiler",
    "-fopenmp",
]

setup(
    name="llmd_fs_connector",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "storage_offload",
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
