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

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="llmd_fs_connector",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "storage_offload",
            sources=[
                "csrc/storage/storage_offload.cpp",
                "csrc/storage/storage_offload_bindings.cpp",
                "csrc/storage/numa_utils.cpp",
                "csrc/storage/file_io.cpp",
                "csrc/storage/thread_pool.cpp",
                "csrc/storage/tensor_copier.cu",
                "csrc/storage/tensor_copier_kernels.cu",
            ],
            libraries=["numa"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fopenmp",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
