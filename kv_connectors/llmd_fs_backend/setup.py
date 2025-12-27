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

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="llmd_fs_connector",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "storage_offload",
             sources=[
                "src/csrc/storage/storage_offload.cu",
                "src/csrc/storage/buffer.cpp",
                "src/csrc/storage/file_io.cpp",
                "src/csrc/storage/thread_pool.cpp",
                "src/csrc/storage/tensor_copy.cu",
            ],
            libraries=['numa', 'cuda'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-std=c++17","-Xcompiler", "-fopenmp"]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
