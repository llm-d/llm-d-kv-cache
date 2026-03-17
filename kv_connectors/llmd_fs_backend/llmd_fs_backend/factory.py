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

import storage_offload
from llmd_fs_backend.obj_backend import ObjBackend
from llmd_fs_backend.posix_backend import PosixBackend
from llmd_fs_backend.gds_backend import GdsBackend, GdsMtBackend

"""Factory for instantiating storage offload engine backends."""

def make_storage_engine(backend: str, **kwargs):
    """
    Instantiate the correct storage engine for the given backend name.
    Args:
        backend: One of "OBJ", "POSIX", "GDS", "GDS_MT", "POSIX_CPP"
        **kwargs: Forwarded to the backend constructor
    """

    if backend == "POSIX_CPP":
        return storage_offload.StorageOffloadEngine(
            kwargs["io_threads"], kwargs["gpu_blocks_per_file"], kwargs["tensors"]
        )

    _backends = {
        "OBJ":    ObjBackend,
        "POSIX":  PosixBackend,
        "GDS":    GdsBackend,
        "GDS_MT": GdsMtBackend,
    }
    cls = _backends.get(backend)
    if cls is None:
        raise ValueError(f"Unknown backend {backend!r}. Valid options: {list(_backends) + ['POSIX_CPP']}")
    return cls(**kwargs)
