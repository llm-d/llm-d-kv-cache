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

"""Creating the shared region, with vLLM's SharedOffloadRegion.

The External Primary Tier is the region's creator: it creates the segment files
with vLLM's own segment split and pre-faults every page, so the region's memory
is allocated in RAM and charged to the External Primary Tier's pod. It never
calls ``cleanup()``, which would unlink the files; the files outlive the process.
"""

import os
import uuid

from vllm.v1.kv_offload.cpu.shared_offload_region import (
    REGION_SEGMENTS,
    SharedOffloadRegion,
)

from external_primary_tier.protocol import ShmFileRegion

REGION_FILE_NAME = "external-primary-tier.mmap"

# Engines in other pods open the files through a group they share with the
# External Primary Tier's pod.
_FILE_MODE = 0o660


def create_region(
    region_dir: str, pool_bytes: int, kv_bytes_per_chunk: int
) -> tuple[SharedOffloadRegion, ShmFileRegion]:
    """Create and pre-fault the region; return it and its descriptor.

    The region object keeps the files mapped for the life of the External
    Primary Tier.
    """
    num_chunks = pool_bytes // kv_bytes_per_chunk
    if num_chunks < 1:
        raise ValueError(f"pool of {pool_bytes} bytes holds no chunk of {kv_bytes_per_chunk} bytes")
    region = SharedOffloadRegion(
        engine_id="",  # unused with mmap_path
        num_chunks=num_chunks,
        rank=None,  # no rank: pre-fault the whole region
        kv_bytes_per_chunk=kv_bytes_per_chunk,
        cpu_page_size=kv_bytes_per_chunk,
        num_segments=REGION_SEGMENTS,
        mmap_path=os.path.join(region_dir, REGION_FILE_NAME),
    )
    # The segments are SharedOffloadRegion's own split; engines check theirs
    # against this list.
    segments = [(path, offset, length) for path, offset, length in region._segments]
    for path, _, _ in segments:
        os.chmod(path, _FILE_MODE)
    descriptor = ShmFileRegion(
        region_id=uuid.uuid4().bytes,
        num_chunks=num_chunks,
        kv_bytes_per_chunk=kv_bytes_per_chunk,
        segments=segments,
    )
    return region, descriptor
