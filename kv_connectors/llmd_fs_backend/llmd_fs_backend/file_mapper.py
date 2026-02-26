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


class FileMapper:
    """
    FileMapper maps KV blocks (given by their hash) to file names.
    """

    def __init__(
        self,
        root_dir: str,
        model_name: str,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        rank: int,
        dtype: str,
    ):
        """
        Initialize the file mapper for a specific worker.
        All KV data files will be nested under a unique base path.

        Base path format:
            <root_dir>
            /<model_name>
            /block_size_<gpu_block_size>_blocks_per_file_<blocks_per_file>
            /tp_<tp_size>_pp_<pp_size>_pcp_<pcp_size>
            /rank_<kv_rank>
            /<dtype>

        Args:
            root_dir: Root directory for shared storage.
            model_name: Model identifier.
            gpu_block_size: Number of tokens per GPU block.
            gpu_blocks_per_file: Number of GPU blocks stored in a single file.
            tp_size: Number of pipeline parallel groups.
            pp_size: Number of tensor parallel groups.
            pcp_size: Number of prefill context parallel groups.
            rank: Worker rank.
            dtype: Torch dtype of the KV-cache tensors.

        Returns:
            Base path under which KV-cache files are stored or loaded.
        """

        self.base_path = (
            f"{root_dir}"
            f"/{model_name}"
            f"/block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}"
            f"/tp_{tp_size}_pp_size_{pp_size}_pcp_size_{pcp_size}"
            f"/rank_{rank}"
            f"/{dtype}"
        )

    def get_file_name(self, block_hash: int | bytes) -> str:
        """
        Return the file path for a KV block.
        The path is built using hash-based subdirectories:
        <base>/<hhh>/<hh>/<hash>.bin, to limit directory fan-out.

        Args:
            block_hash: Hash identifying the KV-cache block (int or bytes).

        Returns:
            Full file path for the given block.
        """
        if isinstance(block_hash, bytes):  # convert bytes to int
            block_hash = int.from_bytes(block_hash, "little")
        assert isinstance(block_hash, int)

        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self.base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"
