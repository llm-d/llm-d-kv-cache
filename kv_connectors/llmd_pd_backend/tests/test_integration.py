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

"""Integration tests: real NIXL/UCX transfers over TCP loopback with CPU buffers.

Uses real NIXL with UCX transport fallback to TCP. Tests byte-for-byte correctness
and clean handle and memory teardown without GPU or RDMA hardware requirements.
"""

from __future__ import annotations

import ctypes
import time

import pytest

try:
    from nixl._api import nixl_agent, nixl_agent_config

    HAS_NIXL = True
except ImportError:
    HAS_NIXL = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_NIXL, reason="nixl not available"),
]

BLOCK_SIZE = 4096
NUM_BLOCKS = 4


def _create_agent(name: str) -> nixl_agent:
    cfg = nixl_agent_config(backends=[])
    agent = nixl_agent(name, cfg)
    agent.create_backend("UCX", {})
    return agent


class TestPDTransferTCPLoopback:
    """End-to-end NIXL transfer between simulated P and D agents."""

    def test_single_block_transfer(self):
        prefill_agent = _create_agent("prefill_single")
        decode_agent = _create_agent("decode_single")

        pattern = bytes(range(256)) * (BLOCK_SIZE // 256)
        prefill_buf = bytearray(pattern)
        decode_buf = bytearray(BLOCK_SIZE)

        p_addr = ctypes.addressof(ctypes.c_char.from_buffer(prefill_buf))
        d_addr = ctypes.addressof(ctypes.c_char.from_buffer(decode_buf))

        prefill_desc = prefill_agent.register_memory(
            [(p_addr, BLOCK_SIZE, 0, "p_buf")], "DRAM", ["UCX"]
        )
        decode_desc = decode_agent.register_memory(
            [(d_addr, BLOCK_SIZE, 0, "d_buf")], "DRAM", ["UCX"]
        )

        try:
            # Metadata handshake
            p_md = prefill_agent.get_agent_metadata()
            d_md = decode_agent.get_agent_metadata()

            p_remote_name = decode_agent.add_remote_agent(p_md)
            d_remote_name = prefill_agent.add_remote_agent(d_md)

            decode_agent.make_connection(p_remote_name, ["UCX"])
            prefill_agent.make_connection(d_remote_name, ["UCX"])

            # Decode pulls from prefill via NIXL READ
            d_local_xfer = decode_agent.get_xfer_descs(
                [(d_addr, BLOCK_SIZE, 0)], "DRAM"
            )
            p_remote_xfer = decode_agent.get_xfer_descs(
                [(p_addr, BLOCK_SIZE, 0)], "DRAM"
            )

            xfer = decode_agent.initialize_xfer(
                "READ", d_local_xfer, p_remote_xfer, p_remote_name, backends=["UCX"]
            )
            assert xfer is not None

            state = decode_agent.transfer(xfer)
            assert state != "ERR"

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                state = decode_agent.check_xfer_state(xfer)
                if state == "DONE":
                    break
                time.sleep(0.001)
            else:
                pytest.fail("Transfer did not complete within 5s")

            assert decode_buf == prefill_buf, "Byte-for-byte data mismatch"
            decode_agent.release_xfer_handle(xfer)

        finally:
            prefill_agent.deregister_memory(prefill_desc)
            decode_agent.deregister_memory(decode_desc)

    def test_multiple_block_transfer(self):
        prefill_agent = _create_agent("prefill_multi")
        decode_agent = _create_agent("decode_multi")

        total_size = BLOCK_SIZE * NUM_BLOCKS
        prefill_buf = bytearray(total_size)
        decode_buf = bytearray(total_size)

        for i in range(NUM_BLOCKS):
            offset = i * BLOCK_SIZE
            prefill_buf[offset : offset + BLOCK_SIZE] = bytes([i & 0xFF]) * BLOCK_SIZE

        p_addr = ctypes.addressof(ctypes.c_char.from_buffer(prefill_buf))
        d_addr = ctypes.addressof(ctypes.c_char.from_buffer(decode_buf))

        prefill_desc = prefill_agent.register_memory(
            [(p_addr, total_size, 0, "p_multi")], "DRAM", ["UCX"]
        )
        decode_desc = decode_agent.register_memory(
            [(d_addr, total_size, 0, "d_multi")], "DRAM", ["UCX"]
        )

        try:
            p_md = prefill_agent.get_agent_metadata()
            d_md = decode_agent.get_agent_metadata()

            p_remote_name = decode_agent.add_remote_agent(p_md)
            d_remote_name = prefill_agent.add_remote_agent(d_md)

            decode_agent.make_connection(p_remote_name, ["UCX"])
            prefill_agent.make_connection(d_remote_name, ["UCX"])

            for i in range(NUM_BLOCKS):
                offset = i * BLOCK_SIZE
                d_local_xfer = decode_agent.get_xfer_descs(
                    [(d_addr + offset, BLOCK_SIZE, 0)], "DRAM"
                )
                p_remote_xfer = decode_agent.get_xfer_descs(
                    [(p_addr + offset, BLOCK_SIZE, 0)], "DRAM"
                )

                xfer = decode_agent.initialize_xfer(
                    "READ", d_local_xfer, p_remote_xfer, p_remote_name, backends=["UCX"]
                )
                assert xfer is not None

                state = decode_agent.transfer(xfer)
                assert state != "ERR"

                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    state = decode_agent.check_xfer_state(xfer)
                    if state == "DONE":
                        break
                    time.sleep(0.001)
                else:
                    pytest.fail(f"Block {i} transfer did not complete")

                decode_agent.release_xfer_handle(xfer)

            assert decode_buf == prefill_buf, "Multi-block byte-for-byte mismatch"

        finally:
            prefill_agent.deregister_memory(prefill_desc)
            decode_agent.deregister_memory(decode_desc)

    def test_clean_teardown(self):
        agent = _create_agent("teardown_agent")
        buf = bytearray(BLOCK_SIZE)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))

        desc = agent.register_memory([(addr, BLOCK_SIZE, 0, "t_buf")], "DRAM", ["UCX"])
        agent.deregister_memory(desc)
