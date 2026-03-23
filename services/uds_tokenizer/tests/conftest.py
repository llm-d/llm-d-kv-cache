# Copyright 2026 The llm-d Authors.
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

"""Shared pytest fixtures for UDS tokenizer tests."""

import asyncio
import os
import tempfile
import threading
from collections.abc import Iterator

import grpc
import pytest

import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc
from tokenizer_service.tokenizer import TokenizerService
from tokenizer_service.renderer import RendererService
from tokenizer_grpc_service import create_grpc_server


DEFAULT_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def test_model() -> str:
    """Return the model name to use for tests, from env var or default."""
    return os.getenv("TEST_MODEL", DEFAULT_TEST_MODEL)


@pytest.fixture(scope="session")
def uds_socket_path() -> Iterator[str]:
    """Return a unique UDS socket path with cleanup.

    Uses /tmp with a short name to avoid macOS 103-char limit.
    """
    # Create temp directory - auto-cleanup on exit
    with tempfile.TemporaryDirectory(prefix="tok-") as socket_dir:
        socket_path = f"{socket_dir}/uds.sock"
        yield socket_path


@pytest.fixture(scope="session")
def grpc_server(uds_socket_path: str) -> Iterator[None]:
    """Start an async gRPC server in a background event loop for the test session."""
    tokenizer_service = TokenizerService()
    renderer_service = RendererService()

    async def _start():
        server = create_grpc_server(
            tokenizer_service,
            uds_socket_path,
            renderer_service,
        )
        await server.start()
        return server

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        server = asyncio.run_coroutine_threadsafe(_start(), loop).result(timeout=30)
    except Exception:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
        raise

    yield

    asyncio.run_coroutine_threadsafe(server.stop(grace=5), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture(scope="session")
def grpc_channel(grpc_server: None, uds_socket_path: str) -> Iterator[grpc.Channel]:
    """Create a gRPC channel connected to the test server."""
    channel = grpc.insecure_channel(f"unix://{uds_socket_path}")
    try:
        grpc.channel_ready_future(channel).result(timeout=10.0)
    except grpc.FutureTimeoutError:
        channel.close()
        raise RuntimeError(f"gRPC channel to {uds_socket_path} not ready within 10s")
    yield channel
    channel.close()


@pytest.fixture(scope="session")
def grpc_stub(grpc_channel: grpc.Channel) -> tokenizer_pb2_grpc.TokenizationServiceStub:
    """Create a ``TokenizationService`` stub."""
    return tokenizer_pb2_grpc.TokenizationServiceStub(grpc_channel)
