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

import os
from collections.abc import Iterator
import tempfile

import grpc
import pytest

import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc
from tokenizer_service.tokenizer import TokenizerService
from tokenizer_grpc_service import create_grpc_server
from utils.thread_pool_utils import get_thread_pool


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
def tokenizer_service(uds_socket_path: str) -> Iterator[TokenizerService]:
    """Provide the TokenizerService instance used by the gRPC server."""
    service = TokenizerService()
    thread_pool = get_thread_pool()
    server = create_grpc_server(service, uds_socket_path, thread_pool)
    server.start()

    yield service

    # Graceful shutdown with matching timeout
    stop_future = server.stop(grace=5)
    stop_future.wait(timeout=5)


@pytest.fixture(scope="session")
def grpc_channel(tokenizer_service: TokenizerService, uds_socket_path: str) -> Iterator[grpc.Channel]:
    """Create a gRPC channel connected to the test server.
    
    Uses wait_for_ready to automatically retry connection until server is ready.
    """
    channel = grpc.insecure_channel(f"unix://{uds_socket_path}")
    
    # Verify channel can connect by waiting for it to be ready
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
