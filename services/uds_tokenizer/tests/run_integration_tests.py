#!/usr/bin/env python3
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

"""
Integration test runner that waits for the gRPC server to be ready before running tests.
"""

import grpc
import time
import sys
import os

# Import protobuf-generated modules
# Add the parent directory to the path to find the protobuf modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc

def wait_for_server(socket_path="/tmp/tokenizer/tokenizer-uds.socket", max_wait_time=60):
    """Wait for the gRPC server to be ready by checking the Unix Domain Socket."""

    # Wait for the socket file to be created
    start_time = time.time()
    while not os.path.exists(socket_path):
        if time.time() - start_time > max_wait_time:
            print(f"Server did not start within {max_wait_time} seconds")
            return False
        print("Waiting for server to start...")
        time.sleep(1)

    print("Socket file created, testing gRPC connection...")
        # After the socket exists, wait for the gRPC channel to become ready.
    channel = grpc.insecure_channel(f"unix://{socket_path}")
    try:
        while True:
            elapsed = time.time() - start_time
            remaining = max_wait_time - elapsed
            if remaining <= 0:
                print(f"gRPC server did not become ready within {max_wait_time} seconds")
                return False
            # Wait for the channel to be ready, with a bounded per-attempt timeout.
            wait_timeout = min(5, max(0.1, remaining))
            future = grpc.channel_ready_future(channel)
            try:
                future.result(timeout=wait_timeout)
                print("gRPC server is ready to accept connections.")
                return True
            except grpc.FutureTimeoutError:
                print("Socket exists but gRPC server not ready yet, retrying...")
                time.sleep(1)
    finally:
        channel.close()


def run_tests():
    """Run the gRPC integration tests."""
    # Unix Domain Socket path
    uds_path = "/tmp/tokenizer/tokenizer-uds.socket"

    # Create gRPC channel using Unix Domain Socket
    channel = grpc.insecure_channel(f"unix://{uds_path}")

    # Create gRPC stub
    stub = tokenizer_pb2_grpc.TokenizationServiceStub(channel)

    print("=== Testing gRPC TokenizationService ===")

    # 1. Test Tokenize endpoint
    print("\n1. Testing tokenize...")
    try:
        request = tokenizer_pb2.TokenizeRequest(
            input="Hello, how are you?",
            model_name="test-model",  # Optional field
            add_special_tokens=True   # Optional field
        )
        response = stub.Tokenize(request)
        print(f"Tokenize result: {list(response.input_ids)[:10]}...")  # Print first 10 tokens
        print(f"Success: {response.success}")
        print(f"Number of tokens: {len(response.input_ids)}")

        # Check if tokenization was successful
        if response.success and len(response.input_ids) > 0:
            print("✓ Tokenize test passed")
        else:
            print("✗ Tokenize test failed")
            return False

    except grpc.RpcError as e:
        print(f"Error tokenizing via gRPC: {e}")
        return False

    # 2. Test RenderChatTemplate endpoint
    print("\n2. Testing chat template...")
    try:
        # Create messages for chat template
        messages = [
            tokenizer_pb2.ChatMessage(role="system", content="You are a helpful assistant."),
            tokenizer_pb2.ChatMessage(role="user", content="Hello, how are you?")
        ]

        request = tokenizer_pb2.ChatTemplateRequest(
            messages=messages,
            add_generation_prompt=True
        )
        response = stub.RenderChatTemplate(request)

        print(f"Chat template result: {response.rendered_prompt[:100]}...")
        print(f"Success: {response.success}")

        # Check if chat template rendering was successful
        if response.success and len(response.rendered_prompt) > 0:
            print("✓ Chat template test passed")
        else:
            print("✗ Chat template test failed")
            return False

    except grpc.RpcError as e:
        print(f"Error applying chat template via gRPC: {e}")
        return False

    # 3. Test Tokenize with different parameters
    print("\n3. Testing tokenize with different parameters...")
    try:
        request = tokenizer_pb2.TokenizeRequest(
            input="This is another test sentence.",
            add_special_tokens=False
        )
        response = stub.Tokenize(request)

        print(f"Tokenize result: {list(response.input_ids)[:10]}...")
        print(f"Success: {response.success}")
        print(f"Number of tokens: {len(response.input_ids)}")

        # Check if tokenization was successful
        if response.success and len(response.input_ids) > 0:
            print("✓ Tokenize with parameters test passed")
        else:
            print("✗ Tokenize with parameters test failed")
            return False

    except grpc.RpcError as e:
        print(f"Error tokenizing with parameters via gRPC: {e}")
        return False

    # 4. Test health check endpoint (via TCP port) using HTTP client
    print("\n4. Testing health check...")
    try:
        import requests
        response = requests.get("http://localhost:8082/health")  # Changed to 8082 to match server
        health_data = response.json()
        print(f"Health check result: {health_data}")
        print(f"Status code: {response.status_code}")

        # Check if health check was successful
        if response.status_code == 200 and health_data.get("status") == "healthy":
            print("✓ Health check test passed")
        else:
            print("✗ Health check test failed")
            return False

    except Exception as e:
        print(f"Error checking health: {e}")
        return False

    print("\nAll gRPC tests passed!")
    return True


def main():
    """Main function."""
    # Wait for server to be ready
    if not wait_for_server():
        print("Server failed to start or become ready")
        sys.exit(1)

    # Run tests
    success = run_tests()

    if success:
        print("\nAll gRPC integration tests passed!")
        sys.exit(0)
    else:
        print("\nSome gRPC integration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()