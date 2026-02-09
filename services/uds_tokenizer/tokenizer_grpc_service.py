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

"""Synchronous gRPC service for tokenizer operations optimized for CPU-intensive tasks."""

import grpc
from grpc_reflection.v1alpha import reflection
import logging

import os
import sys
# Ensure current directory is on sys.path for protobuf imports
sys.path.append(os.path.dirname(__file__))

# Import protobuf-generated modules
import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc
from tokenizer_service.tokenizer import TokenizerService
from utils.thread_pool_utils import get_thread_pool_size


class TokenizationServiceServicer(tokenizer_pb2_grpc.TokenizationServiceServicer):
    """Synchronous gRPC service implementation class, optimized for CPU-intensive operations"""

    def __init__(self, tokenizer_service: TokenizerService):
        self.tokenizer_service = tokenizer_service
        logging.info("TokenizationServiceServicer initialized")

    def Tokenize(self, request, context):
        """Implement the synchronous Tokenize RPC method"""
        try:
            # logging.info(f"Received tokenize request for model: {request.model_name}")

            # Use tokenizer_service for tokenization, with add_special_tokens from request
            batch_encoding = self.tokenizer_service.tokenize_and_process(
                request.input,
                request.add_special_tokens,
                request.model_name
            )

            # Convert result format
            input_ids = batch_encoding['input_ids']
            offset_mapping = batch_encoding.get('offset_mapping', [])

            # Create offset_pairs format (flattened array of [start, end, start, end, ...])
            offset_pairs = []
            for offset in offset_mapping:
                offset_pairs.extend([int(offset[0]), int(offset[1])])

            response = tokenizer_pb2.TokenizeResponse(
                input_ids=list(input_ids),
                offset_pairs=offset_pairs,  # Only use offset_pairs field
                success=True
            )

            # logging.info(f"Tokenization completed with {len(input_ids)} tokens")
            return response

        except Exception as e:
            logging.error(f"Tokenization failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def RenderChatTemplate(self, request, context):
        """Implement the synchronous RenderChatTemplate RPC method"""
        try:
            # logging.info(f"Received chat template request")

            # Convert the nested conversation turns to a flat list of messages
            messages = []
            for turn in request.conversation_turns:
                for msg in turn.messages:
                    messages.append({"role": msg.role, "content": msg.content})

            # Call tokenizer_service method with model name
            prompt = self.tokenizer_service.apply_template(messages, request.model_name)

            response = tokenizer_pb2.ChatTemplateResponse(
                rendered_prompt=prompt,
                success=True
            )

            # logging.info(f"Chat template rendered successfully")
            return response

        except Exception as e:
            logging.error(f"Chat template rendering failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def InitializeTokenizer(self, request, context):
        """Implement the synchronous InitializeTokenizer RPC method"""
        try:
            logging.info(f"Initializing tokenizer for model: {request.model_name}")

            success = self.tokenizer_service.load_tokenizer(
                request.model_name,
                request.enable_thinking,
                request.add_generation_prompt
            )

            if success:
                response = tokenizer_pb2.InitializeTokenizerResponse(
                    success=True
                )
            else:
                response = tokenizer_pb2.InitializeTokenizerResponse(
                    success=False,
                    error_message=f"Failed to initialize tokenizer for model: {request.model_name}"
                )

            return response

        except Exception as e:
            logging.error(f"Tokenizer initialization failed: {e}", exc_info=True)
            return tokenizer_pb2.InitializeTokenizerResponse(
                success=False,
                error_message=str(e)
            )


def create_grpc_server(tokenizer_service: TokenizerService, uds_socket_path: str, thread_pool):
    """Create a synchronous gRPC server"""
    # Create synchronous gRPC server with optimized configuration for multi-threaded processing
    server = grpc.server(
        thread_pool,
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            # Performance optimizations
            ('grpc.keepalive_time_ms', 7200000),  # 2 hours
            ('grpc.keepalive_timeout_ms', 20000),  # 20 seconds
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 300000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ('grpc.http2.max_frame_size', 8192),
            ('grpc.max_concurrent_streams', get_thread_pool_size() * 2),  # Adjust concurrent streams based on CPU cores
        ]
    )

    # Create service implementation
    servicer = TokenizationServiceServicer(tokenizer_service)

    # Register service
    tokenizer_pb2_grpc.add_TokenizationServiceServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools (only if explicitly enabled)
    # Reflection increases the exposed surface area, so it's disabled by default
    enable_reflection = os.getenv('ENABLE_GRPC_REFLECTION', '')
    if enable_reflection:
        SERVICE_NAMES = (
            tokenizer_pb2.DESCRIPTOR.services_by_name['TokenizationService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        logging.info("gRPC reflection enabled for service discovery")
    else:
        logging.info("gRPC reflection disabled (set `ENABLE_GRPC_REFLECTION=1` to enable)")

    # Bind to UDS
    server.add_insecure_port(f"unix://{uds_socket_path}")

    logging.info(f"Synchronous gRPC server configured to listen on {uds_socket_path}")

    return server