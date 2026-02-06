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
            # Convert the nested conversation turns to a flat list of messages
            conversation = []
            for turn in request.conversation_turns:
                messages = []
                for msg in turn.messages:
                    messages.append({"role": msg.role, "content": msg.content})
                conversation.append(messages)

            # Convert tools from protobuf format to dict
            tools = []
            for tool in request.tools:
                for entry in tool.tool:
                    tools.append({entry.key: self._protobuf_value_to_python(entry.value)})

            # Convert documents from protobuf format to dict
            documents = []
            for document in request.documents:
                for entry in document.document:
                    documents.append({entry.key: self._protobuf_value_to_python(entry.value)})

            # Convert chat_template_kwargs from protobuf format to dict
            chat_template_kwargs = {}
            for entry in request.chat_template_kwargs:
                chat_template_kwargs[entry.key] = self._protobuf_value_to_python(entry.value)

            # Call tokenizer_service method with all parameters
            prompt = self.tokenizer_service.apply_template(
                conversation,
                request.model_name,
                chat_template=request.chat_template if request.HasField("chat_template") else None,
                tools=tools if tools else None,
                documents=documents if documents else None,
                return_assistant_tokens_mask=request.return_assistant_tokens_mask,
                continue_final_message=request.continue_final_message,
                add_generation_prompt=request.add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None
            )

            response = tokenizer_pb2.ChatTemplateResponse(
                rendered_prompt=prompt,
                success=True
            )

            # logging.info(f"Chat template rendered successfully")
            return response

        except Exception as e:
            logging.error(f"Chat template rendering failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def _protobuf_value_to_python(self, value):
        """Convert protobuf Value to Python native type"""
        if value.HasField("string_value"):
            return value.string_value
        elif value.HasField("number_value"):
            return value.number_value
        elif value.HasField("bool_value"):
            return value.bool_value
        elif value.HasField("list_value"):
            return [self._protobuf_value_to_python(v) for v in value.list_value.values]
        elif value.HasField("struct_value"):
            result = {}
            for entry in value.struct_value.fields:
                result[entry.key] = self._protobuf_value_to_python(value.struct_value.fields[entry.key])
            return result
        else:
            return None

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

    # Bind to UDS
    server.add_insecure_port(f"unix://{uds_socket_path}")

    logging.info(f"Synchronous gRPC server configured to listen on {uds_socket_path}")

    return server