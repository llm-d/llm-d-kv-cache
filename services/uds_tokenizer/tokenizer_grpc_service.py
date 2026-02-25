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
import json
import threading

import os
import sys

from readerwriterlock import rwlock
# Ensure current directory is on sys.path for protobuf imports
sys.path.append(os.path.dirname(__file__))

# Import protobuf-generated modules
import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc
from utils.thread_pool_utils import get_thread_pool_size

# Add the preprocessing directory to the Python path to import tokenizer_wrapper
runtime_path = '/app/preprocessing/chat_completions'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
dev_path = os.path.join(os.path.dirname(os.path.dirname(current_file_dir)), 'pkg', 'preprocessing', 'chat_completions')
dev_path = os.path.normpath(dev_path)  # Normalize the path to resolve '..'

if runtime_path not in sys.path:
    sys.path.insert(0, runtime_path)
if dev_path not in sys.path:
    sys.path.insert(0, dev_path)

# Import the tokenizer wrapper functions
from tokenizer_wrapper import render, render_chat, get_or_create_tokenizer_key


class TokenizationServiceServicer(tokenizer_pb2_grpc.TokenizationServiceServicer):
    """Synchronous gRPC service implementation class, optimized for CPU-intensive operations"""

    def __init__(self):
        self._model_to_key_map = {}
        self._map_lock = rwlock.RWLockWrite()  # Reader-writer lock for thread-safe access

    def _get_tokenizer_key(self, model_name):
        """Thread-safe method to get tokenizer key for a model name"""
        with self._map_lock.gen_rlock():
            return self._model_to_key_map.get(model_name)

    def _set_tokenizer_key(self, model_name, tokenizer_key):
        """Thread-safe method to set tokenizer key for a model name"""
        with self._map_lock.gen_wlock():
            self._model_to_key_map[model_name] = tokenizer_key

    def _has_model(self, model_name):
        """Thread-safe method to check if model is initialized"""
        with self._map_lock.gen_rlock():
            return model_name in self._model_to_key_map

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
            for key, val in value.struct_value.fields.items():
                result[key] = self._protobuf_value_to_python(val)
            return result
        else:
            return None

    def Render(self, request, context):
        """Implement the synchronous Render RPC method"""
        try:
            # Get tokenizer key from model name mapping
            model_name = request.model_name
            if not self._has_model(model_name):
                # Model not initialized, raise gRPC error
                logging.warning(f"Model {request.model_name} not initialized, cannot render")
                context.abort(grpc.StatusCode.INTERNAL, f"Model {model_name} not initialized")
                return tokenizer_pb2.RenderResponse()

            tokenizer_key = self._get_tokenizer_key(model_name)

            # Prepare request for Python wrapper function
            render_request = {
                "key": tokenizer_key,
                "text": request.text,
                "add_special_tokens": request.add_special_tokens
            }

            # Call the Python render function directly
            result_json = render(json.dumps(render_request))
            logging.debug(f"Render result: {result_json}")
            result_data = json.loads(result_json)

            input_ids = result_data.get('input_ids', [])
            offset_mapping = result_data.get('offset_mapping', [])

            # Create offset_pairs format (flattened array of [start, end, start, end, ...])
            offset_pairs = []
            for offset in offset_mapping:
                offset_pairs.extend([int(offset[0]), int(offset[1])])

            response = tokenizer_pb2.RenderResponse(
                input_ids=list(input_ids),
                offset_pairs=offset_pairs,
                success=True
            )

            return response

        except Exception as e:
            logging.error(f"Render failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def RenderChat(self, request, context):
        """Implement the synchronous RenderChat RPC method"""
        try:
            # Get tokenizer key from model name mapping
            model_name = request.model_name
            if not self._has_model(model_name):
                # Model not initialized, raise gRPC error
                logging.warning(f"Model {request.model_name} not initialized, cannot render chat")
                context.abort(grpc.StatusCode.INTERNAL, f"Model {model_name} not initialized")
                return tokenizer_pb2.RenderResponse()

            tokenizer_key = self._get_tokenizer_key(model_name)

            # Convert conversation messages (list of ChatMessage objects) to the expected format
            conversation = []
            for msg in request.conversation:
                conversation.append({"role": msg.role, "content": msg.content})

            # Convert tools from protobuf Value array to Python objects
            tools = []
            for tool in request.tools:
                tools.append(self._protobuf_value_to_python(tool))

            # Convert documents from protobuf Value array to Python objects
            documents = []
            for document in request.documents:
                documents.append(self._protobuf_value_to_python(document))

            # Convert chat_template_kwargs from protobuf format to dict
            chat_template_kwargs = {}
            for key, value in request.chat_template_kwargs.items():
                chat_template_kwargs[key] = self._protobuf_value_to_python(value)

            # Prepare request for Python wrapper function
            render_chat_request = {
                "key": tokenizer_key,
                "conversation": conversation,
                "return_assistant_tokens_mask": request.return_assistant_tokens_mask,
                "continue_final_message": request.continue_final_message,
                "add_generation_prompt": request.add_generation_prompt,
                "chat_template_kwargs": chat_template_kwargs
            }

            # Add optional fields if they exist
            if request.HasField("chat_template"):
                render_chat_request["chat_template"] = request.chat_template

            if tools:
                render_chat_request["tools"] = tools

            if documents:
                render_chat_request["documents"] = documents

            # Call the Python render_chat function directly
            result_json = render_chat(json.dumps(render_chat_request))
            logging.debug(f"RenderChat result: {result_json}")
            result_data = json.loads(result_json)

            input_ids = result_data.get('input_ids', [])
            offset_mapping = result_data.get('offset_mapping', [])

            # Create offset_pairs format (flattened array of [start, end, start, end, ...])
            offset_pairs = []
            for offset in offset_mapping:
                offset_pairs.extend([int(offset[0]), int(offset[1])])

            response = tokenizer_pb2.RenderResponse(
                input_ids=list(input_ids),
                offset_pairs=offset_pairs,
                success=True
            )

            return response

        except Exception as e:
            logging.error(f"RenderChat failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def InitializeTokenizer(self, request, context):
        """Implement the synchronous InitializeTokenizer RPC method"""
        try:
            logging.info(f"Initializing tokenizer for model: {request.model}")

            # Check if tokenizer is already initialized for this model
            model_name = request.model
            if self._has_model(model_name):
                logging.info(f"Tokenizer for model {request.model} already initialized")
                response = tokenizer_pb2.InitializeTokenizerResponse(
                    success=True
                )
                return response

            # Create tokenizer key request using parameters from the gRPC request
            tokenizer_request = {
                "is_local": request.is_local,
                "model": request.model,
                "revision": request.revision if request.HasField("revision") else None,
                "token": request.token if request.HasField("token") else os.getenv("HF_TOKEN", ""),
                "download_dir": request.download_dir if request.HasField("download_dir") else None
            }

            # Create the tokenizer key which will cache the tokenizer
            tokenizer_key = get_or_create_tokenizer_key(json.dumps(tokenizer_request))

            # Store the mapping from model name to tokenizer key
            self._set_tokenizer_key(model_name, tokenizer_key)

            # If we reach here, the tokenizer was successfully created/cached
            response = tokenizer_pb2.InitializeTokenizerResponse(
                success=True
            )

            return response

        except Exception as e:
            logging.error(f"Tokenizer initialization failed: {e}", exc_info=True)
            return tokenizer_pb2.InitializeTokenizerResponse(
                success=False,
                error_message=str(e)
            )


def create_grpc_server(uds_socket_path: str, thread_pool, tcp_port: str = ""):
    """Create a synchronous gRPC server.
    
    Args:
        tokenizer_service: The tokenizer service implementation
        uds_socket_path: Path to Unix Domain Socket
        thread_pool: ThreadPoolExecutor for handling requests
        tcp_port: TCP port for testing only (leave empty for production)
    """
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
    servicer = TokenizationServiceServicer()

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

    # Bind to UDS (production)
    server.add_insecure_port(f"unix://{uds_socket_path}")
    logging.info(f"Synchronous gRPC server configured to listen on {uds_socket_path}")

    # Optionally bind to TCP port (FOR TESTING ONLY)
    if tcp_port:
        server.add_insecure_port(f"0.0.0.0:{tcp_port}")
        logging.warning(f"TCP mode enabled on port {tcp_port} - FOR TESTING ONLY, DO NOT USE IN PRODUCTION")

    return server