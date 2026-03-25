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

"""Async gRPC service for tokenizer operations using grpc.aio."""

import asyncio
import json
import grpc
from typing import Any
from grpc_reflection.v1alpha import reflection
import logging
import os
import sys

sys.path.append(os.path.dirname(__file__))

import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc
from google.protobuf.json_format import MessageToDict
from tokenizer_service.tokenizer import TokenizerService
from tokenizer_service.renderer import RendererService
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


class TokenizationServiceServicer(tokenizer_pb2_grpc.TokenizationServiceServicer):
    def __init__(
        self, tokenizer_service: TokenizerService, renderer_service: RendererService
    ):
        self.tokenizer_service = tokenizer_service
        self.renderer_service = renderer_service
        logging.info("TokenizationServiceServicer initialized")

    async def Tokenize(
        self,
        request: tokenizer_pb2.TokenizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenizer_pb2.TokenizeResponse:
        try:
            # Use tokenizer_service for tokenization, with add_special_tokens from request
            batch_encoding = await asyncio.to_thread(
                self.tokenizer_service.tokenize_and_process,
                request.input,
                request.add_special_tokens,
                request.model_name,
            )

            # Convert result format
            input_ids: list[int] = batch_encoding["input_ids"]
            offset_mapping = batch_encoding.get("offset_mapping", [])

            # Create offset_pairs format (flattened array of [start, end, start, end, ...])
            offset_pairs: list[int] = []
            for offset in offset_mapping:
                offset_pairs.extend([int(offset[0]), int(offset[1])])

            response = tokenizer_pb2.TokenizeResponse(
                input_ids=list(input_ids),
                offset_pairs=offset_pairs,  # Only use offset_pairs field
                success=True,
            )
            return response

        except Exception as e:
            logging.error(f"Tokenization failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderChatTemplate(
        self,
        request: tokenizer_pb2.ChatTemplateRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenizer_pb2.ChatTemplateResponse:
        try:
            messages: list[dict[str, Any]] = []
            for turn in request.conversation_turns:
                for msg in turn.messages:
                    if msg.content_parts:
                        content_value = [
                            MessageToDict(part, preserving_proto_field_name=True)
                            for part in msg.content_parts
                        ]
                    else:
                        content_value = msg.content if msg.HasField("content") else None
                    messages.append({"role": msg.role, "content": content_value})

            prompt = await asyncio.to_thread(
                self.tokenizer_service.apply_template, messages, request.model_name
            )
            return tokenizer_pb2.ChatTemplateResponse(
                rendered_prompt=prompt, success=True
            )
        except Exception as e:
            logging.error(f"Chat template rendering failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def InitializeTokenizer(
        self,
        request: tokenizer_pb2.InitializeTokenizerRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenizer_pb2.InitializeTokenizerResponse:
        logging.info(f"Initializing tokenizer for model: {request.model_name}")

        renderer_success = await asyncio.to_thread(
            self.renderer_service.load_renderer, request.model_name
        )
        if not renderer_success:
            return tokenizer_pb2.InitializeTokenizerResponse(
                success=False,
                error_message=f"Failed to initialize renderer for model: {request.model_name}",
            )

        try:
            await asyncio.to_thread(
                self.tokenizer_service.load_tokenizer,
                request.model_name,
                request.enable_thinking,
                request.add_generation_prompt,
            )
        except Exception as e:
            logging.warning("Tokenizer load failed (non-critical): %s", e)

        return tokenizer_pb2.InitializeTokenizerResponse(success=True)

    @staticmethod
    def _generate_request_to_proto(
        result,
    ) -> tokenizer_pb2.RenderChatCompletionResponse:
        """Convert a vLLM GenerateRequest to a RenderChatCompletionResponse proto."""
        response = tokenizer_pb2.RenderChatCompletionResponse(
            request_id=result.request_id,
            token_ids=list(result.token_ids),
            success=True,
        )

        if result.features is not None:
            mm_hashes: dict[str, tokenizer_pb2.StringList] = {
                modality: tokenizer_pb2.StringList(values=hashes)
                for modality, hashes in result.features.mm_hashes.items()
            }
            mm_placeholders: dict[str, tokenizer_pb2.PlaceholderRangeList] = {
                modality: tokenizer_pb2.PlaceholderRangeList(
                    ranges=[
                        tokenizer_pb2.PlaceholderRange(offset=r.offset, length=r.length)
                        for r in ranges
                    ]
                )
                for modality, ranges in result.features.mm_placeholders.items()
            }
            response.features.CopyFrom(
                tokenizer_pb2.MultiModalFeatures(
                    mm_hashes=mm_hashes,
                    mm_placeholders=mm_placeholders,
                )
            )

        return response

    async def RenderChatCompletion(
        self,
        request: tokenizer_pb2.RenderChatCompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenizer_pb2.RenderChatCompletionResponse:
        """Render an OpenAI chat completion request via OpenAIServingRender."""
        try:
            request_dict = MessageToDict(request, preserving_proto_field_name=True)

            messages = request_dict.get("messages", [])
            for msg in messages:
                if "content_parts" in msg:
                    msg["content"] = msg.pop("content_parts")

            tools = (
                json.loads(request_dict["tools_json"])
                if request_dict.get("tools_json")
                else None
            )
            chat_template = request.chat_template or None
            chat_template_kwargs = (
                json.loads(request_dict["chat_template_kwargs"])
                if request_dict.get("chat_template_kwargs")
                else None
            )
            add_generation_prompt = (
                request.add_generation_prompt
                if request.HasField("add_generation_prompt")
                else True
            )

            chat_request = ChatCompletionRequest(
                model=request_dict["model_name"],
                messages=messages,
                tools=tools,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=request.continue_final_message,
                chat_template_kwargs=chat_template_kwargs,
            )
            result = await self.renderer_service.render_chat(
                chat_request, request.model_name
            )
            return self._generate_request_to_proto(result)
        except Exception as e:
            logging.error(f"RenderChatCompletion failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderCompletion(
        self,
        request: tokenizer_pb2.RenderCompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenizer_pb2.RenderCompletionResponse:
        """Render an OpenAI completion request via OpenAIServingRender."""
        try:
            completion_request = CompletionRequest(
                model=request.model_name,
                prompt=request.prompt,
            )
            results = await self.renderer_service.render_completion(
                completion_request, request.model_name
            )
            result = results[0]
            return tokenizer_pb2.RenderCompletionResponse(
                request_id=result.request_id,
                token_ids=list(result.token_ids),
                success=True,
            )
        except Exception as e:
            logging.error(f"RenderCompletion failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def create_grpc_server(
    tokenizer_service: TokenizerService,
    uds_socket_path: str,
    renderer_service: RendererService,
    tcp_port: str = "",
) -> grpc.aio.Server:
    """Create an async gRPC server.

    Args:
        tokenizer_service: The tokenizer service implementation
        uds_socket_path: Path to Unix Domain Socket
        renderer_service: The renderer service wrapping OpenAIServingRender
        tcp_port: TCP port for testing only (leave empty for production)
    """
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
            # Performance optimizations
            ("grpc.keepalive_time_ms", 7200000),  # 2 hours
            ("grpc.keepalive_timeout_ms", 20000),  # 20 seconds
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            (
                "grpc.http2.min_time_between_pings_ms",
                10000,
            ),  # 10s - tolerate frequent pings from Envoy/Istio sidecars
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ("grpc.http2.max_frame_size", 8192),
        ]
    )
    servicer = TokenizationServiceServicer(tokenizer_service, renderer_service)

    # Register service
    tokenizer_pb2_grpc.add_TokenizationServiceServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools (only if explicitly enabled)
    # Reflection increases the exposed surface area, so it's disabled by default
    enable_reflection = os.getenv("ENABLE_GRPC_REFLECTION", "")
    if enable_reflection:
        SERVICE_NAMES = (
            tokenizer_pb2.DESCRIPTOR.services_by_name["TokenizationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        logging.info("gRPC reflection enabled for service discovery")
    else:
        logging.info(
            "gRPC reflection disabled (set `ENABLE_GRPC_REFLECTION=1` to enable)"
        )

    # Bind to UDS (production)
    server.add_insecure_port(f"unix://{uds_socket_path}")
    logging.info(f"gRPC server configured to listen on {uds_socket_path}")

    # Optionally bind to TCP port (FOR TESTING ONLY)
    if tcp_port:
        server.add_insecure_port(f"0.0.0.0:{tcp_port}")
        logging.warning(f"TCP mode enabled on port {tcp_port} - FOR TESTING ONLY")

    return server
