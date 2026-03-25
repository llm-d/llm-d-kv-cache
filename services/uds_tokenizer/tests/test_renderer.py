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

"""
Integration tests for the RenderChatCompletion and RenderCompletion gRPC methods.

These tests require a running gRPC server (provided by conftest.py) and a locally
available model (controlled via the TEST_MODEL env var, default Qwen/Qwen2.5-0.5B-Instruct).

Run with:
    pytest tests/test_renderer.py -v
"""

import asyncio

import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
from tokenizer_service.renderer import RendererService
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


class TestRenderChatCompletion:
    """Tests for the RenderChatCompletion gRPC method."""

    def test_render_no_mm_features_for_text(self, grpc_stub, test_model):
        """Text-only requests should have no multimodal features."""
        resp = grpc_stub.RenderChatCompletion(
            tokenizer_pb2.RenderChatCompletionRequest(
                model_name=test_model,
                messages=[
                    tokenizer_pb2.ChatMessage(role="user", content="Just text."),
                ],
            )
        )
        assert not resp.HasField("features")

    def test_render_deterministic(self, grpc_stub, test_model):
        """The same request rendered twice produces identical token IDs."""
        req = tokenizer_pb2.RenderChatCompletionRequest(
            model_name=test_model,
            messages=[
                tokenizer_pb2.ChatMessage(role="user", content="Determinism check."),
            ],
        )
        resp1 = grpc_stub.RenderChatCompletion(req)
        resp2 = grpc_stub.RenderChatCompletion(req)
        assert list(resp1.token_ids) == list(resp2.token_ids)

    def test_render_matches_direct(self, grpc_stub, test_model):
        """RenderChatCompletion token IDs match a direct RendererService call."""
        messages_proto = [
            tokenizer_pb2.ChatMessage(role="user", content="What is 2+2?"),
            tokenizer_pb2.ChatMessage(role="assistant", content="4"),
            tokenizer_pb2.ChatMessage(role="user", content="And 3+3?"),
        ]
        grpc_resp = grpc_stub.RenderChatCompletion(
            tokenizer_pb2.RenderChatCompletionRequest(
                model_name=test_model,
                messages=messages_proto,
            )
        )
        assert grpc_resp.request_id
        messages_json = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        renderer_service = RendererService()
        direct = asyncio.run(
            renderer_service.render_chat(
                ChatCompletionRequest(model=test_model, messages=messages_json),
                test_model,
            )
        )
        assert list(grpc_resp.token_ids) == list(direct.token_ids)


class TestRenderCompletion:
    """Tests for the RenderCompletion gRPC method."""

    def test_render_deterministic(self, grpc_stub, test_model):
        """The same completion request rendered twice produces identical token IDs."""
        req = tokenizer_pb2.RenderCompletionRequest(
            model_name=test_model,
            prompt="Determinism check.",
        )
        resp1 = grpc_stub.RenderCompletion(req)
        resp2 = grpc_stub.RenderCompletion(req)
        assert list(resp1.token_ids) == list(resp2.token_ids)

    def test_render_matches_direct(self, grpc_stub, test_model):
        """RenderCompletion token IDs match a direct RendererService call."""
        prompt = "Hello world"
        grpc_resp = grpc_stub.RenderCompletion(
            tokenizer_pb2.RenderCompletionRequest(
                model_name=test_model,
                prompt=prompt,
            )
        )
        assert grpc_resp.request_id

        renderer_service = RendererService()
        direct = asyncio.run(
            renderer_service.render_completion(
                CompletionRequest(model=test_model, prompt=prompt),
                test_model,
            )
        )
        assert list(grpc_resp.token_ids) == list(direct[0].token_ids)
