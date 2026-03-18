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
Integration tests for the RenderChatCompletion gRPC method.

These tests require a running gRPC server (provided by conftest.py) and a locally
available model (controlled via the TEST_MODEL env var, default Qwen/Qwen2.5-0.5B-Instruct).

Run with:
    pytest tests/test_renderer.py -v
"""

import asyncio
import json

import grpc
import pytest

import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
from tokenizer_service.renderer import RendererService


def _chat_request_json(model: str, messages: list[dict]) -> str:
    """Build a minimal OpenAI ChatCompletionRequest JSON string."""
    return json.dumps({"model": model, "messages": messages})


class TestRenderChatCompletion:
    """Tests for the RenderChatCompletion gRPC method."""

    def test_render_no_mm_features_for_text(self, grpc_stub, test_model):
        """Text-only requests should have no multimodal features."""
        request_json = _chat_request_json(
            test_model,
            [{"role": "user", "content": "Just text."}],
        )
        resp = grpc_stub.RenderChatCompletion(
            tokenizer_pb2.RenderChatCompletionRequest(
                request_json=request_json,
                model_name=test_model,
            )
        )
        assert not resp.HasField("features")

    def test_render_deterministic(self, grpc_stub, test_model):
        """The same request rendered twice produces identical token IDs."""
        request_json = _chat_request_json(
            test_model,
            [{"role": "user", "content": "Determinism check."}],
        )
        req = tokenizer_pb2.RenderChatCompletionRequest(
            request_json=request_json,
            model_name=test_model,
        )
        resp1 = grpc_stub.RenderChatCompletion(req)
        resp2 = grpc_stub.RenderChatCompletion(req)
        assert list(resp1.token_ids) == list(resp2.token_ids)

    def test_render_invalid_json(self, grpc_stub, test_model):
        """RenderChatCompletion returns an error for malformed request JSON."""
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderChatCompletion(
                tokenizer_pb2.RenderChatCompletionRequest(
                    request_json="not valid json",
                    model_name=test_model,
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_render_matches_direct(self, grpc_stub, test_model):
        """RenderChatCompletion token IDs match a direct RendererService call."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        request_json = _chat_request_json(test_model, messages)
        grpc_resp = grpc_stub.RenderChatCompletion(
            tokenizer_pb2.RenderChatCompletionRequest(
                request_json=request_json,
                model_name=test_model,
            )
        )
        assert grpc_resp.request_id
        direct = asyncio.run(RendererService().render_chat(request_json, test_model))
        assert list(grpc_resp.token_ids) == list(direct.token_ids)


class TestRenderCompletion:
    """Tests for the RenderCompletion gRPC method."""

    def test_render_invalid_json(self, grpc_stub, test_model):
        """RenderCompletion returns an error for malformed request JSON."""
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderCompletion(
                tokenizer_pb2.RenderCompletionRequest(
                    request_json="not valid json",
                    model_name=test_model,
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_render_deterministic(self, grpc_stub, test_model):
        """The same completion request rendered twice produces identical token IDs."""
        request_json = json.dumps({"model": test_model, "prompt": "Determinism check."})
        req = tokenizer_pb2.RenderCompletionRequest(
            request_json=request_json,
            model_name=test_model,
        )
        resp1 = grpc_stub.RenderCompletion(req)
        resp2 = grpc_stub.RenderCompletion(req)
        assert list(resp1.items[0].token_ids) == list(resp2.items[0].token_ids)

    def test_render_matches_direct(self, grpc_stub, test_model):
        """RenderCompletion token IDs match a direct RendererService call."""
        prompts = ["Hello world", "foo bar"]
        request_json = json.dumps({"model": test_model, "prompt": prompts})
        grpc_resp = grpc_stub.RenderCompletion(
            tokenizer_pb2.RenderCompletionRequest(
                request_json=request_json,
                model_name=test_model,
            )
        )
        assert len(grpc_resp.items) == len(prompts)
        for item in grpc_resp.items:
            assert item.request_id
        direct = asyncio.run(
            RendererService().render_completion(request_json, test_model)
        )
        for grpc_item, direct_item in zip(grpc_resp.items, direct):
            assert list(grpc_item.token_ids) == list(direct_item.token_ids)
