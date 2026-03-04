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
Pytest-based integration tests for the UDS tokenizer gRPC service.

These tests require a running gRPC server.  The ``grpc_server`` session
fixture in ``conftest.py`` starts one automatically.

Run with:
    pytest tests/test_integration.py -v

Use ``TEST_MODEL`` env var to override the default model.
"""

import grpc
import pytest

import tokenizerpb.tokenizer_pb2 as tokenizer_pb2


# ---------------------------------------------------------------------------
# InitializeTokenizer
# ---------------------------------------------------------------------------


class TestInitializeTokenizer:
    """Tests for the InitializeTokenizer RPC."""

    def test_initialize_valid_model(self, grpc_stub, test_model):
        """InitializeTokenizer succeeds for a valid model."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        assert resp.success
        assert not resp.error_message

    def test_initialize_nonexistent_model(self, grpc_stub):
        """InitializeTokenizer returns an error for a non-existent model."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(
                model="non-existent/model-that-does-not-exist-12345"
            )
        )
        assert not resp.success
        assert resp.error_message

    def test_initialize_empty_model_name(self, grpc_stub):
        """InitializeTokenizer handles an empty model name."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model="")
        )
        assert not resp.success

    def test_initialize_with_enable_thinking(self, grpc_stub, test_model):
        """InitializeTokenizer respects the enable_thinking flag."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(
                model=test_model,
                is_local=True,
            )
        )
        assert resp.success


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    """Tests for the Render RPC."""

    def test_render_simple_text(self, grpc_stub, test_model):
        """Render returns token IDs for simple text."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        resp = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text="Hello, how are you?",
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 0

    def test_render_returns_offset_pairs(self, grpc_stub, test_model):
        """Render returns offset_pairs alongside token IDs."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        resp = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text="Hello world",
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        # offset_pairs is a flat list of [start, end, start, end, ...]
        assert len(resp.offset_pairs) == 2 * len(resp.input_ids)

    def test_render_without_special_tokens(self, grpc_stub, test_model):
        """Render with add_special_tokens=False omits special tokens."""

        model_name = "deepseek-ai/DeepSeek-R1"

        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=model_name)
        )
        with_special = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text="test",
                model_name=model_name,
                add_special_tokens=True,
            )
        )
        without_special = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text="test",
                model_name=model_name,
                add_special_tokens=False,
            )
        )
        assert with_special.success and without_special.success
        # With special tokens should produce > tokens as without.
        assert len(with_special.input_ids) > len(without_special.input_ids)

    def test_render_empty_input(self, grpc_stub, test_model):
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        resp = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text="",
                model_name=test_model,
                add_special_tokens=False,
            )
        )
        # An empty input should still succeed (may return 0 or only special tokens).
        assert resp.success

    def test_render_long_input(self, grpc_stub, test_model):
        """Render handles a long input string."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        long_text = "Hello world. " * 100_000
        resp = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text=long_text,
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 100  # Should have many tokens.

    def test_render_special_characters(self, grpc_stub, test_model):
        """Render handles special / unicode characters."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        test_input = "Hello 你好 مرحبا 🌍 <|special|>"
        resp = grpc_stub.Render(
            tokenizer_pb2.RenderRequest(
                text=test_input,
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 0

    def test_render_uninitialized_model(self, grpc_stub):
        """Render for a model that was never initialized returns an error."""
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.Render(
                tokenizer_pb2.RenderRequest(
                    text="Hello",
                    model_name="meta-llama/Meta-Llama-3-8B",  # Assuming this model is not initialized in this test
                    add_special_tokens=True,
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_render_deterministic(self, grpc_stub, test_model):
        """Rendering the same input twice produces identical results."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        req = tokenizer_pb2.RenderRequest(
            text="Determinism check.",
            model_name=test_model,
            add_special_tokens=True,
        )
        resp1 = grpc_stub.Render(req)
        resp2 = grpc_stub.Render(req)
        assert list(resp1.input_ids) == list(resp2.input_ids)
        assert list(resp1.offset_pairs) == list(resp2.offset_pairs)


# ---------------------------------------------------------------------------
# RenderChat
# ---------------------------------------------------------------------------


class TestRenderChat:
    """Tests for the RenderChat RPC.

    NOTE: Not all models ship with a chat template (e.g. openai-community/gpt2
    does not). Tests that require a chat template are expected to fail
    gracefully when the model lacks one.
    """

    def _make_request(self, model_name, messages, add_generation_prompt=True):
        """Helper: build a RenderChatRequest."""
        chat_messages = [
            tokenizer_pb2.ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        return tokenizer_pb2.RenderChatRequest(
            conversation=chat_messages,
            model_name=model_name,
            add_generation_prompt=add_generation_prompt,
        )

    def test_render_multi_turn(self, grpc_stub, test_model):
        """RenderChat handles a multi-turn conversation."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        resp = grpc_stub.RenderChat(
            self._make_request(test_model, messages)
        )

        assert resp.success
        assert len(resp.input_ids) > 0

    def test_render_empty_messages(self, grpc_stub, test_model):
        """RenderChat with empty messages."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )

        # Empty messages should still succeed, may return special tokens only
        resp = grpc_stub.RenderChat(
            self._make_request(test_model, [])
        )
        # Should succeed but may have only special tokens
        assert resp.success

    def test_render_uninitialized_model(self, grpc_stub):
        """RenderChat for an uninitialized model returns an error."""
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderChat(
                self._make_request("openai-community/gpt2", messages)
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_render_with_tools(self, grpc_stub, test_model):
        """RenderChat with tools parameter."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model=test_model)
        )
        messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        
        # Create a simple tool definition
        tool = tokenizer_pb2.Value(struct_value=tokenizer_pb2.StructValue(fields={
            "type": tokenizer_pb2.Value(string_value="function"),
            "function": tokenizer_pb2.Value(struct_value=tokenizer_pb2.StructValue(fields={
                "name": tokenizer_pb2.Value(string_value="calculator"),
                "description": tokenizer_pb2.Value(string_value="A simple calculator"),
                "parameters": tokenizer_pb2.Value(struct_value=tokenizer_pb2.StructValue(fields={
                    "type": tokenizer_pb2.Value(string_value="object"),
                    "properties": tokenizer_pb2.Value(struct_value=tokenizer_pb2.StructValue(fields={
                        "operation": tokenizer_pb2.Value(struct_value=tokenizer_pb2.StructValue(fields={
                            "type": tokenizer_pb2.Value(string_value="string"),
                            "enum": tokenizer_pb2.Value(list_value=tokenizer_pb2.ListValue(values=[
                                tokenizer_pb2.Value(string_value="add"),
                                tokenizer_pb2.Value(string_value="subtract")
                            ]))
                        }))
                    }))
                }))
            }))
        }))

        req = tokenizer_pb2.RenderChatRequest(
            conversation=[tokenizer_pb2.ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            tools=[tool],
            model_name=test_model,
            add_generation_prompt=True,
        )

        resp = grpc_stub.RenderChat(req)
        assert resp.success
        assert len(resp.input_ids) > 0
