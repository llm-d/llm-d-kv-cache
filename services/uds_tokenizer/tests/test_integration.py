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
from tokenizer_service.tokenizer import TokenizerService


# ---------------------------------------------------------------------------
# InitializeTokenizer
# ---------------------------------------------------------------------------


class TestInitializeTokenizer:
    """Tests for the InitializeTokenizer RPC."""

    def test_initialize_valid_model(self, grpc_stub, test_model):
        """InitializeTokenizer succeeds for a valid model."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        assert resp.success
        assert not resp.error_message

    def test_initialize_nonexistent_model(self, grpc_stub):
        """InitializeTokenizer returns an error for a non-existent model."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(
                model_name="non-existent/model-that-does-not-exist-12345"
            )
        )
        assert not resp.success
        assert resp.error_message

    def test_initialize_empty_model_name(self, grpc_stub):
        """InitializeTokenizer handles an empty model name."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name="")
        )
        assert not resp.success

    def test_initialize_with_enable_thinking(self, grpc_stub, test_model):
        """InitializeTokenizer respects the enable_thinking flag."""
        resp = grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(
                model_name=test_model,
                enable_thinking=True,
                add_generation_prompt=True,
            )
        )
        assert resp.success


# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the Tokenize RPC."""

    def test_tokenize_simple_text(self, grpc_stub, test_model):
        """Tokenize returns token IDs for simple text."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        resp = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input="Hello, how are you?",
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 0

    def test_tokenize_returns_offset_pairs(self, grpc_stub, test_model, tokenizer_service: TokenizerService):
        """Tokenize returns offset_pairs alongside token IDs."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        resp = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input="Hello world",
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        # offset_pairs is a flat list of [start, end, start, end, ...]
        assert len(resp.offset_pairs) == 2 * len(resp.input_ids)
        
        # Verify token count matches tokenizer
        tokenizer, _ = tokenizer_service.get_tokenizer_for_model(test_model)
        expected_tokens = tokenizer.encode("Hello world", add_special_tokens=True)
        assert list(resp.input_ids) == expected_tokens

    def test_tokenize_without_special_tokens(self, grpc_stub, tokenizer_service: TokenizerService):
        """Tokenize with add_special_tokens=False omits special tokens."""

        model_name = "google-bert/bert-base-uncased"

        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=model_name)
        )
        with_special = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input="test",
                model_name=model_name,
                add_special_tokens=True,
            )
        )
        without_special = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input="test",
                model_name=model_name,
                add_special_tokens=False,
            )
        )
        assert with_special.success and without_special.success
        # With special tokens should produce > tokens as without.
        assert len(with_special.input_ids) > len(without_special.input_ids)
        
        # Verify special tokens using actual tokenizer
        tokenizer, _ = tokenizer_service.get_tokenizer_for_model(model_name)

        # BERT adds [CLS] at start and [SEP] at end
        assert with_special.input_ids[0] == tokenizer.cls_token_id
        assert with_special.input_ids[-1] == tokenizer.sep_token_id

        # Without special tokens should not have [CLS] or [SEP]
        assert without_special.input_ids[0] != tokenizer.cls_token_id
        assert without_special.input_ids[-1] != tokenizer.sep_token_id

    def test_tokenize_empty_input(self, grpc_stub, test_model):
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        resp = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input="",
                model_name=test_model,
                add_special_tokens=False,
            )
        )
        # An empty input should still succeed (may return 0 or only special tokens).
        assert resp.success

    def test_tokenize_long_input(self, grpc_stub, test_model):
        """Tokenize handles a long input string."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        long_text = "Hello world. " * 100_000
        resp = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input=long_text,
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 100  # Should have many tokens.

    def test_tokenize_special_characters(self, grpc_stub, test_model, tokenizer_service: TokenizerService):
        """Tokenize handles special / unicode characters."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        test_input = "Hello 你好 مرحبا 🌍 <|special|>"
        resp = grpc_stub.Tokenize(
            tokenizer_pb2.TokenizeRequest(
                input=test_input,
                model_name=test_model,
                add_special_tokens=True,
            )
        )
        assert resp.success
        assert len(resp.input_ids) > 0
        
        # Verify tokenization matches actual tokenizer
        tokenizer, _ = tokenizer_service.get_tokenizer_for_model(test_model)

        expected_tokens = tokenizer.encode(test_input, add_special_tokens=True)
        assert list(resp.input_ids) == expected_tokens

    def test_tokenize_uninitialized_model(self, grpc_stub):
        """Tokenize for a model that was never initialized returns an error."""
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.Tokenize(
                tokenizer_pb2.TokenizeRequest(
                    input="Hello",
                    model_name="meta-llama/Meta-Llama-3-8B",  # Assuming this model is not initialized in this test
                    add_special_tokens=True,
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_tokenize_deterministic(self, grpc_stub, test_model):
        """Tokenizing the same input twice produces identical results."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        req = tokenizer_pb2.TokenizeRequest(
            input="Determinism check.",
            model_name=test_model,
            add_special_tokens=True,
        )
        resp1 = grpc_stub.Tokenize(req)
        resp2 = grpc_stub.Tokenize(req)
        assert list(resp1.input_ids) == list(resp2.input_ids)
        assert list(resp1.offset_pairs) == list(resp2.offset_pairs)


# ---------------------------------------------------------------------------
# RenderChatTemplate
# ---------------------------------------------------------------------------


class TestRenderChatTemplate:
    """Tests for the RenderChatTemplate RPC.

    NOTE: Not all models ship with a chat template (e.g. openai-community/gpt2
    does not). Tests that require a chat template are expected to fail
    gracefully when the model lacks one.
    """

    def _make_request(self, model_name, messages, add_generation_prompt=True):
        """Helper: build a ChatTemplateRequest."""
        turns = [
            tokenizer_pb2.ConversationTurn(
                messages=[
                    tokenizer_pb2.ChatMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]
            )
        ]
        return tokenizer_pb2.ChatTemplateRequest(
            conversation_turns=turns,
            model_name=model_name,
            add_generation_prompt=add_generation_prompt,
        )

    def test_render_multi_turn(self, grpc_stub, test_model):
        """RenderChatTemplate handles a multi-turn conversation."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        resp = grpc_stub.RenderChatTemplate(
            self._make_request(test_model, messages)
        )

        assert resp.success

        for msg in messages:
            assert msg["role"] in resp.rendered_prompt
            assert msg["content"] in resp.rendered_prompt

    def test_render_empty_messages(self, grpc_stub, test_model):
        """RenderChatTemplate with empty messages."""
        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=test_model)
        )

        # Empty messages should raise an error
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderChatTemplate(
                self._make_request(test_model, [])
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    def test_render_uninitialized_model(self, grpc_stub):
        """RenderChatTemplate for an uninitialized model returns an error."""
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderChatTemplate(
                self._make_request("openai-community/gpt2", messages)
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL
    
    def test_render_for_model_without_template(self, grpc_stub):
        """RenderChatTemplate for a model without a chat template returns an error."""

        model_name = "openai-community/gpt2"  # This model is known to lack a chat template.

        grpc_stub.InitializeTokenizer(
            tokenizer_pb2.InitializeTokenizerRequest(model_name=model_name)
        )
        messages = [{"role": "user", "content": "Hi"}]

        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.RenderChatTemplate(
                self._make_request(model_name, messages)
            )
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL
        assert "chat template" in str(exc_info.value.details()).lower()
