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

"""Unit tests for tokenizer service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from tokenizer_service.tokenizer import TokenizerService, TokenizerConfig
from tokenizer_service.exceptions import ModelDownloadError, TokenizationError
from transformers.tokenization_utils_base import BatchEncoding


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.bos_token = ""
    tokenizer.encode_plus.return_value = BatchEncoding({
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "offset_mapping": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    })
    return tokenizer


@patch('tokenizer_service.tokenizer.AutoTokenizer')
def test_tokenizer_service_initialization(mock_auto_tokenizer, mock_tokenizer):
    """Test TokenizerService initialization."""
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    with patch.object(TokenizerService, '_is_remote_model', return_value=False):
        config = TokenizerConfig(model="test/model")
        service = TokenizerService(config)
        
        # Verify that from_pretrained was called
        mock_auto_tokenizer.from_pretrained.assert_called()
        assert service.tokenizer == mock_tokenizer
        assert service.config == config


def test_apply_template(mock_tokenizer):
    """Test apply_template method."""
    with patch('tokenizer_service.tokenizer.AutoTokenizer'):
        with patch.object(TokenizerService, '_is_remote_model', return_value=False):
            config = TokenizerConfig(model="test/model")
            service = TokenizerService(config)
            service.tokenizer = mock_tokenizer
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Mock the apply_chat_template method
    mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"
    
    result = service.apply_template(messages)
    
    assert result == "Formatted prompt"
    mock_tokenizer.apply_chat_template.assert_called_once_with(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def test_apply_template_exception(mock_tokenizer):
    """Test apply_template method with exception."""
    with patch('tokenizer_service.tokenizer.AutoTokenizer'):
        with patch.object(TokenizerService, '_is_remote_model', return_value=False):
            config = TokenizerConfig(model="test/model")
            service = TokenizerService(config)
            service.tokenizer = mock_tokenizer
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Mock the apply_chat_template method to raise an exception
    mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")
    
    with pytest.raises(TokenizationError):
        service.apply_template(messages)


def test_tokenize_and_process(mock_tokenizer):
    """Test tokenize_and_process method."""
    with patch('tokenizer_service.tokenizer.AutoTokenizer'):
        with patch.object(TokenizerService, '_is_remote_model', return_value=False):
            config = TokenizerConfig(model="test/model")
            service = TokenizerService(config)
            service.tokenizer = mock_tokenizer
    
    prompt = "Hello, world!"
    
    result = service.tokenize_and_process(prompt, True)
    
    assert isinstance(result, BatchEncoding)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "offset_mapping" in result
    
    mock_tokenizer.encode_plus.assert_called_once_with(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True
    )


def test_tokenize_and_process_with_bos_token(mock_tokenizer):
    """Test tokenize_and_process method when prompt starts with BOS token."""
    with patch('tokenizer_service.tokenizer.AutoTokenizer'):
        with patch.object(TokenizerService, '_is_remote_model', return_value=False):
            config = TokenizerConfig(model="test/model")
            service = TokenizerService(config)
            service.tokenizer = mock_tokenizer
    
    prompt = "Hello, world!"  # Starts with BOS token
    # Set the bos_token to match beginning of prompt
    mock_tokenizer.bos_token = "H"
    
    result = service.tokenize_and_process(prompt, False)
    
    # Should call encode_plus with add_special_tokens=False to avoid duplication
    mock_tokenizer.encode_plus.assert_called_once_with(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True
    )


def test_tokenize_and_process_exception(mock_tokenizer):
    """Test tokenize_and_process method with exception."""
    with patch('tokenizer_service.tokenizer.AutoTokenizer'):
        with patch.object(TokenizerService, '_is_remote_model', return_value=False):
            config = TokenizerConfig(model="test/model")
            service = TokenizerService(config)
            service.tokenizer = mock_tokenizer
    
    prompt = "Hello, world!"
    
    # Mock encode_plus to raise an exception
    mock_tokenizer.encode_plus.side_effect = Exception("Tokenization error")
    
    with pytest.raises(TokenizationError):
        service.tokenize_and_process(prompt, True)


@patch('tokenizer_service.tokenizer.os.path.exists')
@patch('tokenizer_service.tokenizer.AutoTokenizer')
def test_create_tokenizer_local_path(mock_auto_tokenizer, mock_exists, mock_tokenizer):
    """Test _create_tokenizer with local path."""
    mock_exists.return_value = True
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    with patch.object(TokenizerService, '_is_remote_model', return_value=False):
        config = TokenizerConfig(model="./local/model")
        service = TokenizerService(config)
        
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "./local/model",
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,
        )
        assert service.tokenizer == mock_tokenizer


@patch('tokenizer_service.tokenizer.os.getenv')
@patch('tokenizer_service.tokenizer.os.path.exists')
@patch('tokenizer_service.tokenizer.hf_snapshot_download')
@patch('tokenizer_service.tokenizer.AutoTokenizer')
def test_create_tokenizer_remote_hf(mock_auto_tokenizer, mock_hf_download, mock_exists, mock_getenv, mock_tokenizer):
    """Test _create_tokenizer with remote model from Hugging Face."""
    # Setup mocks
    mock_getenv.return_value = "false"  # USE_MODELSCOPE=false
    mock_exists.return_value = False  # Model not cached locally
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    with patch.object(TokenizerService, '_is_remote_model', return_value=True):
        config = TokenizerConfig(model="test/remote-model")
        service = TokenizerService(config)
        
        # Verify the calls
        mock_hf_download.assert_called_once()
        mock_auto_tokenizer.from_pretrained.assert_called()
        assert service.tokenizer == mock_tokenizer


@patch('tokenizer_service.tokenizer.os.getenv')
@patch('tokenizer_service.tokenizer.os.path.exists')
@patch('tokenizer_service.tokenizer.hf_snapshot_download')
def test_create_tokenizer_remote_hf_download_error(mock_hf_download, mock_exists, mock_getenv):
    """Test _create_tokenizer with remote model download error."""
    # Setup mocks
    mock_getenv.return_value = "false"  # USE_MODELSCOPE=false
    mock_exists.return_value = False  # Model not cached locally
    mock_hf_download.side_effect = Exception("Download error")
    
    with patch.object(TokenizerService, '_is_remote_model', return_value=True):
        config = TokenizerConfig(model="test/remote-model")
        
        with pytest.raises(ModelDownloadError):
            TokenizerService(config)


@patch('tokenizer_service.tokenizer.os.getenv')
@patch('tokenizer_service.tokenizer.os.path.exists')
@patch('tokenizer_service.tokenizer.snapshot_download')
def test_create_tokenizer_modelscope_download_error(mock_snapshot_download, mock_exists, mock_getenv):
    """Test _create_tokenizer with ModelScope download error."""
    # Setup mocks
    mock_getenv.return_value = "true"  # USE_MODELSCOPE=true
    mock_exists.return_value = False  # Model not cached locally
    mock_snapshot_download.side_effect = Exception("Download error")
    
    with patch.object(TokenizerService, '_is_remote_model', return_value=True):
        config = TokenizerConfig(model="test/remote-model")
        
        with pytest.raises(ModelDownloadError):
            TokenizerService(config)