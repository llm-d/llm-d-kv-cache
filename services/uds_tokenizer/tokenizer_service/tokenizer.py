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

"""Tokenizer service for handling LLM tokenization operations."""

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)
from transformers.tokenization_utils_base import BatchEncoding
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from .exceptions import TokenizerError, ModelDownloadError, TokenizationError

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer processing"""
    model: str
    add_special_tokens: bool = True
    enable_thinking: bool = False
    add_generation_prompt: bool = True


class TokenizerService:
    """Service for handling tokenizer operations"""

    def __init__(self, config: TokenizerConfig):
        """Initialize service with configuration"""
        self.tokenizer = self._create_tokenizer(config.model)
        self.config = config
    
    def _create_tokenizer(self, model_identifier: str) -> AnyTokenizer:
        """Create a tokenizer, using cached files if available or downloading from ModelScope or Hugging Face"""
        # Check if the model_identifier is a remote model name or a local path
        # More robust check similar to what vLLM does
        is_remote_model = self._is_remote_model(model_identifier)
        
        # For local paths, use directly
        if not is_remote_model:
            logging.info(f"Loading tokenizer from {model_identifier}")
            base_tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
            )
            return base_tokenizer
            
        # Determine download source: ModelScope (if USE_MODELSCOPE=true) or Hugging Face (default)
        use_modelscope = os.getenv('USE_MODELSCOPE', 'false').lower() == 'true'
        
        # Path to the models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        # Convert model identifier to local path (e.g., qwen/Qwen2-7B -> models/qwen/Qwen2-7B)
        org_name, model_name = model_identifier.split('/', 1)
        local_model_path = os.path.join(models_dir, org_name, model_name)
        
        # If the model is already cached, use the cached version
        # Check that required files exist before trying to load
        required_files = [
            "config.json",
            "tokenizer.json",
        ]
        if (os.path.exists(local_model_path) and 
            all(os.path.exists(os.path.join(local_model_path, f)) for f in required_files)):
            logging.info(f"Using cached tokenizer from {local_model_path}")
            base_tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
            )
            return base_tokenizer
            
        # Download the tokenizer files from ModelScope or Hugging Face
        if use_modelscope:
            return self._download_from_modelscope(model_identifier, local_model_path)
        else:
            return self._download_from_huggingface(model_identifier, local_model_path)
    
    def _download_from_modelscope(self, model_identifier: str, local_model_path: str) -> AnyTokenizer:
        """Download tokenizer files from ModelScope"""
        logging.info(f"Downloading tokenizer for {model_identifier} from ModelScope")
        try:
            # Ensure the local model directory exists
            os.makedirs(local_model_path, exist_ok=True)
            
            # Download only the tokenizer related files from ModelScope
            snapshot_download(
                model_identifier,
                local_dir=local_model_path,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "config.json",
                    "generation_config.json"
                ]
            )
            logging.info(f"Successfully downloaded tokenizer to {local_model_path}")
        except Exception as e:
            # Clean up potentially incomplete download directory
            if os.path.exists(local_model_path) and not os.listdir(local_model_path):
                os.rmdir(local_model_path)
                logging.info(f"Removed empty directory {local_model_path}")
            logging.error(f"Failed to download tokenizer for {model_identifier} from ModelScope: {e}")
            raise ModelDownloadError(f"Failed to download model from ModelScope: {e}") from e
            
        # Load the tokenizer from the downloaded files
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
            )
            return base_tokenizer
        except Exception as e:
            logging.error(f"Failed to load tokenizer from {local_model_path}: {e}")
            raise TokenizerError(f"Failed to load tokenizer: {e}") from e
    
    def _download_from_huggingface(self, model_identifier: str, local_model_path: str) -> AnyTokenizer:
        """Download tokenizer files from Hugging Face"""
        logging.info(f"Downloading tokenizer for {model_identifier} from Hugging Face")
        try:
            # Ensure the local model directory exists
            os.makedirs(local_model_path, exist_ok=True)
            
            # Download only the tokenizer related files from Hugging Face
            hf_snapshot_download(
                model_identifier,
                local_dir=local_model_path,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "config.json",
                    "generation_config.json"
                ]
            )
            logging.info(f"Successfully downloaded tokenizer to {local_model_path}")
        except Exception as e:
            # Clean up potentially incomplete download directory
            if os.path.exists(local_model_path) and not os.listdir(local_model_path):
                os.rmdir(local_model_path)
                logging.info(f"Removed empty directory {local_model_path}")
            logging.error(f"Failed to download tokenizer for {model_identifier} from Hugging Face: {e}")
            raise ModelDownloadError(f"Failed to download model from Hugging Face: {e}") from e
            
        # Load the tokenizer from the downloaded files
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
            )
            return base_tokenizer
        except Exception as e:
            logging.error(f"Failed to load tokenizer from {local_model_path}: {e}")
            raise TokenizerError(f"Failed to load tokenizer: {e}") from e
    
    def _is_remote_model(self, model_identifier: str) -> bool:
        """Check if the model identifier is a remote model name or a local path."""
        # Check if it's an absolute path
        if os.path.isabs(model_identifier):
            return False
            
        # Check if it's a relative path (starts with ./ or ../)
        if model_identifier.startswith("./") or model_identifier.startswith("../"):
            return False
            
        # Check if it's a local directory that exists
        if os.path.exists(model_identifier):
            return False
            
        # Check for protocol prefixes (s3://, etc.)
        if "://" in model_identifier.split("/")[0]:
            return False
            
        # If none of the above, it's likely a remote model identifier
        # containing organization/model format
        return "/" in model_identifier
    
    def apply_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages"""
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=self.config.add_generation_prompt,
                enable_thinking=self.config.enable_thinking,
            )

            logging.debug(f"Prompt: {prompt}")
            return prompt
        except Exception as e:
            logging.error(f"Failed to apply chat template: {e}")
            raise TokenizationError(f"Failed to apply chat template: {e}") from e
    
    def tokenize_and_process(self, prompt: str) -> BatchEncoding:
        """
        Tokenize the prompt with proper handling of special tokens.

        When using chat templates, we need to be careful not to add BOS tokens twice:
        1. Once from the chat template itself (which may include BOS token)
        2. Once from the add_special_tokens parameter

        vLLM handles this by setting add_special_tokens=False when using chat templates.
        """
        try:
            # Check if the prompt already contains BOS token at the beginning
            # This can happen when using chat templates that explicitly include BOS token
            add_special_tokens = self.config.add_special_tokens

            # If add_special_tokens is None, use the tokenizer's default behavior
            if add_special_tokens is None:
                # For tokenizers with add_bos_token attribute, we still need to check
                # if the prompt already contains BOS token to avoid duplication
                if (hasattr(self.tokenizer, 'bos_token') and
                    self.tokenizer.bos_token and
                    prompt.startswith(self.tokenizer.bos_token)):
                    # If prompt already has BOS token, explicitly set add_special_tokens=False
                    # to avoid adding it twice
                    add_special_tokens = False
                else:
                    # Otherwise, let the tokenizer use its default behavior
                    add_special_tokens = True  # Default behavior for most tokenizers

            # If the prompt already starts with BOS token, set add_special_tokens to False
            # to avoid adding it twice
            elif (hasattr(self.tokenizer, 'bos_token') and
                  self.tokenizer.bos_token and
                  prompt.startswith(self.tokenizer.bos_token)):
                add_special_tokens = False

            token_id_offsets = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True
            )
            logging.debug(f"Encoded prompt: {token_id_offsets}")
            return token_id_offsets
        except Exception as e:
            logging.error(f"Failed to tokenize prompt: {e}")
            raise TokenizationError(f"Failed to tokenize prompt: {e}") from e