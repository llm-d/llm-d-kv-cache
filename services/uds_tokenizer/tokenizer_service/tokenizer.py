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

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from transformers.tokenization_utils_base import BatchEncoding
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from .exceptions import TokenizerError, ModelDownloadError, TokenizationError

# Add the preprocessing directory to the Python path to import tokenizer_wrapper
# First try to import from the runtime location, then from the development location
import sys
import os

# Add both runtime and development paths to Python path
runtime_path = '/app/preprocessing/chat_completions'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
dev_path = os.path.join(os.path.dirname(os.path.dirname(current_file_dir)), '..', 'pkg', 'preprocessing', 'chat_completions')
dev_path = os.path.normpath(dev_path)  # Normalize the path to resolve '..'

if runtime_path not in sys.path:
    sys.path.insert(0, runtime_path)
if dev_path not in sys.path:
    sys.path.insert(0, dev_path)

from tokenizer_wrapper import get_or_create_tokenizer_key, apply_chat_template_direct, encode_direct

AnyTokenizer = object  # Placeholder for vLLM tokenizer type


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer processing"""
    model: str
    enable_thinking: bool = False
    add_generation_prompt: bool = True


class TokenizerService:
    """Service for handling tokenizer operations"""

    def __init__(self, config: TokenizerConfig = None):
        """Initialize service with optional configuration"""
        self.tokenizer_keys = {}  # Dictionary to store tokenizer keys by model name
        self.configs = {}         # Dictionary to store configurations by model name

        if config:
            self.tokenizer_key = self._create_tokenizer_key(config.model)
            self.config = config
            self.tokenizer_keys[config.model] = self.tokenizer_key
            self.configs[config.model] = config

    def _create_tokenizer_key(self, model_identifier: str) -> str:
        """Create a tokenizer key using the shared tokenizer_wrapper functionality"""
        is_remote_model = self._is_remote_model(model_identifier)

        request = {
            "is_local": not is_remote_model,
            "model": model_identifier,
            "revision": None,
            "token": os.getenv("HF_TOKEN", ""),
            "download_dir": None
        }

        # Create the tokenizer key (which caches the tokenizer)
        key = get_or_create_tokenizer_key(json.dumps(request))
        return key

    def load_tokenizer(self, model_name: str, enable_thinking: bool = False, add_generation_prompt: bool = True) -> bool:
        """Load a tokenizer for a specific model"""
        try:
            config = TokenizerConfig(
                model=model_name,
                enable_thinking=enable_thinking,
                add_generation_prompt=add_generation_prompt
            )

            tokenizer_key = self._create_tokenizer_key(model_name)
            self.tokenizer_keys[model_name] = tokenizer_key
            self.configs[model_name] = config

            logging.info(f"Successfully initialized tokenizer for model: {model_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer for model {model_name}: {e}")
            return False

    def get_tokenizer_key(self, model_name: str):
        """Get the tokenizer key for a specific model"""
        if model_name not in self.tokenizer_keys:
            raise TokenizerError(f"Tokenizer not initialized for model: {model_name}")

        return self.tokenizer_keys[model_name], self.configs[model_name]

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

    def apply_template(self, conversation: List[list[Dict[str, str]]], model_name: str,
                      chat_template: Optional[str] = None,
                      tools: Optional[List[Dict]] = None,
                      documents: Optional[List[Dict]] = None,
                      return_assistant_tokens_mask: bool = False,
                      continue_final_message: bool = False,
                      add_generation_prompt: bool = True,
                      chat_template_kwargs: Optional[Dict] = None) -> str:
        """Apply chat template to messages using the shared tokenizer_wrapper functionality"""
        try:
            tokenizer_key, config = self.get_tokenizer_key(model_name)

            logging.debug(f"chat template is None: {chat_template is None}")

            # Apply the chat template using the direct method (no JSON serialization needed)
            result = apply_chat_template_direct(
                key=tokenizer_key,
                conversation=conversation,
                chat_template=chat_template,
                tools=tools,
                documents=documents,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                continue_final_message=continue_final_message,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs
            )

            logging.debug(f"Prompt: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to apply chat template: {e}")
            raise TokenizationError(f"Failed to apply chat template: {e}") from e

    def tokenize_and_process(self, prompt: str, add_special_tokens: bool, model_name: str) -> BatchEncoding:
        """
        Tokenize the prompt with the specified add_special_tokens value.
        """
        try:
            # Get the tokenizer key for the specified model
            tokenizer_key, config = self.get_tokenizer_key(model_name)

            # Encode the text using the direct method (no JSON serialization needed)
            result_data = encode_direct(
                key=tokenizer_key,
                text=prompt,
                add_special_tokens=add_special_tokens
            )

            input_ids = result_data.get('input_ids', [])
            offset_mapping = result_data.get('offset_mapping', [])

            # Create a BatchEncoding object with the results
            token_id_offsets = BatchEncoding({
                'input_ids': input_ids,
                'offset_mapping': offset_mapping
            })

            logging.debug(f"Encoded prompt: {token_id_offsets}")
            return token_id_offsets
        except Exception as e:
            logging.error(f"Failed to tokenize prompt: {e}")
            raise TokenizationError(f"Failed to tokenize prompt: {e}") from e