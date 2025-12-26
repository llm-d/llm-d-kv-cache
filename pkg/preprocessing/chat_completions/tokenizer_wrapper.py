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

#!/usr/bin/env python3
"""
Standalone wrapper for tokenizer from vllm.
"""

import json
import logging
import os
import sys
from vllm.transformers_utils.tokenizer import get_tokenizer

# Basic logging setup
logger = logging.getLogger(__name__)

# Module-level cache for templates
_tokenizer_cache = {}
_tokenizer_cache_lock = None

def _get_tokenizer_cache_lock():
    """Get or create a threading lock for tokenizer cache access."""
    global _tokenizer_cache_lock
    if _tokenizer_cache_lock is None:
        import threading
        _tokenizer_cache_lock = threading.RLock()
    return _tokenizer_cache_lock

def clear_caches():
    """Clear the tokenizer cache for testing purposes."""
    lock = _get_tokenizer_cache_lock()
    with lock:
        global _tokenizer_cache
        _tokenizer_cache.clear()
    return "Tokenizer caches cleared"

def apply_chat_template(request_json):
    """
    Render a chat template using the transformers library.
    This function is aligned with the Go cgo_functions.go structs.

    Args:
        request_json (str): JSON string containing the request parameters:
            - is_local (bool, optional): Whether the model is local.
            - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
            - revision (str, optional): Model revision.
            - token (str, optional): Hugging Face token for private models.
            - conversation (list): List of conversation lists
            - chat_template (str, optional): The template to use
            - tools (list, optional): Tool schemas
            - documents (list, optional): Document schemas
            - return_assistant_tokens_mask (bool, optional): Whether to return assistant tokens mask
            - continue_final_message (bool, optional): Whether to continue final message
            - add_generation_prompt (bool, optional): Whether to add generation prompt
            - chat_template_kwargs (dict, optional): Additional rendering variables

    Returns:
        str: The rendered chat template as a string.
    """

    try:
        # Parse the JSON request
        request = json.loads(request_json)

        # Get template_vars and spread them as individual arguments
        template_vars = request.pop('chat_template_kwargs', {})
        request.update(template_vars)

        model_name = request.pop("model")
        revision = request.get("revision", None)
        is_local = request.pop("is_local", False)
        token = request.pop("token", "")
        download_dir = request.pop("download_dir", None)

        if is_local and os.path.isfile(model_name):
            # If it's a file path (tokenizer.json), get the directory
            model_name = os.path.dirname(model_name)

        lock = _get_tokenizer_cache_lock()
        with lock:
            cache_key = f"{model_name}:{revision or 'main'}:{is_local}"
            tokenizer = _tokenizer_cache.get(cache_key)
            if tokenizer is None:
                os.environ["HF_TOKEN"] = token
                tokenizer = get_tokenizer(model_name, trust_remote_code=True, revision=revision, download_dir=download_dir)
                _tokenizer_cache[cache_key] = tokenizer
            
            request["tokenize"] = False
            return tokenizer.apply_chat_template(**request)

    except Exception as e:
        raise RuntimeError(f"Error applying chat template: {e}") from e


def encode(request_json: str) -> str:
    """
    Encode text using the specified tokenizer.

    Args:
        request_json (str): JSON string containing:
            - model (str): The model ID or path.
            - revision (str, optional): Model revision.
            - is_local (bool, optional): Whether the model is local.
            - download_dir (str, optional): Directory to download the model.
            - text (str): The text to encode.
            - token (str, optional): Hugging Face token for private models.

    Returns:
        str: JSON string containing 'encoded_texts' key with list of token ID lists.
    """
    try:
        request = json.loads(request_json)
        model_name = request["model"]
        revision = request.get("revision", None)
        is_local = request.get("is_local", False)
        download_dir = request.pop("download_dir", None)
        text = request["text"]
        token = request.get("token", "")
        add_special_tokens = request.get("add_special_tokens", False)

        if is_local and os.path.isfile(model_name):
            # If it's a file path (tokenizer.json), get the directory
            model_name = os.path.dirname(model_name)

        lock = _get_tokenizer_cache_lock()
        with lock:
            cache_key = f"{model_name}:{revision or 'main'}:{is_local}"
            tokenizer = _tokenizer_cache.get(cache_key)
            if tokenizer is None:
                os.environ["HF_TOKEN"] = token
                tokenizer = get_tokenizer(model_name, trust_remote_code=True, revision=revision, download_dir=download_dir)
                _tokenizer_cache[cache_key] = tokenizer

            return json.dumps(tokenizer(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens).data)

    except Exception as e:
        raise RuntimeError(f"Error encoding texts: {e}") from e

def main():
    """Example usage and testing function."""

    if len(sys.argv) < 2:
        print("Usage: python tokenizer_wrapper.py <chat_template> [conversation_json]")
        print("Example:")
        print('python tokenizer_wrapper.py "{% for message in messages %}{{ message.role }}: {{ message.content }}\\n{% endfor %}"')
        return

    chat_template = sys.argv[1]

    # Default conversation if none provided
    conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"}
    ]

    if len(sys.argv) > 2:
        try:
            conversation = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Error: Invalid JSON for conversation")
            return

    try:
        # Construct the request JSON string similar to how Go would
        request_str = json.dumps({
            "model": "facebook/opt-125m",
            "conversation": [conversation],
            "chat_template": chat_template
        })
        response = apply_chat_template(request_str)

        print("Rendered chat:")
        print(response[0])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()