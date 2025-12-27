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

_tokenizer_cache = {}

def clear_caches():
    """Clear the tokenizer cache for testing purposes."""
    _tokenizer_cache.clear()
    return "Tokenizer caches cleared"

def apply_chat_template(request_json):
    """
    Render a chat template using the vllm library.
    This function is aligned with the Go cgo_functions.go structs.

    Args:
        request_json (str): JSON string containing the request parameters:
            - load_tokenizer_with_cache_request (dict): Parameters for loading the tokenizer:
                - is_local (bool, optional): Whether the model is local.
                - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
                - revision (str, optional): Model revision.
                - token (str, optional): Hugging Face token for private models.
                - download_dir (str, optional): Directory to download the model.
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
        tokenizer_request = request.pop("load_tokenizer_with_cache_request")
        tokenizer = load_tokenizer_with_cache(json.dumps(tokenizer_request))

        # Get template_vars and spread them as individual arguments
        template_vars = request.pop('chat_template_kwargs', {})
        request.update(template_vars)

        request["tokenize"] = False
        return tokenizer.apply_chat_template(**request)[0]

    except Exception as e:
        raise RuntimeError(f"Error applying chat template: {e}") from e

def load_tokenizer_with_cache(request_json):
    """
    Initialize and cache the tokenizer based on the request.
    Args:
        request_json (str): JSON string containing the request parameters:
            - is_local (bool, optional): Whether the model is local.
            - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
            - revision (str, optional): Model revision.
            - token (str, optional): Hugging Face token for private models.
            - download_dir (str, optional): Directory to download the model.
    Returns:
        tokenizer: The initialized tokenizer object.
    """
    # Parse the JSON request
    request = json.loads(request_json)

    try:
        model_name = request.pop("model")
        revision = request.get("revision", None)
        is_local = request.pop("is_local", False)
        token = request.pop("token", "")
        download_dir = request.pop("download_dir", None)

        if is_local and os.path.isfile(model_name):
            # If it's a file path (tokenizer.json), get the directory
            model_name = os.path.dirname(model_name)

        cache_key = f"{model_name}:{revision or 'main'}:{is_local}"
        tokenizer = _tokenizer_cache.get(cache_key)
        if tokenizer is not None:
            return tokenizer
        os.environ["HF_TOKEN"] = token
        tokenizer = get_tokenizer(model_name, trust_remote_code=True, revision=revision, download_dir=download_dir)
        _tokenizer_cache[cache_key] = tokenizer
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Error initializing tokenizer: {e}") from e

def example_usage():
    """Example usage of apply_chat_template function."""
    request_str = json.dumps({
        "load_tokenizer_with_cache_request": {
            "is_local": False,
            "model": "ibm-granite/granite-3.3-8b-instruct",
        },
        "conversation": [ [{"role": "system", "content": "You are a helpful assistant."}] , [{"role": "user", "content": "who are you?"}] ],
    })
    print(apply_chat_template(request_str))

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
            "load_tokenizer_with_cache_request": {
                "is_local": True,
                "model": "facebook/opt-125m",
            },
            "conversation": [conversation],
            "chat_template": chat_template
        })
        response = apply_chat_template(request_str)

        print("Rendered chat:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()