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


def get_or_create_tokenizer_key(request_json):
    """
    Return the cache key for the tokenizer specified in the request.
    If the tokenizer is not already cached, initialize and cache it first.

    Args:
        request_json (str): JSON string containing the request parameters:
            - is_local (bool, optional): Whether the model is local.
            - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
            - revision (str, optional): Model revision.
            - token (str, optional): Hugging Face token for private models.
            - download_dir (str, optional): Directory to download the model.
    Returns:
        str: The cache key for the initialized tokenizer.

    Note:
        Setting is_local=True does NOT prevent downloading if the model path is not a file or directory.
        For example, if is_local=True but model is a HuggingFace model ID, it will still be downloaded.
        Conversely, if is_local=False but model is a file or directory path, the model will NOT be downloaded and will be loaded locally.
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

        key = f"{model_name}:{revision or 'main'}:{is_local}"
        tokenizer = _tokenizer_cache.get(key)
        if tokenizer is not None:
            return key
        os.environ["HF_TOKEN"] = token
        tokenizer = get_tokenizer(
            model_name,
            trust_remote_code=True,
            revision=revision,
            download_dir=download_dir,
        )
        _tokenizer_cache[key] = tokenizer
        return key
    except Exception as e:
        raise RuntimeError(f"Error initializing tokenizer: {e}") from e


def apply_chat_template(request_json):
    """
    Render a chat template using the vllm library.
    This function is aligned with the Go cgo_functions.go structs.

    Args:
        request_json (str): JSON string containing the request parameters:
            - key (str): The tokenizer cache key
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
        key = request.pop("key")
        tokenizer = _tokenizer_cache.get(key)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer with key {key} not found in cache")

        # Get template_vars and spread them as individual arguments
        template_vars = request.pop("chat_template_kwargs", {})
        request.update(template_vars)

        request["tokenize"] = False
        return tokenizer.apply_chat_template(**request)[0]

    except Exception as e:
        raise RuntimeError(f"Error applying chat template: {e}") from e


def encode(request_json: str) -> str:
    """
    Encode text using the specified tokenizer.

    Args:
        request_json (str): JSON string containing:
            - key (str): The tokenizer cache key
            - text (str): The text to encode
            - add_special_tokens (bool, optional): Whether to add special tokens

    Returns:
        JSON string containing:
            - input_ids (list of int): The list of token IDs.
            - offset_mapping (list of [int, int]): The list of offset mappings for each token.
    """
    try:
        request = json.loads(request_json)
        key = request["key"]
        text = request["text"]
        add_special_tokens = request.get("add_special_tokens", False)

        tokenizer = _tokenizer_cache.get(key)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer with key {key} not found in cache")

        return json.dumps(
            tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=add_special_tokens
            ).data
        )

    except Exception as e:
        raise RuntimeError(f"Error encoding texts: {e}") from e


# python pkg/preprocessing/chat_completions/tokenizer_wrapper.py True '{"model": "/mnt/models/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", "conversation": [[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "who are you?"}]]}'
def main():
    """Example usage and testing function."""
    is_local = False
    if len(sys.argv) > 1:
        is_local = sys.argv[1].lower() == "true"

    # Default body if none provided
    body = {
        "model": "facebook/opt-125m",
        "conversation": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "who are you?"},
            ]
        ],
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    }
    if len(sys.argv) > 2:
        body = json.loads(sys.argv[2])

    try:
        # Construct the request JSON string similar to how Go would
        key = get_or_create_tokenizer_key(
            json.dumps(
                {
                    "is_local": is_local,
                    "model": body.get("model"),
                }
            )
        )
        body["key"] = key
        request_str = json.dumps(body)
        templated_str = apply_chat_template(request_str)
        print(f"Templated string:\n{templated_str}")
        encoded_str = encode(
            json.dumps(
                {
                    "key": key,
                    "text": templated_str,
                    "add_special_tokens": False,
                }
            )
        )
        print(f"Encoded string:\n{encoded_str}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
