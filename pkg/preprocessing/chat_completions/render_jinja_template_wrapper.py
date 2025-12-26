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
Standalone wrapper for render_jinja_template function from transformers.
"""

import json
import logging
import sys
from typing import Optional, Union

# Import core functions from transformers - moved to function level to avoid import errors
TRANSFORMERS_AVAILABLE = None  # Will be set when first needed

def _ensure_transformers_available():
    """Ensure transformers is available, importing it if needed."""
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE is None:
        try:
            print("[Python] Attempting to import transformers...")
            from transformers.utils.chat_template_utils import render_jinja_template as transformers_render_jinja_template, get_json_schema
            from transformers import AutoTokenizer
            print("[Python] Successfully imported transformers!")
            TRANSFORMERS_AVAILABLE = True
            return True
        except ImportError as e:
            print(f"[Python] Failed to import transformers: {e}")
            print("[Python] Ensure the 'transformers' library is installed in the Python environment.")
            TRANSFORMERS_AVAILABLE = False
            return False
    return TRANSFORMERS_AVAILABLE

# Basic logging setup
logger = logging.getLogger(__name__)

# Module-level cache for templates
_template_cache = {}
_cache_lock = None

def _get_cache_lock():
    """Get or create a threading lock for cache access."""
    global _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    return _cache_lock


def _collect_template_vars(tokenizer):
    """Collect extra rendering variables from a tokenizer."""
    kwargs = {}
    for k in ["bos_token", "eos_token", "eot_token", "pad_token", "unk_token", "sep_token", "additional_special_tokens"]:
        v = getattr(tokenizer, k, None)
        if v is not None:
            kwargs[k] = v
    return kwargs


def clear_caches():
    """Clear all caches for testing purposes."""
    lock = _get_cache_lock()
    with lock:
        global _template_cache
        _template_cache.clear()
    return "Caches cleared"


def render_jinja_template(request_json):
    """
    Render a chat template using the transformers library.
    This function is aligned with the Go cgo_functions.go structs.

    Args:
        request_json (str): JSON string containing the request parameters:
            - messages (list): List of conversation lists
            - chat_template (str, optional): The template to use
            - tools (list, optional): Tool schemas
            - documents (list, optional): Document schemas
            - return_assistant_tokens_mask (bool, optional): Whether to return assistant tokens mask
            - continue_final_message (bool, optional): Whether to continue final message
            - add_generation_prompt (bool, optional): Whether to add generation prompt
            - kwargs (dict, optional): Additional rendering variables
    Returns:
        str: JSON string containing 'rendered_chats' and 'generation_indices' keys.
    """
    if not _ensure_transformers_available():
        raise ImportError("transformers library is required for render_jinja_template")

    # Import the modules we need
    from transformers.utils.chat_template_utils import render_jinja_template as transformers_render_jinja_template

    # Parse the JSON request
    request = json.loads(request_json)

    # Align Go's `messages` field with transformers' `conversations` parameter.
    if 'messages' in request:
        request['conversations'] = [request.pop('messages')] # wrap to match expected format

    try:
        # Get template_vars and spread them as individual arguments
        template_vars = request.pop('chat_template_kwargs', {})
        request.update(template_vars)

        rendered_chats, generation_indices = transformers_render_jinja_template(**request)

    except Exception as e:
        raise

    # Return as JSON string, aligning with the Go response struct.
    result = json.dumps({
        "rendered_chats": rendered_chats,
        "generation_indices": generation_indices
    })
    return result


def get_model_chat_template(request_json):
    """
    Load a tokenizer from Hugging Face Hub or local path and return its chat template string and required variables.
    Args:
        request_json (str): JSON string containing the request parameters:
            - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
            - chat_template (str, optional): The template name or string to use.
            - tools (list[dict], optional): Tool schemas to pass.
            - revision (str, optional): Model revision.
            - token (str, optional): Hugging Face token for private models.
            - is_local_path (bool, optional): Whether the model is a local path (default: False).
    Returns:
        str: JSON string containing 'template' and 'kwargs' keys, aligning with the Go response struct.
    """
    if not _ensure_transformers_available():
        print("[Python] get_model_chat_template ERROR - Transformers not available")
        raise ImportError("transformers library is required for get_model_chat_template")

    # Parse the JSON request
    request = json.loads(request_json)

    model_name = request.get("model")
    chat_template = request.get("chat_template")
    tools = request.get("tools")
    revision = request.get("revision")
    token = request.get("token")
    is_local_path = request.get("is_local_path", False)

    if not model_name:
        print("[Python] get_model_chat_template ERROR - model_name is required")
        raise ValueError("model_name is required in request")

    # Create cache key
    cache_key = f"{model_name}:{revision or 'main'}:{token or 'none'}:{is_local_path}"

    # Check cache first
    lock = _get_cache_lock()
    with lock:
        if cache_key in _template_cache:
            cached_result = _template_cache[cache_key]
            # If a specific chat_template was requested, override the cached template
            if chat_template is not None:
                cached_result["template"] = chat_template
            return json.dumps(cached_result)

    # Import the modules we need
    from transformers import AutoTokenizer
    import os

    # Determine if we're loading from local path or HuggingFace
    if is_local_path:
        # For local paths, model_name can be either a directory containing tokenizer files
        # or a path to a specific tokenizer file. Ensure we extract the directory if needed.
        if os.path.isfile(model_name):
            # If it's a file path (tokenizer.json), get the directory
            tokenizer_dir = os.path.dirname(model_name)
        else:
            # If it's already a directory, use it directly
            tokenizer_dir = model_name

        print(f"[Python] Loading tokenizer from local path: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=True)
    else:
        # Load from Hugging Face
        print(f"[Python] Loading tokenizer from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token, trust_remote_code=True)

    template = tokenizer.chat_template if chat_template is None else chat_template

    # Collect special tokens
    template_vars = _collect_template_vars(tokenizer)

    # Cache the result, aligning with the Go response struct.
    result = {"chat_template": template, "chat_template_kwargs": template_vars}
    with lock:
        _template_cache[cache_key] = result.copy()  # Cache a copy to avoid reference issues

    return json.dumps(result)


def main():
    """Example usage and testing function."""
    if not _ensure_transformers_available():
        print("Error: transformers library is required but not available")
        print("Please install transformers: pip install transformers")
        return

    if len(sys.argv) < 2:
        print("Usage: python render_jinja_template_wrapper.py <chat_template> [conversation_json]")
        print("Example:")
        print('python render_jinja_template_wrapper.py "{% for message in messages %}{{ message.role }}: {{ message.content }}\\n{% endfor %}"')
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
            "conversations": [conversation],
            "chat_template": chat_template
        })
        response_str = render_jinja_template(request_str)
        response_data = json.loads(response_str)

        rendered = response_data['rendered_chats']
        generation_indices = response_data['generation_indices']

        print("Rendered chat:")
        print(rendered[0])
        if generation_indices and len(generation_indices) > 0 and generation_indices[0]:
            print(f"Generation indices: {generation_indices[0]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()