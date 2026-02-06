# Chat ChatTemplate Integration for OpenAI-API v1/chat/completions Compatibility

> Note: this does not support multi-modality at this point.

## Why Templating is Needed

When processing OpenAI ChatCompletions requests, vLLM templates the input before tokenization. 
For KV-cache lookups to work correctly, we must replicate this templating process in our indexer.

**Example:**
```json
{
  "messages": [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "Let me calculate that."},
    {"role": "user", "content": "Thanks!"}
  ]
}
```

```jinja2
<!-- Model template (e.g., Llama-2) -->
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}
{% elif message['role'] == 'assistant' %}
{{ message['content'] + '</s>' }}
{% endif %}
{% endfor %}
```

```text
<!-- Flattened prompt the model actually sees -->
<s>[INST] What's 2+2? [/INST]Let me calculate that.</s><s>[INST] Thanks! [/INST]
```

**Without templating**, we'd not be able to recreate the same tokens vLLM will produce, leading to incorrect KV-cache lookups.

## Integration with Existing Pipeline

This package provides a library to be used for templating before using the `kvcache.Indexer` entry point.

### Request Structures

The following request structures are used for tokenization:

**GetOrCreateTokenizerKeyRequest** - Initialize and cache a tokenizer:
- `Model` - Model ID or path (HF model ID, local directory, or tokenizer file path)
- `IsLocal` - (Optional) Whether the model is local
- `Revision` - (Optional) Model revision
- `Token` - (Optional) Hugging Face token for private models
- `DownloadDir` - (Optional) Directory to download the model

**RenderChatRequest** - Render chat template and tokenize:
- `Key` - Tokenizer cache key from `GetOrCreateTokenizerKey`
- `Conversation` - List of messages (role/content pairs)
- `Tools` - (Optional) List of tool schemas
- `Documents` - (Optional) List of document dicts
- `ChatTemplate` - (Optional) Override for the chat template
- `ReturnAssistantTokensMask` - (Optional) Whether to return assistant token indices
- `ContinueFinalMessage` - (Optional) Whether to continue from the final message
- `AddGenerationPrompt` - (Optional) Whether to add a generation prompt
- `ChatTemplateKWArgs` - (Optional) Extra parameters for template rendering

These fields align with the transformers library's [`apply_chat_template`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/tokenization_utils_base.py#L1571) method parameters.

**RenderRequest** - Direct tokenization without chat template:
- `Key` - Tokenizer cache key from `GetOrCreateTokenizerKey`
- `Text` - The text to tokenize
- `AddSpecialTokens` - (Optional) Whether to add special tokens

### ChatTemplate Processing Flow

The templating process (steps 1.1-1.4) handles the conversion from structured request to tokens:

```
1.1. **CGO Binding**: preprocessing.NewChatTemplatingProcessor()
    └── cgo_functions.go:NewChatTemplatingProcessor()
        └── Creates ChatTemplatingProcessor struct

1.2. **Get Tokenizer Key**: wrapper.GetOrCreateTokenizerKey(ctx, req)
    ├── cgo_functions.go:GetOrCreateTokenizerKey(ctx, req)
    │   ├── C.Py_CallGetOrCreateTokenizerKey() - **CGO Binding** to Python
    │   └── **Python Wrapper**: tokenizer_wrapper.py:get_or_create_tokenizer_key()
    │       └── from vllm.tokenizers import get_tokenizer
    │           tokenizer = get_tokenizer(...)
    │           _tokenizer_cache[key] = tokenizer
    └── Returns: Cache key string (e.g., "model:revision:is_local")

1.3. **RenderChat (Template + Tokenization)**: wrapper.RenderChat(ctx, req)
    ├── cgo_functions.go:RenderChat(ctx, req)
    │   ├── C.Py_CallRenderChat() - **CGO Binding** to Python
    │   └── **Python Wrapper**: tokenizer_wrapper.py:render_chat()
    │       └── tokenizer.apply_chat_template(**request)
    │           └── Applies template AND tokenizes in one step
    │           └── return json.dumps(result.data)  # {"input_ids": [...], "offset_mapping": [...]}
    └── Returns: ([]uint32, []Offset, error)

    OR

    **Render (Direct Tokenization)**: wrapper.Render(ctx, req)
    ├── cgo_functions.go:Render(ctx, req)
    │   ├── C.Py_CallRender() - **CGO Binding** to Python
    │   └── **Python Wrapper**: tokenizer_wrapper.py:render()
    │       └── tokenizer(text, return_offsets_mapping=True, add_special_tokens=...)
    │           └── return json.dumps(result.data)  # {"input_ids": [...], "offset_mapping": [...]}
    └── Returns: ([]uint32, []Offset, error)

1.4. **Token Processing** (in tokenization/pool.go:processTask)
    └── pool.tokenizer.RenderChat(task.RenderReq) or pool.tokenizer.Render(task.Prompt)
    └── Continue with existing pipeline: KV Block Keys → Pod Scoring
```
### Optimized Preprocessing Architecture

#### **Performance Optimizations**

##### **Single Python Interpreter**
- **Process-Level Initialization**: Single Python interpreter per process, initialization at EPP startup. Scalable, low overhead and reduces memory footprint
- **Thread-Safe Initialization**: Global locks prevent multiple initializations

##### **Function Caching**
- **Cached Python Functions**: `get_or_create_tokenizer_key`, `render_chat`, and `render` cached globally
- **Module-Level Caching**: Python modules imported once and reused
- **Thread Safety**: GIL management for concurrent access

##### **Template Caching**
- **Model-Specific Templates**: Templates cached per model to avoid repeated fetching
- **Hugging Face Integration**: Efficient template retrieval using AutoTokenizer, matching vLLM's



## Experiment Overview & Results

### Benchmark Configuration:

- **Dataset**: ShareGPT conversations with variable length
- **Model**: 2 pods of Qwen/Qwen2.5-0.5B-Instruct
- **Load Pattern**: Progressive QPS from 3→4→5→6→8→10→12→15→20 QPS
- **Duration**: ~18 minutes total with progressive load increases
- **Input Distribution**: 600-800 tokens per request
- **Output Distribution**: 1-100 tokens per request
- **API Comparison**: Chat Completions vs Completions (head-to-head)
- **Success Rate**: 100% for both APIs across all load levels

### Performance Analysis

![Performance Analysis](TTFT_TPOT_THROUGHPUT_TRIPANEL.png)

#### **Overhead Analysis**
- **TTFT (Time to First Token)**: +10% increase (0.122s vs 0.111s) - **Negligible**
- **ITL (Inter-Token Latency)**: +14% increase (0.0032s vs 0.0028s) - **Negligible**
