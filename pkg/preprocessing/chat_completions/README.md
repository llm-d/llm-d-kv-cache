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

The following is the major request structure used for templating:
- Some fields are provided by the router (serving vLLM's OpenAI-compatible API).
- Some fields are fetched from the model's tokenizer (e.g., chat template).

The `ApplyChatTemplateRequest` matches the `transformers` library's `ChatTemplateRequest` structure, which is used to render the chat template.

**ApplyChatTemplateRequest accepts these fields, that match the `apply_chat_template`'s expected parameters:**
- `Conversation` - List of message lists (role/content pairs)
- `Tools` - (Optional) List of tool schemas
- `Documents` - (Optional) List of document dicts
- `ChatTemplate` - (Optional) Override for the chat template
- `ReturnAssistantTokensMask` - (Optional) Whether to return assistant token indices
- `ContinueFinalMessage` - (Optional) Whether to continue from the final message
- `AddGenerationPrompt` - (Optional) Whether to add a generation prompt
- `ChatTemplateKWArgs` - (Optional) Extra parameters for template rendering

See the transformers library's [code documentation](https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/processing_utils.py#L390).
And the vLLM OpenAI API [documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_1).

### ChatTemplate Processing Flow

The templating process (steps 1.1-1.4) handles the conversion from structured request to flattened prompt:

```
1.1. **CGO Binding**: chattemplatego.NewChatTemplatingProcessor()
    └── cgo_functions.go:NewChatTemplatingProcessor()
        └── Creates ChatTemplatingProcessor struct with initialized=false

1.2. **ChatTemplate Rendering**: wrapper.ApplyChatTemplate(ctx, req)
    ├── cgo_functions.go:ApplyChatTemplate(ctx, req)
    │   ├── Initialize() Python interpreter via CGO (if not already done)
    │   ├── executePythonCode() - **CGO Binding** to Python
    │   └── **Python Wrapper**: tokenizer_wrapper.py:apply_chat_template()
    │       └── from vllm.transformers_utils.tokenizer import get_tokenizer
    │           tokenizer = get_tokenizer(...)
    │           rendered = tokenizer.apply_chat_template(...)
    │           └── Uses vllm library's core template rendering functionality
    └── Returns: String

1.3. **Tokenization**: wrapper.Encode(ctx, req)
  ├── cgo_functions.go:Encode(ctx, req)
  │   ├── Initialize() Python interpreter via CGO (if not already done)
  │   ├── executePythonCode() - **CGO Binding** to Python
  │   └── **Python Wrapper**: tokenizer_wrapper.py:encode()
  │       └── from vllm.transformers_utils.tokenizer import get_tokenizer
  │           tokenizer = get_tokenizer(...)
  │           encoded = tokenizer(...)
  │           return json.dumps(encoded.data)  # {"input_ids": [...], "offset_mapping": [...]}
  └── Returns: JSON string with input_ids and offset_mapping

1.4. **Extract Flattened Prompt**
    └── prompt := response
    └── Continue with existing pipeline: Tokenize → KV Block Keys → Pod Scoring
```
### Optimized Preprocessing Architecture

#### **Performance Optimizations**

##### **Single Python Interpreter**
- **Process-Level Initialization**: Single Python interpreter per process, initialization at EPP startup. Scalable, low overhead and reduces memory footprint
- **Thread-Safe Initialization**: Global locks prevent multiple initializations

##### **Function Caching**
- **Cached Python Functions**: `apply_chat_template` and `encode` cached globally
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
