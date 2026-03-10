# Tokenization

Tokenization and prompt preprocessing for the KV-Cache subsystem. This component includes a Go tokenizer pool with pluggable backends,
a Python gRPC sidecar, and vLLM-compatible chat template rendering.

See [Architecture](architecture.md) for how tokenization fits into the scoring (read) path.

-----

## Tokenization Pool

The `tokenization.Pool` (`pkg/tokenization/`) manages a configurable number of worker goroutines for tokenization requests.
The tokenization output feeds into the `kvblock.TokenProcessor` for block key generation (see [Indexer](indexer.md)).

It supports two modes:

- **Synchronous**: Used by scoring requests to ensure complete results are always returned.
- **Asynchronous** (fire-and-forget): Used for background/preemptive tokenization.

-----

## Tokenizer Backends

The system supports multiple tokenizer backends with a composite fallback strategy.

### CachedLocalTokenizer

Loads tokenizers from local files on disk. Useful for air-gapped environments, custom tokenizers, or pre-loaded models.

Supports:
- **Manual Configuration**: Direct mapping from model names to tokenizer file paths.
- **Auto-Discovery**: Recursive scanning of a directory for tokenizer files.
- **HuggingFace Cache Structure**: Automatically detects and parses HF cache directories (e.g., `models--Qwen--Qwen3-0.6B/snapshots/{hash}/tokenizer.json` → `Qwen/Qwen3-0.6B`).
- **Custom Directory Structures**: Arbitrary nesting (e.g., `/mnt/models/org/model/tokenizer.json` → `org/model`).

### CachedHFTokenizer

Downloads and caches tokenizers from HuggingFace Hub. Wraps HuggingFace's Rust tokenizers and maintains an LRU cache of active instances.

### UDS Tokenizer Client

Delegates tokenization to the UDS Tokenizer Service (see below) over gRPC via Unix Domain Sockets.
This avoids the need for embedded Python/cgo tokenizers in the Go process.

### CompositeTokenizer (Default)

Tries backends in order, enabling graceful fallback. The default configuration attempts local tokenizers first, then falls back to HuggingFace if the model isn't found locally.

### Caching

All tokenizer implementations maintain an LRU cache of loaded tokenizer instances to minimize repeated loading from disk.

-----

## UDS Tokenizer Service

**Python service** - [`services/uds_tokenizer/`](../services/uds_tokenizer/)

A sidecar service that provides tokenization and chat template rendering via gRPC over Unix Domain Sockets.
Designed to run alongside the inference scheduler in Kubernetes, it eliminates the need for embedded Python/cgo tokenizers in the Go process.

The service supports HuggingFace and ModelScope models, with automatic downloading and caching.
It exposes a health check endpoint for Kubernetes probes.

See the [UDS Tokenizer README](../services/uds_tokenizer/README.md) for setup, API reference, environment variables, and Kubernetes deployment.

-----

## Chat Completions Preprocessing

**Go/Python (cgo)** - [`pkg/preprocessing/chat_completions/`](../pkg/preprocessing/chat_completions/)

Applies vLLM-compatible chat template rendering so that `/v1/chat/completions` requests can be preprocessed before scoring.
Uses a Python binding under the hood to match vLLM's Jinja2 templating behavior exactly.

This is required when the indexer needs to score chat-format requests - the messages must first be rendered into a flat prompt string
before tokenization and block key generation can proceed.

-----

## Configuration

See [Configuration - Tokenization](configuration.md#tokenization-configuration) for all options including pool size, backend selection, and auto-discovery settings.

For UDS Tokenizer Service configuration, see the [UDS Tokenizer README](../services/uds_tokenizer/README.md#environment-variables).
