# KVCacheIndex Example

This example demonstrates how to configure and use the `kvcache.Indexer` module from the `llm-d-kv-cache` project.

## What it does

- Initializes a `kvcache.Indexer` with optional Redis, in-memory backend, or cost-aware memory.
- Optionally uses a HuggingFace token for tokenizer pool configuration.
- Demonstrates adding and querying KV cache index entries for a model prompt.
- Shows how to retrieve pod scores for a given prompt.

## Usage

1. **Set environment variables as needed:**

   - `REDIS_ADDR` (optional): Redis connection string (e.g., `redis://localhost:6379/0`). If unset, uses in-memory index.
   - `HF_TOKEN` (optional): HuggingFace token for tokenizer pool.
   - `MODEL_NAME` (optional): Model name to use (defaults to test data).

2. **Run the example:**

```sh
make run-example kv_cache_index
```

3. **What to expect:**

   - The program will print logs showing the creation and startup of the indexer.
   - It will attempt to get pod scores for a test prompt (initially empty).
   - It will manually add entries to the index and then retrieve pod scores again.

## Example output

```
I... Created Indexer
I... Started Indexer model=...
I... Got pods pods=[]
I... Got pods pods=[{pod1 gpu}]
```

## See also

- [`main.go`](./main.go) for the full example code.
- [`testdata`](../testdata) for sample prompts and model names.
