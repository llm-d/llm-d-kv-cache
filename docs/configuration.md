# Configuration

This document describes all configuration options available in the llm-d KV Cache libraries. 
All configurations are JSON-serializable.

## Main Configuration

This package consists of two components:
1. **KV Cache Indexer**: Manages the KV cache index, allowing efficient retrieval of cached blocks.
2. **KV Event Processing**: Handles events from vLLM to update the cache index.

See the [Architecture Overview](architecture.md) for a high-level view of how these components work and interact.

The two components are configured separately, but share both the index backend for storing KV block localities and the token processor for converting tokens into blocks.
The token processor is configured via the `tokenProcessorConfig` field in the main configuration.
The index backend is configured via the `kvBlockIndexConfig` field in the KV Cache Indexer configuration.

### Main Configuration Structure

The main configuration structure for the llm-d KV Cache system.

```json
{
  "indexerConfig": { ... },
  "kvEventsConfig": { ... },
  "tokenProcessorConfig": { ... }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `indexerConfig` | [IndexerConfig](#indexer-configuration-config) | Configuration for the KV Cache Indexer module | See defaults |
| `kvEventsConfig` | [KVEventsConfig](#kv-event-pool-configuration-config) | Configuration for the KV Event Processing pool | See defaults |
| `tokenProcessorConfig` | [TokenProcessorConfig](#token-processor-configuration-tokenprocessorconfig) | Configuration for token processing | See defaults |

## KV-Cache Indexer Configuration

### Indexer Configuration (`Config`)

The indexer configuration structure for the KV Cache Indexer module.

```json
{
  "prefixStoreConfig": { ... },
  "kvBlockIndexConfig": { ... },
  "tokenizersPoolConfig": { ... },
  "kvCacheBackendConfigs": { ... }
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `prefixStoreConfig` | [LRUStoreConfig](#lru-store-configuration-lrustoreconfig) | Configuration for the prefix store | See defaults |
| `kvBlockIndexConfig` | [IndexConfig](#index-configuration-indexconfig) | Configuration for KV block indexing | See defaults |
| `tokenizersPoolConfig` | [Config](#tokenization-pool-configuration-config) | Configuration for tokenization pool | See defaults |
| `kvCacheBackendConfigs` | [KVCacheBackendConfig](#kv-cache-backend-configuration-kvcachebackendconfig) | Configuration for KV Cache Device Backends | See defaults |


## Complete Example Configuration

Here's a complete configuration example with all options:

```json
{
  "prefixStoreConfig": {
    "cacheSize": 500000,
    "blockSize": 256
  },
  "kvBlockIndexConfig": {
    "inMemoryConfig": {
      "size": 100000000,
      "podCacheSize": 10
    },
    "enableMetrics": true,
    "metricsLoggingInterval": "1m0s"
  },
  "tokenizersPoolConfig": {
    "modelName": "namespace/model-name",
    "workersCount": 8,
    "minPrefixOverlapRatio": 0.85,
    "hf": {
      "huggingFaceToken": "your_hf_token_here",
      "tokenizersCacheDir": "/tmp/tokenizers"
    },
    "local": {
      "autoDiscoveryDir": "/mnt/models",
      "autoDiscoveryTokenizerFileName": "tokenizer.json"
    }
  },
  "kvCacheBackendConfigs": [
    {
      "name": "gpu",
      "weight": 1.0
    },
    {
      "name": "cpu",
      "weight": 0.8
    }
  ]
}
```

## KV-Block Index Configuration

### Index Configuration (`IndexConfig`)

Configures the KV-block index backend. Multiple backends can be configured, but only the first available one will be used.

```json
{
  "inMemoryConfig": { ... },
  "costAwareMemoryConfig": { ... },
  "redisConfig": { ... },
  "enableMetrics": false
}
```

| Field | Type                                                  | Description | Default |
|-------|-------------------------------------------------------|-------------|---------|
| `inMemoryConfig` | [InMemoryIndexConfig](#in-memory-index-configuration) | In-memory index configuration | See defaults |
| `costAwareMemoryConfig` | [CostAwareMemoryIndexConfig](#cost-aware-memory-index-configuration) | Cost-aware memory index configuration | `null` |
| `redisConfig` | [RedisIndexConfig](#redis-index-configuration)        | Redis index configuration | `null` |
| `enableMetrics` | `boolean`                                             | Enable admissions/evictions/hits/misses recording | `false` |
| `metricsLoggingInterval` | `string` (duration) | Interval at which metrics are logged (e.g., `"1m0s"`). If zero or omitted, metrics logging is disabled. Requires `enableMetrics` to be `true`. | `"0s"` |

### In-Memory Index Configuration (`InMemoryIndexConfig`)

Configures the in-memory KV block index implementation.

```json
{
  "size": 100000000,
  "podCacheSize": 10
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `size` | `integer` | Maximum number of keys that can be stored | `100000000` |
| `podCacheSize` | `integer` | Maximum number of pod entries per key | `10` |

### Cost-Aware Memory Index Configuration (`CostAwareMemoryIndexConfig`)

Configures the cost-aware memory-based KV block index implementation using Ristretto cache.

```json
{
  "size": "2GiB"
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `size` | `string` | Maximum memory size for the cache. Supports human-readable formats like "2GiB", "500MiB", "1GB", etc. | `"2GiB"` |

### Redis Index Configuration (`RedisIndexConfig`)

Configures the Redis-backed KV block index implementation.

```json
{
  "address": "redis://127.0.0.1:6379"
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `address` | `string` | Redis server address (can include auth: `redis://user:pass@host:port/db`) | `"redis://127.0.0.1:6379"` |
| `backendType` | `string` | Backend type: "redis" or "valkey" (optional, mainly for documentation) | `"redis"` |
| `enableRDMA` | `boolean` | Enable RDMA transport for Valkey (experimental, requires Valkey with RDMA support) | `false` |

### Valkey Index Configuration (`RedisIndexConfig`) 

Configures the Valkey-backed KV block index implementation. Valkey is a Redis-compatible, open-source alternative that supports RDMA for improved latency.

```json
{
  "address": "valkey://127.0.0.1:6379",
  "backendType": "valkey", 
  "enableRDMA": false
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `address` | `string` | Valkey server address. Supports `valkey://`, `valkeys://` (SSL), `redis://`, or plain address | `"valkey://127.0.0.1:6379"` |
| `backendType` | `string` | Should be "valkey" for Valkey instances | `"valkey"` |
| `enableRDMA` | `boolean` | Enable RDMA transport (requires Valkey server with RDMA support) | `false` |

**Note**: Both Redis and Valkey configurations use the same `RedisIndexConfig` structure since Valkey is API-compatible with Redis.

## Prefix Store Configuration

### LRU Store Configuration (`LRUStoreConfig`)

Configures the LRU-based prefix token store.

```json
{
  "cacheSize": 500000,
  "blockSize": 256
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `cacheSize` | `integer` | Maximum number of blocks the LRU cache can store | `500000` |
| `blockSize` | `integer` | Number of **characters** per block in the tokenization prefix-cache | `256` |

## Tokenization Configuration

### Tokenization Pool Configuration (`Config`)

Configures the tokenization worker pool and cache utilization strategy.

```json
{
  "modelName": "namespace/model-name",
  "workersCount": 5,
  "minPrefixOverlapRatio": 0.8,
  "hf": {
    "enabled": true,
    "huggingFaceToken": "",
    "tokenizersCacheDir": ""
  },
  "local": {
    "autoDiscoveryDir": "/mnt/models",
    "autoDiscoveryTokenizerFileName": "tokenizer.json",
    "modelTokenizerMap": {
      "my-model": "/path/to/custom-model/tokenizer.json"
    }
  }
}
```

| Field                   | Type                   | Description                                                 | Default |
|-------------------------|------------------------|-------------------------------------------------------------|---------|
| `modelName`             | `string`               | Base model name for the tokenizer.                          |         |
| `workersCount`          | `integer`              | Number of tokenization worker goroutines                    | `5`     |
| `minPrefixOverlapRatio` | `float64`              | Minimum overlap ratio to use cached prefix tokens (0.0-1.0) | `0.8`   |
| `hf`                    | `HFTokenizerConfig`    | HuggingFace tokenizer config                                |         |
| `local`                 | `LocalTokenizerConfig` | Local tokenizer config                                      |         |

### Local Tokenizer Configuration (`LocalTokenizerConfig`)

Configures loading tokenizers from local files. Useful for air-gapped environments or when models are pre-loaded.

```json
{
  "autoDiscoveryDir": "/mnt/models",
  "autoDiscoveryTokenizerFileName": "tokenizer.json",
  "modelTokenizerMap": {
    "my-model": "/path/to/custom-model/tokenizer.json"
  }
}
```

| Field                            | Type                | Description                                                                                                   | Default            |
|----------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|--------------------|
| `autoDiscoveryDir`               | `string`            | Directory to recursively scan for tokenizer files. Can be set via `LOCAL_TOKENIZER_DIR` environment variable. | `"/mnt/models"`    |
| `autoDiscoveryTokenizerFileName` | `string`            | Filename to search for during auto-discovery. Can be set via `LOCAL_TOKENIZER_FILENAME` environment variable. | `"tokenizer.json"` |
| `modelTokenizerMap`              | `map[string]string` | Manual mapping from model name to tokenizer file path. Overrides auto-discovered model mappings.              | `{}`               |

**Auto-Discovery Behavior:**

When `autoDiscoveryDir` is set, the system recursively scans the directory for files matching `autoDiscoveryTokenizerFileName`. It supports two directory structure patterns:

1. **HuggingFace Cache Structure** (automatically detected):
   ```
   ~/.cache/huggingface/hub/
     models--Qwen--Qwen3-0.6B/snapshots/{hash}/tokenizer.json
       → Model name: "Qwen/Qwen3-0.6B"
     models--meta-llama--Llama-2-7b-chat-hf/snapshots/{hash}/tokenizer.json
       → Model name: "meta-llama/Llama-2-7b-chat-hf"
   ```

2. **Custom Directory Structure** (arbitrary nesting):
   ```
   /mnt/models/
     llama-7b/tokenizer.json
       → Model name: "llama-7b"
     Qwen/Qwen3/tokenizer.json
       → Model name: "Qwen/Qwen3"
     org/team/model/tokenizer.json
       → Model name: "org/team/model"
   ```

**Environment Variables:**
- `LOCAL_TOKENIZER_DIR`: Overrides the default auto-discovery directory
- `LOCAL_TOKENIZER_FILENAME`: Overrides the default tokenizer filename

### HuggingFace Tokenizer Configuration (`HFTokenizerConfig`)

Configures the HuggingFace tokenizer backend for downloading tokenizers from HuggingFace Hub.

```json
{
  "enabled": true,
  "huggingFaceToken": "",
  "tokenizersCacheDir": "./bin"
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `enabled` | `boolean` | Enable HuggingFace tokenizer backend | `true` |
| `huggingFaceToken` | `string` | HuggingFace API token for accessing private models | `""` |
| `tokenizersCacheDir` | `string` | Local directory for caching downloaded tokenizers | `"./bin"` |

**Note**: The system uses a composite tokenizer by default that tries local tokenizers first, then falls back to HuggingFace tokenizers if enabled and the model is not found locally.

## KV Cache Backend Tiers

### KV Cache Backend Configuration (`KVCacheBackendConfig`)

Configures the available device backends which store the KV Cache blocks. This will be used in scoring. 

```json
{
  [
    {
      "name": "gpu",
      "weight": 1.0,
    },
    {
      "name": "cpu",
      "weight": 0.8,
    }
  ]
}
```

## KV-Event Processing Configuration

### KV-Event Pool Configuration (`Config`)

Configures the ZMQ event processing pool for handling KV cache events.

```json
{
  "zmqEndpoint": "tcp://*:5557",
  "topicFilter": "kv@",
  "concurrency": 4
}
```

## Event Processing Configuration Example

For the ZMQ event processing pool:

```json
{
  "zmqEndpoint": "tcp://indexer:5557",
  "topicFilter": "kv@",
  "concurrency": 8
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `zmqEndpoint` | `string` | ZMQ address to connect to | `"tcp://*:5557"` |
| `topicFilter` | `string` | ZMQ subscription filter | `"kv@"` |
| `concurrency` | `integer` | Number of parallel workers | `4` |

## Token Processing Configuration

### Token Processor Configuration (`TokenProcessorConfig`)

Configures how tokens are converted to KV-block keys.

```json
{
  "blockSize": 16,
  "hashSeed": ""
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `blockSize` | `integer` | Number of tokens per block | `16` |
| `hashSeed` | `string` | Seed for hash generation (should align with vLLM's PYTHONHASHSEED) | `""` |

---
## Notes

1. **Hash Seed Alignment**: The `hashSeed` in `TokenProcessorConfig` should be aligned with vLLM's `PYTHONHASHSEED` environment variable to ensure consistent hashing across the system.

2. **Memory Considerations**: 
   - The `size` parameter in `InMemoryIndexConfig` directly affects memory usage. Each key-value pair consumes memory proportional to the number of associated pods.
   - The `size` parameter in `CostAwareMemoryIndexConfig` controls the maximum memory footprint and supports human-readable formats (e.g., "2GiB", "500MiB", "1GB").

3. **Performance Tuning**: 
   - Increase `workersCount` in tokenization config for higher tokenization throughput
   - Adjust `minPrefixOverlapRatio`: lower values accept shorter cached prefixes, reducing full tokenization overhead
   - Adjust `concurrency` in event processing for better event handling performance
   - Tune cache sizes based on available memory and expected workload

4. **Cache Directories**: If used, ensure the `tokenizersCacheDir` has sufficient disk space and appropriate permissions for the application to read/write tokenizer files.

5. **Redis Configuration**: When using Redis backend, ensure Redis server is accessible and has sufficient memory. The `address` field supports full Redis URLs including authentication: `redis://user:pass@host:port/db`.
