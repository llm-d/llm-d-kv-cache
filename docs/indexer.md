# KV-Cache Indexer

The KV-Cache Indexer is a high-performance Go library that maintains a global, near-real-time view of KV-Cache block locality across a fleet of vLLM pods.
It ingests `KVEvents` streamed from vLLM, tracks which blocks reside on which nodes and tiers, and exposes a scoring API for KV-cache-aware scheduling.

See [Architecture](architecture.md) for high-level data flow diagrams showing how the indexer, event processing, and block index interact.

-----

## Block Hashing & Key Generation

To guarantee compatibility, the block key generation logic exactly matches vLLM's content-addressing scheme.

* **Token Chunking**: Prompts are converted to tokens, then grouped into fixed-size chunks (default: 16).
* **Hash Algorithm**: A chained hash is computed. Each block's key is an **FNV-64a hash** generated from the CBOR-encoded `[parentHash, tokenChunk, extra]` tuple.
* **Initialization**: The hash chain starts with a configurable `HashSeed`.
* **Extra Parameter**: The third component of the hash tuple enables cache differentiation:
  - **nil** (default): Standard prompts without LoRA or multi-modal content
  - **int**: LoRA adapter ID (e.g., 42)
  - **string**: Adapter name or content-affecting identifier (e.g., "lora-v2")
  - **map**: Structured metadata (e.g., `{"lora_id": 42, "medium": "gpu"}`)

Different `extra` values produce different block hashes, preventing cache pollution when the same tokens are used with different adapters or multi-modal inputs.

-----

## Index Backends

The `kvblock.Index` is an interface with swappable backends.

### In-Memory (Default)

A fast, thread-safe, two-level LRU cache using `hashicorp/golang-lru`. The first level maps a block key to a second-level cache of pods that have the block. Prioritizes speed over persistence, which is usually the right trade-off for ephemeral cache data.

### Cost-Aware Memory

A memory-efficient implementation using `hypermodeinc/ristretto` that provides cost-aware eviction based on actual memory usage. Unlike the basic in-memory backend, it calculates the memory footprint of each cache entry and uses this for eviction decisions. Useful when memory usage patterns vary significantly across different keys.

### Redis

A distributed backend that can be shared by multiple indexer replicas. Offers scalability and persistence, but may be overkill given the short lifetime of most KV-cache blocks.

### Valkey

A Redis-compatible, open-source alternative under the BSD license. Provides the same distributed capabilities as Redis with additional RDMA support for reduced latency. API-compatible with Redis, so it can be used as a drop-in replacement.

-----

## KV-Event Processing

The `kvevents` package handles ingestion and processing of KV-cache events streamed from vLLM pods, keeping the block index up to date in near-real-time.

### Engine Keys vs. Request Keys

The index maintains two types of block keys:

- **Engine keys**: Opaque block hashes produced by vLLM's engine and carried inside `KVEvents`. These arrive in `BlockStored` and `BlockRemoved` events.
- **Request keys**: Deterministic hashes computed by the library from the event's token IDs using the same FNV-64a + CBOR scheme used in the read path (see [Block Hashing](#block-hashing--key-generation) above).

When a `BlockStored` event arrives, the library resolves the parent engine hash to its request key (via the index), then recomputes request keys from the tokens. Both the engine-to-request key mapping and the request-key-to-pod entries are stored. `BlockRemoved` events reference engine keys, which are resolved to request keys through the stored mapping.

This design decouples the index from vLLM's internal hash implementation: the read path computes request keys from a prompt's tokens, and the write path independently computes the same request keys from the event's tokens, so the two always agree on block identity without depending on engine hash stability.

### Event Processing Pool

The `kvevents.Pool` is a sharded worker pool that consumes ZMQ messages from vLLM pods.

**Message flow:**

1. A `zmqSubscriber` receives a ZMQ message and parses the topic (`kv@pod-id@model`) to extract the pod identifier and model name.
2. The pool hashes the pod identifier (FNV-1a) to select a worker shard. This guarantees that events from the same pod are always processed in order.
3. The worker decodes the msgpack payload (which can contain a batch of events), recomputes request keys from the event tokens, and updates the index.

**Event types:**

| Event | Data | Action |
|:------|:-----|:-------|
| `BlockStored` | Engine block hashes, parent hash, token IDs, device tier, LoRA metadata | Recompute request keys from tokens; store engine→request mapping and request→pod entries |
| `BlockRemoved` | Engine block hash | Resolve to request key; evict pod entry |
| `AllBlocksCleared` | - | No-op (pod-level cleanup) |

### Pod Discovery

The pool supports two modes for connecting to vLLM pods:

**Static Endpoint Mode** - Connects to a single ZMQ endpoint. Useful for development or when a ZMQ proxy aggregates events.

**Auto-Discovery Mode (Default)** - Uses a Kubernetes pod reconciler to automatically discover vLLM pods and create per-pod ZMQ subscribers.
The reconciler watches pods matching a label selector and manages subscriber lifecycle based on pod readiness.

A pod must meet all of the following conditions for a subscriber to be created:
- Labels match the configured `podLabelSelector`
- `pod.Status.Phase == Running`
- `pod.Status.PodIP != ""`
- Pod has condition `PodReady == ConditionTrue`

When any condition becomes false, the subscriber is automatically removed.

-----

## Configuration

See [Configuration](configuration.md) for all indexer, block index, and event processing options including pod discovery settings and RBAC requirements.
