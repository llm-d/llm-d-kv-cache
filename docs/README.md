# Documentation

## Component Documentation

| Document | Description |
|:---------|:------------|
| [Architecture](architecture.md) | High-level system design and data flows |
| [KV-Cache Indexer](indexer.md) | Block hashing, index backends, event ingestion, pod discovery |
| [Tokenization](tokenization.md) | Tokenizer pool, backends, UDS service, chat preprocessing |
| [Configuration](configuration.md) | Configuration reference for the indexer, event processing, tokenization, and index backends |
| [Deployment](deployment/) | Kubernetes deployment guides and manifests |

## Component-Specific READMEs

- [UDS Tokenizer Service](../services/uds_tokenizer/README.md) - Python gRPC tokenizer sidecar (setup, API reference, Kubernetes deployment)
- [FS Backend Connector](../kv_connectors/llmd_fs_backend/README.md) - vLLM file-system offloading connector (installation, configuration, deployment)

## Examples

See the [examples/](../examples/) directory for runnable demos:

- [KV-Cache Index](../examples/kv_cache_index/README.md) - Using the `kvcache.Indexer` library directly
- [KV-Cache Aware Scorer](../examples/kv_cache_aware_scorer/README.md) - Integrating the indexer into a scheduler
- [KV-Events](../examples/kv_events/README.md) - Offline and online KV-event processing demos
- [KV-Cache Index Service](../examples/kv_cache_index_service/) - gRPC-based indexer service (client/server)
- [Valkey Example](../examples/valkey_example/README.md) - Using Valkey as the index backend
