# Valkey Example for KV-Cache

This example demonstrates how to use Valkey as the backend for the KV-Cache's KV-block indexing system.

## What is Valkey?

Valkey is a community-forked version of Redis that remains under the original BSD license. It's fully API-compatible with Redis and offers additional features like RDMA support for improved latency in high-performance scenarios.

## Benefits of Using Valkey

- **Open Source**: Remains under the BSD license
- **Redis Compatibility**: Drop-in replacement for Redis
- **RDMA Support**: Lower latency networking for high-performance workloads (Note: RDMA is not yet supported in the Go client library - see [RDMA Limitations](#rdma-limitations) below)
- **Community Backed**: Supported by major cloud vendors and Linux Foundation
- **Performance**: Optimizations for modern hardware

## Prerequisites

1. **Valkey Server**: Install and run a Valkey server
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 valkey/valkey:latest
   
   # Or install from source/package manager
   ```

2. **Go Environment**: Go 1.24.1 or later

3. **Optional**: Hugging Face token for tokenizer access
   ```bash
   export HF_TOKEN="your-huggingface-token"
   ```

## Running the Example

### Basic Usage

```bash
# Run with default Valkey configuration
make run-example valkey

# Run with custom Valkey address
VALKEY_ADDR="valkey://your-valkey-server:6379" make run-example valkey
```

### With RDMA Support

**Note**: RDMA is currently not supported in the Go client library. The configuration flag is a placeholder for future support. See [RDMA Limitations](#rdma-limitations) for details.

```bash
# This will run but RDMA will not be active (falls back to TCP)
VALKEY_ADDR="valkey://rdma-valkey-server:6379" \
VALKEY_ENABLE_RDMA="true" \
make run-example valkey
```

### Environment Variables

- `VALKEY_ADDR`: Valkey server address (default: `valkey://127.0.0.1:6379`)
- `VALKEY_ENABLE_RDMA`: Enable RDMA transport flag (default: `false`) - **Note: Currently non-functional, see [RDMA Limitations](#rdma-limitations)**
- `HF_TOKEN`: Hugging Face token for tokenizer access (optional)

## What the Example Does

1. **Configuration**: Sets up a KV-Cache indexer with Valkey backend
2. **Cache Operations**: Demonstrates storing KV-block hashes (derived from tokenized prompts) mapped to pod identifiers
3. **Cache Lookup**: Shows how to query which pods have specific KV-blocks cached
4. **Scoring**: Demonstrates computing cache hit scores based on consecutive prefix matches
5. **Multi-Pod Operations**: Shows cache sharing and lookups across multiple pods
6. **Eviction**: Demonstrates removing KV-block entries from the Valkey backend
7. **Metrics**: Enables metrics collection for monitoring cache performance

**Important**: The system stores mappings of `KV-block hash → pod IDs`, not the prompts themselves. Prompts are tokenized, chunked, and hashed to generate KV-block keys for lookup.

## Expected Output

```
I0104 10:30:00.123456       1 main.go:49] Initializing KV-Cache with Valkey backend valkeyAddr="valkey://127.0.0.1:6379" rdmaEnabled=false
I0104 10:30:00.234567       1 main.go:122] Processing testdata prompt model="bert-base-uncased" promptLength=3489
I0104 10:30:00.345678       1 main.go:130] Initial cache scores (should be empty) scores=map[]
I0104 10:30:00.456789       1 main.go:133] Adding cache entries manually to demonstrate Valkey backend
I0104 10:30:00.567890       1 main.go:148] Added cache entries keys=4 pods=2
I0104 10:30:00.678901       1 main.go:156] Cache scores after adding entries scores=map[demo-pod-1:1.0 demo-pod-2:1.0]
I0104 10:30:00.789012       1 main.go:165] Cache lookup results keysFound=4
I0104 10:30:00.890123       1 main.go:167] Key found key="bert-base-uncased:9377470987350831920" pods="[{demo-pod-1 gpu} {demo-pod-2 gpu}]"
...
I0104 10:30:01.123456       1 main.go:191] Final cache scores scores=map[demo-pod-1:0.75 demo-pod-2:1.0]
I0104 10:30:01.234567       1 main.go:69] Valkey example completed successfully
```

The example uses a test prompt from `examples/testdata/prompt.txt` (Lorem Ipsum text), tokenizes it with the `bert-base-uncased` model, and demonstrates KV-block hash lookups through the Valkey backend.

## Comparison with Redis

The Valkey backend is API-compatible with Redis, so you can easily switch between them:

### Redis Configuration
```json
{
  "kvBlockIndexConfig": {
    "redisConfig": {
      "address": "redis://127.0.0.1:6379"
    }
  }
}
```

### Valkey Configuration  
```json
{
  "kvBlockIndexConfig": {
    "valkeyConfig": {
      "address": "valkey://127.0.0.1:6379",
      "backendType": "valkey",
      "enableRDMA": false
    }
  }
}
```

## Performance Considerations

- **Connection Pooling**: The underlying Redis client handles connection pooling automatically
- **Persistence**: Valkey data persists across restarts (unlike in-memory backends)
- **Scalability**: Suitable for distributed deployments with multiple indexer replicas
- **RDMA**: See [RDMA Limitations](#rdma-limitations) below - currently not supported in Go clients

## RDMA Limitations

**Current Status**: RDMA support is **not yet available** in Go client libraries for Valkey.

While Valkey server supports RDMA transport for ultra-low latency networking, neither the Go Redis client (`go-redis/redis`) nor the Valkey Go client ([`valkey-io/valkey-go`](https://github.com/valkey-io/valkey-go)) currently expose configuration options to enable RDMA connections. The `enableRDMA` configuration flag in this codebase is a placeholder for future support.

**What happens when you enable RDMA:**
- The configuration flag is accepted and stored
- A warning message is logged: "RDMA requested for Valkey but not yet supported in Go client - using TCP"
- The connection falls back to standard TCP transport
- All functionality works normally, just without RDMA benefits

**Future Support:**
When RDMA support becomes available, it will require migrating from `go-redis/redis` to the `valkey-io/valkey-go` client, as Redis does not support RDMA.

## Troubleshooting

### Connection Issues
- Ensure Valkey server is running and accessible
- Check network connectivity and firewall rules
- Verify the address format (supports `valkey://`, `redis://`, or plain addresses)

### RDMA Issues
- **Note**: RDMA is not currently supported in Go clients - see [RDMA Limitations](#rdma-limitations)
- If enabling RDMA in the future: Confirm Valkey server is compiled with RDMA support, verify RDMA hardware and drivers are properly configured, and check that both client and server are on RDMA-enabled networks

### Performance Issues
- Monitor cache hit rates using the built-in metrics
- Adjust block size in TokenProcessorConfig for your use case
- Consider using multiple Valkey instances for horizontal scaling

## See Also

- [Valkey Configuration Guide](../valkey_configuration.md)
- [KV-Cache Architecture](../../docs/architecture.md)
- [Configuration Reference](../../docs/configuration.md)