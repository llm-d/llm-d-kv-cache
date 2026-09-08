# Valkey Configuration Example for KV-Cache

This example demonstrates how to configure the KV-Cache indexer to use Valkey as the backend for KV-block indexing.

## Basic Valkey Configuration

```json
{
  "kvBlockIndexConfig": {
    "valkeyConfig": {
      "address": "valkey://127.0.0.1:6379",
      "backendType": "valkey",
      "enableRDMA": false
    },
    "enableMetrics": true,
    "metricsLoggingInterval": "30s"
  }
}
```

## Valkey with RDMA Support

> **Not usable yet.** The Go client has no way to actually enable RDMA, so
> setting `enableRDMA: true` makes the indexer fail to start rather than
> silently connect over TCP. See [RDMA Configuration Notes](#rdma-configuration-notes)
> below. This example config will currently error out at startup.

```json
{
  "kvBlockIndexConfig": {
    "valkeyConfig": {
      "address": "valkey://valkey-server:6379",
      "backendType": "valkey", 
      "enableRDMA": true
    },
    "enableMetrics": true
  }
}
```

## Valkey with SSL/TLS

```json
{
  "kvBlockIndexConfig": {
    "valkeyConfig": {
      "address": "valkeys://valkey-cluster:6380",
      "backendType": "valkey",
      "enableRDMA": false
    }
  }
}
```

## Environment Variables

You can also configure Valkey using environment variables:

```bash
export VALKEY_ADDR="valkey://127.0.0.1:6379"
export VALKEY_ENABLE_RDMA="false"
```

## Migration from Redis

To migrate from Redis to Valkey, simply change the configuration:

**Before (Redis):**
```json
{
  "kvBlockIndexConfig": {
    "redisConfig": {
      "address": "redis://127.0.0.1:6379"
    }
  }
}
```

**After (Valkey):**
```json
{
  "kvBlockIndexConfig": {
    "valkeyConfig": {
      "address": "valkey://127.0.0.1:6379",
      "backendType": "valkey"
    }
  }
}
```

## Benefits of Using Valkey

1. **Open Source**: Valkey remains under the BSD license, ensuring long-term availability
2. **Redis Compatibility**: Drop-in replacement for Redis with full API compatibility
3. **RDMA Support**: Lower latency networking for high-performance workloads
4. **Community Backed**: Supported by major cloud vendors and the Linux Foundation
5. **Performance**: Optimizations specifically for modern hardware

## RDMA Configuration Notes

`enableRDMA: true` is not currently usable: the Go client has no
configuration surface to actually enable RDMA, so `NewValkeyIndex` /
`NewRedisIndex` return an error and refuse to start rather than silently
connecting over TCP as a substitute. Leave `enableRDMA: false` (the default)
until Go client support lands.

When RDMA support does become available:
- Ensure your Valkey server is compiled with RDMA support
- Verify that RDMA hardware and drivers are properly configured
- It will likely require migrating from `go-redis/redis` to a client that
  supports RDMA (e.g. `valkey-io/valkey-go`)