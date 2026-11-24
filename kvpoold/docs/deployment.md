# Deployment

The server and every engine on the node mount the host's `/dev/shm` at the same path, e.g. `/host-shm`, so the paths in the server's `ShmFileRegion` are the ones the engines open. The region's files are mode 0660, so engines in other pods open them through a group they share with the server's pod.

## Running `kvpoold`

```bash
kvpoold \
    --pool-bytes $((64 << 30)) \
    --address ipc:///host-shm/kvpoold.sock \
    --region-dir /host-shm \
    --cache-policy lru \
    --lease-seconds 30 \
    --fence-seconds 10 \
    --kv-events-endpoint 'tcp://*:5557' \
    --kv-events-replay-endpoint 'tcp://*:5558' \
    --kv-events-topic 'kv@${NODE_NAME}@<model>'
```

| Flag | Default | Meaning |
|---|---|---|
| `--pool-bytes` | required | Size of the pool. Its memory is allocated in RAM at the first attach and charged to the server's pod, so the pod's memory limit must stay above it. |
| `--address` | required | ZMQ endpoint to bind; engines set it as `pool_address`. |
| `--region-dir` | required | Directory for the region's files (`kvpoold.mmap*`), the spec file (`kvpoold.spec`) and the snapshot (`kvpoold.snapshot`). |
| `--cache-policy` | `lru` | vLLM's cache policy for the pool, by name: `lru` or `arc`. Only LRU records are snapshotted. |
| `--cache-policy-module-path` | none | Module to load `--cache-policy` from when it is not built into vLLM. |
| `--lease-seconds` | 30 | How long a pin or write lasts unless the engine releases it first; far longer than the longest copy. |
| `--fence-seconds` | 10 | After a clean restart, how long the snapshot's pins and writes are held; after a crash, how long no chunk is handed out. Longer than the longest copy plus the time an engine takes to see a request time out. |
| `--kv-events-endpoint` | required | Where to publish the pool's KV events. Bound when it has a `*` or starts with `ipc://`, connected to otherwise. |
| `--kv-events-topic` | required | The topic of every batch, not empty. The indexer reads the pod and the model from `kv@<pod>@<model>`. |
| `--kv-events-replay-endpoint` | none | A ROUTER the server binds. A subscriber sends the first sequence number it wants, as 8 big-endian bytes, and gets the buffered batches from it on, then a frame with sequence number -1. |
| `--kv-events-buffer-steps` | 10000 | How many past batches are kept for replay. |
| `--log-level` | `INFO` | Level of the server's log. |

SIGTERM and SIGINT stop the server cleanly: it drains, publishes its last events, and writes the snapshot. The region's files stay for the next instance.

## Configuring an engine

```bash
vllm serve <model> \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq",
                       "endpoint": "tcp://*:5557", "topic": "kv@<pod>@<model>"}' \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "PoolOffloadingSpec",
      "canonical_layout": true,
      "pool_address": "ipc:///host-shm/kvpoold.sock",
      "self_describing_kv_events": true
    }}'
```

`spec_name` resolves through the package's `vllm.general_plugins` entry point. Without the package installed as a distribution, set `"spec_module_path": "kvpoold.adapters.vllm.spec"` as well.

| Key | Default | Meaning |
|---|---|---|
| `pool_address` | required | The server's `--address`. |
| `hash_function` | `sha256` | The engine's `--prefix-caching-hash-algo`, which vLLM does not pass to offloading specs. The server refuses an engine whose hash function or `PYTHONHASHSEED` differs from the pool's. |
| `attach_timeout_ms` | 300000 | How long a process waits for its attach answer; the first attach waits while the server creates and pre-faults the pool. |
| `call_timeout_ms` | 1000 | How long the scheduler waits for one lookup or prepare reply. A server that does not answer in time is treated as gone for the rest of the step. |
| `self_describing_kv_events` | false | With `--kv-events-config` enabling KV cache events, the engine sends each stored chunk's `BlockStored` for the server to publish. Without both, the engine's chunks are not announced. |

Set the same `PYTHONHASHSEED` in every engine, or leave it unset in all of them.

The engine's own KV events (its GPU cache) go out on its own `--kv-events-config` endpoint and topic, as without kvpoold.
