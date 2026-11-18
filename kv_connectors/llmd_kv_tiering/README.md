# KV tiering (`kvtierd`)

A node-level KV cache tier for LLM inference engines. A Go daemon, `kvtierd`, runs once per node and owns a shared-memory pool of KV blocks. Each engine loads a thin shim that copies blocks between the engine's own CPU offload tier and that pool.

Status: the daemon and the vLLM tier on GPU. Router hints with peer transfer between daemons, and a shim for vLLM on TPU, follow in later changes. See the [roadmap](docs/design.md#roadmap).

## How it relates to `llmd_fs_backend`

[`llmd_fs_backend`](../llmd_fs_backend) replaces vLLM's offloading spec and stores blocks as files on a shared filesystem. This tier plugs in differently: it is a secondary tier behind vLLM's own CPU tier, and it keeps blocks in host memory owned by a per-node daemon. The two can be used together. A future SSD tier under the daemon's pool could reuse the FS backend's file and GPUDirect Storage code.

## Why a daemon

- **Blocks outlive the engine.** The daemon owns the memory. When an engine restarts, the blocks it stored are still loadable.
- **Engines on a node share one pool.** A block one engine stored is a hit for every other engine with the same model layout.
- **One implementation for every engine and accelerator.** Policy, eviction, hints and peer transfer live in the daemon. Only the copy into the engine's memory is engine-specific.

## How it plugs into vLLM

vLLM's `TieringOffloadingSpec` loads out-of-tree secondary tiers through `module_path`. No vLLM patch is needed.

```bash
pip install ./kv_connectors/llmd_kv_tiering

vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 20000000000,
      "secondary_tiers": [{
        "type": "LLMDKVTierManager",
        "module_path": "llmd_kv_tiering.vllm_tier",
        "socket_path": "/var/run/llmd-kv-tiering/kvtierd.sock"
      }]
    }
  }'
```

Set `PYTHONHASHSEED` to the same value on every engine. vLLM's block hashes depend on it. Without it, blocks from before a restart, or from another engine, never match.

Tier options:

| Option | Default | Meaning |
| --- | --- | --- |
| `socket_path` | `/var/run/llmd-kv-tiering/kvtierd.sock` | The daemon's socket |
| `namespace` | derived from the model, dtype, block layout and parallelism | Engines share blocks only within a namespace |
| `num_io_threads` | 4 | Threads that copy blocks |
| `timeout_s` | 5 | Per-request timeout to the daemon |
| `reconnect_interval_s` | 1 | How often to retry a lost daemon |
| `locality` | unset | `LOCAL` or `REMOTE`, reported on KV events |
| `enable_kv_events` | true | Report this tier's blocks on vLLM's KV events, if KV events are enabled |

If the daemon is down, the tier misses every lookup and fails every store. vLLM recomputes. The tier reconnects on its own.

## Running the daemon

```bash
go build -o bin/kvtierd ./services/kvtierd
./bin/kvtierd --socket /var/run/llmd-kv-tiering/kvtierd.sock --capacity 512GiB
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--capacity` | 16GiB | Pool size |
| `--slot-bytes` | 0 | Block size. 0 takes it from the first engine |
| `--hugetlb` | false | Back the pool with 2 MiB huge pages |
| `--prefault` | true | Touch the pool at start, so its memory is charged to the daemon's container |
| `--policy` | lru | Eviction policy |
| `--metrics-addr` | :9400 | `/metrics` and `/healthz` |

On Kubernetes, run the daemon as a sidecar or a DaemonSet. Share the socket directory with the engine through an `emptyDir` or `hostPath` volume.

## Development

From the repository root:

```bash
go test -race ./pkg/kvtier/...
cd kv_connectors/llmd_kv_tiering && PYTHONPATH=<vllm source tree>:. python -m pytest tests
```

The Python tests build `kvtierd` and run it. They need a vLLM source tree that has `vllm/v1/kv_offload/tiering`, and a Python environment with vLLM's dependencies. No GPU is needed.

## Layout

```
services/kvtierd                     daemon entry point
pkg/kvtier/pool                      shared-memory arena, block index, pins, eviction
pkg/kvtier/policy                    eviction policies
pkg/kvtier/protocol                  daemon wire format
pkg/kvtier/server                    Unix socket server
kv_connectors/llmd_kv_tiering        engine shims: vLLM on GPU
kv_connectors/llmd_kv_tiering/docs   design, roadmap and results
```
