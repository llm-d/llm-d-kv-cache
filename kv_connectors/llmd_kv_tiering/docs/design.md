# Design

## Goal

Give llm-d a node-level KV tier that outlives engines, is shared by every engine on the node, moves blocks between nodes when the router asks, and runs on GPU and TPU. llm-d controls the hint actions the router can send, and can add new ones as its routing needs them.

## Architecture

```
            llm-d router (EPP)
                 | requests
   +-------------+--------------------------+
   | node                                   |
   |  vLLM engine            vLLM engine    |
   |  [GPU HBM]              [GPU HBM]      |
   |      |                      |          |
   |  [primary CPU tier]     [primary CPU]  |   vLLM's OffloadingConnector
   |      |  shim                |  shim    |   llmd_kv_tiering.vllm_tier
   |      +---------+------------+          |
   |                |  Unix socket + shared memory
   |            kvtierd (Go)                |
   |       [shared-memory pool]             |
   +----------------------------------------+
```

vLLM moves KV between GPU and its primary CPU tier. Secondary tiers only ever see host memory. That lets the daemon be a separate process with no GPU code.

**Control plane.** The shim talks to the daemon over a Unix socket. Each frame is a 4-byte length and one msgpack map. Requests: `hello`, `alloc`, `commit`, `pin`, `unpin`, `snapshot`, `stats`. The daemon pushes `event` frames when blocks are stored or evicted.

**Data plane.** The daemon creates the pool with `memfd_create` and passes the file descriptor on the `hello` response with `SCM_RIGHTS`. The shim maps it. A store is one `memmove` from the primary tier into a pool slot. A load is one `memmove` back. The copies run on shim worker threads, and `ctypes.memmove` releases the GIL.

**Lookups are local.** vLLM calls `lookup` on the scheduler thread for every block, so it must not block. The shim keeps a mirror of the keys in its namespace. It fills the mirror from a snapshot at connect and keeps it current from events. A stale mirror entry costs one failed load, which vLLM recomputes.

**Store protocol.** `alloc` reserves slots and evicts if needed. The shim copies, then calls `commit`. A block is invisible to loads until it is committed.

**Load protocol.** `pin` returns slots and protects them from eviction. The shim copies, then calls `unpin`. When a session disconnects, the daemon drops its pins and frees slots it was still writing.

**Namespaces.** Engines share a block only when their bytes are interchangeable. The default namespace hashes the model, dtype, tokens per hash, blocks per chunk, block size in bytes, TP, PP, PCP, DCP and the replicated-layout flag.

**Memory accounting.** Pages of a memfd are charged to the memory cgroup of the process that first touches them. The daemon prefaults the pool at start by default, so the pool counts against the daemon's container and not the engine's.

## Upstream dependencies

| Engine | What we need | Status |
| --- | --- | --- |
| vLLM | Out-of-tree secondary tiers via `module_path` | In vLLM main. vLLM calls the API experimental. |
| SGLang | A HiCache storage backend | Not started. HiCache has a storage backend interface that a shim could implement. |

## Roadmap

1. **Done.** Daemon with a shared-memory pool, LRU eviction, pins and session cleanup. vLLM shim with store, load, KV events, engine-restart survival and reconnect. Tested against vLLM's real `TieringOffloadingManager` on CPU.
2. Router hints (`kv.fetch`, `kv.evict_on_fetch`, `kv.prefetch`, `kv.move`) carried in `kv_transfer_params` or `kv_hints`, and peer transfer between daemons over TCP. The next change.
3. TPU shim for tpu-inference, for single-host replicas: save to the pool, pool hits, peer fetch hints and engine-restart survival. A later change.
4. Multi-host TPU replicas. One daemon per host. A block counts as a hit only when every host's daemon holds it, and a load is confirmed by every host. Hints name one endpoint per host, and host k pulls from host k of the source replica.
5. A local SSD tier under the pool, and a policy interface that can be swapped at run time.
6. llm-d router integration. The router consumes block events and emits hints. It publishes each pod's daemon endpoints. Prefetch of the next turn.
7. NIXL/RDMA peer transfer through cgo, an object-store tier, and an SGLang shim.

## Open questions

- **Zero-copy stores.** vLLM's primary tier is also a shared-memory file. If the shim passes its file descriptors to the daemon, the daemon can copy, or send to a peer, without a hop through the shim. The files are unlinked after open, so this needs `SCM_RIGHTS` from the shim for each segment.
- **Mirror size.** The snapshot is one message. At about 40 bytes per key, a pool of a million blocks is a 40 MB snapshot. Paging it may be needed.
- **Multiple block sizes.** The pool has one slot size today. A node serving two models with different block sizes needs one arena per size.
