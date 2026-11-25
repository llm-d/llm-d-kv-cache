# Architecture

## The pool and its region

`kvpoold` runs once per node. It owns the pool's records; the KV data lives in a shared region that every engine on the node maps.

- **The region** is a set of segment files in the host's `/dev/shm` (`kvpoold.mmap`, `kvpoold.mmap.1`, ...), cut by vLLM's own `SharedOffloadRegion` split. The first accepted attach creates them and pre-faults every page, so the memory is allocated in RAM and charged to the server's pod. The server keeps neither a mapping nor a file descriptor of them: the files keep their pages with no process mapping them, and they outlive the server. The attach reply describes them as a `ShmFileRegion`; each GPU worker maps its own rank's part.
- **The records** are vLLM's own `CPUOffloadingManager` over the region's chunks, shared by every engine, so chunk allocation, eviction and the cache policy are vLLM's. The server adds who holds what (below).
- **Copies** never go through the server. A GPU worker copies KV data between its GPU and the region with vLLM's stock `CPUOffloadingWorker`. The server only hands out chunk ids.

## Layout and namespace

Every attach carries a `CompatDescriptor`:

- **Layout**: `kv_bytes_per_chunk`, the bytes of one chunk across every rank of the engine. The first accepted attach fixes the region's row stride to it. The server refuses an engine with another layout.
- **Namespace**: 8 bytes, the start of a SHA-256 over the run config that vLLM's own secondary tiers compare to decide that two engines' blocks are interchangeable. It leaves out the parallel sizes when the layout allows, so engines with different TP share chunks. The pool keeps a key under its namespace followed by vLLM's key: engines in one namespace read each other's chunks, and other namespaces' chunks stay apart while they share eviction.
- **Hash function and seed**: the engine's `--prefix-caching-hash-algo` and `PYTHONHASHSEED`. The server requires the same pair from every engine, across namespaces.

Every process of one engine (each GPU worker and the scheduler) sends the same descriptor; the server refuses a process whose namespace differs from its engine's.

## Who holds what: pins, writes and leases

- **A pin** is one unit of a key's `ref_cnt`, held by one engine. `Lookup` pins each hit, so no other engine evicts it before this engine loads it. The pin carries the load that `PrepareLoad` starts, and `CompleteLoad` releases it. At the end of each step, the engine's `ReleaseHits` releases the step's hits that it did not load.
- **A write** is a key with `ref_cnt` -1 that one engine fills, from `PrepareStore` until `CompleteStore`. A failed store drops its chunks.
- **A lease** ends every pin and write `--lease-seconds` after it is granted, unless the engine releases it first. It is far longer than the longest copy, so a live engine never reaches it, and a dead engine's pins and writes end on their own. The server sweeps ended leases between messages. An engine that stops cleanly sends `Leave`, which releases its leases at once.

The server needs no connection to a GPU worker after its attach, because of the leases: it never has to notice that a worker died.

## Detached engines

A scheduler whose server does not answer, answers `UnknownClient`, or answers `Draining` is detached: every lookup misses and every store is skipped, so the engine serves without the pool rather than wait. It attaches again at the end of a step once the server is back, so a step runs either attached throughout or not at all, and its hits are released to the server that granted them.

A GPU worker that the server refuses, or that cannot reach the server or map its region, serves without the pool: it accepts every copy and reports it failed, and the engine recomputes.

## Start and stop

Beside the region's files, the server keeps two files:

- **The spec file** (`kvpoold.spec`) holds the pool's layout, hash function, seed and region. It is written when the region is created, before any engine learns where the region is, so region files without a spec file were never handed to an engine and are removed at start.
- **The snapshot file** (`kvpoold.snapshot`) holds the records at a clean stop. A new instance reads it once and removes it, so the records of an instance that later crashes are never loaded.

On SIGTERM or SIGINT, the server answers the requests already queued with `Draining`, applies the queued notices, publishes their events, writes the snapshot, and shuts its event publisher down.

A new instance finds the region through the spec file:

- **With a snapshot** of the same region, under LRU, it loads the records. The snapshot's pins and writes are held for `--fence-seconds`, then released like any lease.
- **Without one** (a crash, an unreadable snapshot, another region, or a policy other than LRU), it starts with every chunk free. Until `--fence-seconds` pass, every lookup misses and no chunk is handed out, so no store overwrites a chunk an engine may still be copying under the instance that is gone.

## KV events

The server publishes the pool's KV events, which the llm-d KV-cache indexer reads as it reads an engine's: a `BlockStored` when a chunk becomes readable and a `BlockRemoved` when it is evicted, with medium `CPU`. Only the server knows when a chunk is evicted, but only the engine knows a chunk's block hashes, parent and token ids. So at store time the engine builds each chunk's `BlockStored` with vLLM's own `OffloadingEventsTracker` and sends it in `CompleteStore`. The server keeps it with the chunk, publishes it, and builds the chunk's `BlockRemoved` from it at eviction. A chunk from an engine that sends no events is not announced, and a failed store announces nothing.

The server creates vLLM's own `ZmqEventPublisher` from a `KVEventsConfig`, so the wire format, the replay protocol and the flags' meaning are vLLM's `--kv-events-config`. After each burst of messages, it publishes one `KVEventBatch` with the burst's events. Every batch goes out on one topic, set by the deployment as it is for an engine, e.g. `kv@${NODE_NAME}@<model>`. The engine's own manager publishes nothing for the pool, so nothing is published twice.

### Start and restart

The snapshot keeps each readable chunk's `BlockStored`. A new instance that restores a snapshot publishes nothing at start: the indexer already holds exactly the pool's chunks, because every store and eviction was published and the stop published the last ones. A restored chunk's eviction publishes its `BlockRemoved`, as it would have before the restart.

Any other start publishes one `AllBlocksCleared` before the server binds its socket, and so before it answers any engine. The indexer drops every entry of the topic's pod, which here is the server alone; the engines' own entries are on their own topics and stay. After the clear, the indexer learns each chunk when it is stored. A chunk that was being written when the server stopped is not readable; if its store completes after the restart, its `BlockStored` is published then.

A ZMQ PUB socket drops what it sends before a subscriber connects. A subscriber that connects after the `AllBlocksCleared` gets it only by asking the replay endpoint, from sequence number 0, as for any batch it missed. The server does not wait for a subscriber or resend.

## Logging

Every module logs under the `kvpoold` logger. In an engine, the vLLM adapter forwards those records to vLLM's `vllm` logger, so they appear in the engine's log with vLLM's format and level. The server configures the root logger from `--log-level`.
