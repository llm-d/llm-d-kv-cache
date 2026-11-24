# External Primary Tier

One CPU KV-cache pool per node, shared by every vLLM engine on it.

A per-node process, the External Primary Tier, owns the pool's records. Each engine selects `ExternalOffloadingSpec` in `--kv-transfer-config`; its offloading manager, `ExternalPrimaryTierClient`, sends every call that reads or changes the pool to the External Primary Tier, and its GPU workers copy KV data directly to and from the shared region.

The design is in the RFC, [llm-d/llm-d#3047](https://github.com/llm-d/llm-d/issues/3047).

## Layout

- `external_primary_tier/protocol/`: the messages between engines and the External Primary Tier, and the region descriptor. As in vLLM's CPU EC connector, each side encodes them with msgspec's msgpack `Encoder` and decodes them with a `Decoder` for the messages it receives, one payload per ZMQ message after an empty delimiter frame.
- `external_primary_tier/tier/`: the External Primary Tier.
  - `server.py`: the request loop, attach, start and stop.
  - `records.py`: the pool's records, which are vLLM's `CPUOffloadingManager` shared by every engine, and each engine's pins and writes.
  - `region.py`: creating the region with vLLM's `SharedOffloadRegion`.
  - `snapshot.py`: the spec and snapshot files kept beside the region.
- `external_primary_tier/client/`: the engine side.
  - `session.py`: one engine process's DEALER to the tier.
  - `spec.py`: `ExternalOffloadingSpec`, the `CPUOffloadingSpec` subclass vLLM builds in the scheduler and in each GPU worker.
  - `manager.py`: `ExternalPrimaryTierClient`, the engine's offloading manager, which sends every manager call to the tier.
  - `plugin.py`: registration through `vllm.general_plugins`, so `spec_name` alone selects the spec.
- `tests/`: unit tests, in `tests/tier/` and `tests/client/`.

## Running

```bash
external-primary-tier \
    --pool-bytes $((64 << 30)) \
    --address ipc:///host-shm/external-primary-tier.sock \
    --region-dir /host-shm \
    --cache-policy lru \
    --lease-seconds 30 \
    --fence-seconds 10 \
    --kv-events-endpoint 'tcp://*:5557' \
    --kv-events-topic 'kv@${NODE_NAME}@<model>'
```

- `--lease-seconds`: every pin and write is released this long after it is granted, unless the engine releases it first. It is far longer than the longest copy, so a live engine's leases never reach it, and a dead engine's leases end on their own.
- `--fence-seconds`: after a clean stop, the next instance loads the snapshot and holds the pins and writes in it this long. After a crash, it starts with every chunk free and hands out no chunk this long.
- `--cache-policy`, `--cache-policy-module-path`: vLLM's cache policy for the pool, by name. A policy that is not built into vLLM is loaded from the module path, as vLLM's own `cache_policy_module_path` does. Only LRU records are snapshotted; with any other policy, the next instance starts empty.
- `--kv-events-endpoint` and `--kv-events-topic` (both required), `--kv-events-replay-endpoint`, `--kv-events-buffer-steps`: KV events, below.

The region's files, `external-primary-tier.mmap*`, outlive the process. The spec file beside them, `external-primary-tier.spec`, records their layout, and a new instance reuses them when they match it.

## KV events

The tier publishes the pool's KV events, which the llm-d KV-cache indexer reads as it reads an engine's: a `BlockStored` when a chunk becomes readable and a `BlockRemoved` when it is evicted, with medium `CPU`. The engine sends the content of each stored chunk (see `client/events.py`), and the tier publishes it. A chunk from an engine with KV events off is not announced.

The tier creates vLLM's own `ZmqEventPublisher` from a `KVEventsConfig` that the flags build, so the wire format, the replay protocol and the flags' meaning are vLLM's `--kv-events-config`. After each burst of requests, the tier publishes one `KVEventBatch` with the burst's events. On stop, it publishes the last events and shuts the publisher down.

- `--kv-events-endpoint` (required): where to publish. It is bound when it has a `*` or starts with `ipc://`, and connected to otherwise.
- `--kv-events-topic` (required, not empty): the topic of every batch. The indexer subscribes by topic prefix (`kv@` by default) and reads the pod and the model from `kv@<pod>@<model>`, so an empty topic would reach no indexer. A tier serves one model, so the deployment gives it the one topic, as it gives an engine its own, e.g. `kv@${NODE_NAME}@<model>`.
- `--kv-events-replay-endpoint`: a ROUTER the tier binds. A subscriber sends the first sequence number it wants, as 8 big-endian bytes, and gets the buffered batches from it on, then a frame with sequence number -1. Without it, no replay.
- `--kv-events-buffer-steps` (default 10000): how many past batches are kept for replay.

### Start and restart

The snapshot written at a clean stop keeps each chunk's `BlockStored`. A new instance that restores a snapshot publishes nothing at start: the indexer already holds exactly the pool's chunks, because every store and eviction was published, and the stop published the last ones. The restored chunks keep their `BlockStored`, so evicting one publishes its `BlockRemoved`, as it would have before the restart.

Any other start publishes one `AllBlocksCleared`, before the tier binds its socket and so before it answers any engine. The indexer drops every entry of the topic's pod, which here is the tier alone; the engines' own entries are on their own topics and stay. It is sent because the indexer may hold entries the pool no longer has:

- a fresh start, or a crash, which leaves no snapshot;
- a snapshot that is unreadable, is of another region, or is not used because the cache policy is not LRU, and region files that are damaged;

After the clear, the indexer learns each chunk when it is stored. A chunk that is being written when the tier stops is not readable; if its store completes after the restart, its `BlockStored` is published then.

A ZMQ PUB socket drops what it sends before a subscriber connects, and `AllBlocksCleared` is sent as soon as the instance starts. A subscriber that connects later does not get it unless it asks the replay endpoint for it (`--kv-events-replay-endpoint`), starting from sequence number 0, as for any batch it missed. The tier does not wait for a subscriber or resend. Without replay, an indexer that connects after a tier restarted without a usable snapshot may keep entries for chunks the pool no longer has.

## Configuring an engine

The engine sets no pool size and no region location; both come from the tier:

```json
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "ExternalOffloadingSpec",
    "canonical_layout": true,
    "master_address": "ipc:///host-shm/external-primary-tier.sock"
  }}'
```

`master_address`, required, is the tier's ZMQ endpoint. The DaemonSet and the vLLM pods mount the host's `/dev/shm` at the same path, so the paths in the tier's `ShmFileRegion` are the ones the engines open.

Optional keys:

- `hash_function`: the engine's `--prefix-caching-hash-algo`, which vLLM does not pass to offloading specs; `sha256` by default. The tier refuses an engine whose hash function or `PYTHONHASHSEED` differs from the pool's.
- `attach_timeout_ms`: how long a process waits for its attach answer; 120000 by default, because the first attach waits while the tier creates and pre-faults the pool.
- `call_timeout_ms`: how long the scheduler waits for one lookup or prepare reply; 1000 by default. A tier that does not answer in time is treated as gone for the rest of the step.

A GPU worker that the tier refuses, or that cannot reach the tier or map its region, serves without the pool: it copies nothing, and the engine recomputes what it would have loaded. The reason is logged at warning level.

While the tier is unreachable, the scheduler misses every lookup and skips every store, and it attaches again at the end of the next step once the tier is back. A tier that stops answering between an engine's lookup and its load cannot be recovered from: vLLM gives a manager no way to fail a load, so the engine stops with an error rather than compute on KV it never loaded.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The package imports vLLM internals, including `mmap_path` on `SharedOffloadRegion`, so run the tests against a vLLM checkout that has it.
