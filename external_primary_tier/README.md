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
- `tests/`: unit tests.

## Running

```bash
external-primary-tier \
    --pool-bytes $((64 << 30)) \
    --address ipc:///host-shm/external-primary-tier.sock \
    --region-dir /host-shm \
    --cache-policy lru \
    --lease-seconds 30 \
    --fence-seconds 10
```

- `--lease-seconds`: every pin and write is released this long after it is granted, unless the engine releases it first. It is far longer than the longest copy, so a live engine's leases never reach it, and a dead engine's leases end on their own.
- `--fence-seconds`: after a clean stop, the next instance loads the snapshot and holds the pins and writes in it this long. After a crash, it starts with every chunk free and hands out no chunk this long.
- `--cache-policy`, `--cache-policy-module-path`: vLLM's cache policy for the pool, by name. A policy that is not built into vLLM is loaded from the module path, as vLLM's own `cache_policy_module_path` does. Only LRU records are snapshotted; with any other policy, the next instance starts empty.

The region's files, `external-primary-tier.mmap*`, outlive the process. The spec file beside them, `external-primary-tier.spec`, records their layout, and a new instance reuses them when they match it.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
