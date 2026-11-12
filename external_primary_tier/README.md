# External Primary Tier

One CPU KV-cache pool per node, shared by every vLLM engine on it.

A per-node process, the External Primary Tier, owns the pool's records. Each engine selects `ExternalOffloadingSpec` in `--kv-transfer-config`; its offloading manager, `ExternalPrimaryTierClient`, sends every call that reads or changes the pool to the External Primary Tier, and its GPU workers copy KV data directly to and from the shared region.

The design is in the RFC, [llm-d/llm-d#3047](https://github.com/llm-d/llm-d/issues/3047).

## Layout

- `external_primary_tier/protocol/`: the messages between engines and the External Primary Tier, the region descriptor, and the codec.
- `external_primary_tier/tier/`: the External Primary Tier.
- `tests/`: unit tests.

## Running

```bash
external-primary-tier \
    --pool-bytes $((64 << 30)) \
    --address ipc:///host-shm/external-primary-tier.sock \
    --region-dir /host-shm
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
