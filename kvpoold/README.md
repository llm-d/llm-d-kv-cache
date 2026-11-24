# kvpoold

One CPU KV-cache pool per node, shared by every vLLM engine on it.

`kvpoold` is a per-node server that owns the pool's records. Each engine selects `PoolOffloadingSpec` in `--kv-transfer-config`. Its offloading manager sends every call that reads or changes the pool to `kvpoold`, and its GPU workers copy KV data directly to and from a shared region in the host's `/dev/shm`. An engine's stored chunks are hits for every other engine on the node that runs the same model with the same layout.

The design is in the RFC, [llm-d/llm-d#3047](https://github.com/llm-d/llm-d/issues/3047).

## Quick start

Run the server, on a node where it and the engines mount the host's `/dev/shm` at `/host-shm`:

```bash
kvpoold \
    --pool-bytes $((64 << 30)) \
    --address ipc:///host-shm/kvpoold.sock \
    --region-dir /host-shm \
    --kv-events-endpoint 'tcp://*:5557' \
    --kv-events-topic 'kv@${NODE_NAME}@<model>'
```

Point each engine at it:

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "PoolOffloadingSpec",
    "canonical_layout": true,
    "pool_address": "ipc:///host-shm/kvpoold.sock"
  }}'
```

The engine sets no pool size and no region location; both come from the server. Every flag and key is in [docs/deployment.md](docs/deployment.md).

## Docs

- [docs/architecture.md](docs/architecture.md): the pool and its region, layout and namespace, pins and leases, start and stop, KV events.
- [docs/api.md](docs/api.md): the messages between engines and the server, and the messages of one engine step.
- [docs/deployment.md](docs/deployment.md): running `kvpoold`, and configuring an engine.
- [docs/upstream-versions.md](docs/upstream-versions.md): the vLLM that kvpoold needs.

## Layout

- `kvpoold/api/v1/`: the wire protocol: `messages.py`, the region descriptor in `region.py`, and the `CompatDescriptor` in `compat.py`.
- `kvpoold/client/`: one engine process's side of the protocol, independent of the engine: `Client`, over the DEALER transport in `session.py`.
- `kvpoold/adapters/vllm/`: the vLLM adapter: `PoolOffloadingSpec` (`spec.py`), the `PoolOffloadingManager` over `Client` (`manager.py`), the engine's `CompatDescriptor` (`compat.py`), the KV events an engine sends with its stores (`events.py`), and the `vllm.general_plugins` registration (`plugin.py`).
- `kvpoold/server/`: the server: the CLI and `ServerConfig`, the request loop (`app.py`), the records (`pool.py`), the engines' state (`clients.py`), the region (`region.py`), and the spec and snapshot files (`snapshot.py`).
- `tests/`: unit tests, by package.

## Limits

- One model per server: the pool's KV events go out on one topic.
- Only LRU records are snapshotted. With any other cache policy, a restarted server starts with every chunk free.
- A server that stops answering between an engine's lookup and its load cannot be recovered from: vLLM gives a manager no way to fail a load, so the engine stops with an error rather than compute on KV it never loaded.
- `kvpoold.api.v1` imports vLLM: `CompleteStore` carries vLLM's `BlockStored`.
- The whole pool uses one block hash function and seed.

## Development

```bash
pip install -e ".[dev]"
make test lint
```

The package imports vLLM internals that only recent vLLM has (see [docs/upstream-versions.md](docs/upstream-versions.md)), so run the tests against such a vLLM.
