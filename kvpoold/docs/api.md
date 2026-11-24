# API, version 1

`kvpoold.api.v1` defines the messages between engine processes and the server. `PROTOCOL_VERSION` is 1; the server refuses an attach of any other version.

## Transport

Each engine process has a ZMQ DEALER connected to the server's ROUTER. A message is an empty delimiter frame and one msgpack payload. Every message is a tagged msgspec struct encoded as an array, so the decoder maps the tag on the wire (the class name) to its class. Each side decodes with a `Decoder` of the messages it receives: `ToServer` on the server, `FromServer` on the engine.

- **Requests** get exactly one reply. A request carries a `call_id` that its reply repeats, so a reply that arrives after its request timed out is not taken for the reply to a later one.
- **Notices** get no reply.
- Any request may be answered with `Draining` (the server is stopping) or `UnknownClient` (the sender has not attached to this instance) instead of its usual reply.

An `OffloadKey` is vLLM's: the block hash followed by the KV cache group index, as bytes.

## Requests, from engine processes

| Message | Fields | Reply |
|---|---|---|
| `Attach` | `version`, `client_id` (vLLM's engine id), `rank` (the GPU worker's rank; None for the scheduler), `compat` (`CompatDescriptor`) | `AttachOk` with the region, or `Refused` |
| `Lookup` | `req_id`, `key` | `LookupOk` with `result`: `miss`, `hit` or `hit_pending`. A hit is pinned for the engine. |
| `PrepareLoad` | `req_id`, `keys` this step got as hits, `positions` | `LoadSpec` with the keys' `chunk_ids`, in order; `Refused` if the engine holds no pin on a key |
| `PrepareStore` | `req_id`, `keys`, `positions` | `StoreSpec` with `keys_to_store` and their `chunk_ids`; `NoSpace` when the pool cannot evict enough or is fenced |
| `GetStats` | | `Stats` with the pool's counters and the engine's pins and writes |

`positions` are each key's end-token position in the request, or None when it has none; vLLM's cache policies order a request's keys by it.

## Notices, from engine processes

| Message | Fields | Meaning |
|---|---|---|
| `CompleteLoad` | `keys` | The loads are done; their pins are released. |
| `CompleteStore` | `keys`, `success`, `events` | The stores are done. On success the chunks become readable. `events` is one vLLM `BlockStored` per key for the server to publish, or None. |
| `RequestFinished` | `req_id` | vLLM's `on_request_finished`: the cache policy applies the request's accesses. |
| `ReleaseHits` | `hits`: `(key, count)` pairs | The step's hits that were not loaded; a key hit by several requests is released once per hit. |
| `Leave` | | The engine is stopping and its copies have finished; its leases are released at once. |

## `CompatDescriptor`

| Field | Meaning |
|---|---|
| `layout` | `kv_bytes_per_chunk`: the bytes of one chunk across every rank of the engine |
| `namespace` | 8 bytes naming whose keys are interchangeable; prefixes every pool key |
| `hash_function` | the engine's prefix-caching hash algorithm, e.g. `sha256` |
| `hash_seed` | the engine's `PYTHONHASHSEED`, or None |

See [architecture.md](architecture.md#layout-and-namespace) for what the server checks.

## `ShmFileRegion`

The region, in `AttachOk`: `region_id`, `num_chunks`, `kv_bytes_per_chunk` (the row stride), and `segments`, one `(path, offset, length)` per segment file in region order. A worker checks that each file exists with its length before it maps it.

## One engine's messages

At start, each GPU worker sends `Attach` with its rank, maps the region from the reply, and closes its connection. Then the scheduler sends `Attach` with no rank; the server refuses it until a worker of the engine has attached.

Each scheduler step:

```
engine                                   server
  Lookup(req, key) ─────────────────────▶  pins a hit
  ◀──────────────────────────────── LookupOk(hit)
  PrepareLoad(req, hit keys) ───────────▶  records the access
  ◀──────────────────────────────── LoadSpec(chunk ids)
  PrepareStore(req, keys) ──────────────▶  allocates, evicting if needed
  ◀──────────────────────────────── StoreSpec(keys to store, chunk ids)
  ReleaseHits(unloaded hits) ───────────▶  releases their pins      (end of step)

  ... GPU workers copy to and from the region ...

  CompleteLoad(keys) ───────────────────▶  releases the loads' pins
  CompleteStore(keys, success, events) ─▶  chunks become readable; events published
  RequestFinished(req) ─────────────────▶  cache policy applies the request's accesses
```

`Lookup`, `PrepareLoad` and `PrepareStore` block the step for their reply, so the scheduler waits at most `call_timeout_ms` for each; a later answer detaches it. At shutdown, after its workers have stopped, the scheduler sends `Leave`.
