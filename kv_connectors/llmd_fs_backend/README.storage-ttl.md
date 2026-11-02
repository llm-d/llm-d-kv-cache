# Fixed storage TTL (experimental implementation)

This opt-in feature gives each stored generation a fixed maximum lifetime. It is
separate from PVC pressure eviction and its minimum idle-age protection. Reads
and duplicate stores do not extend the deadline. Expired lookups miss; inference
can recompute. Already accepted storage reads may finish. Removing the pathname
and returning allocated space are asynchronous; an open reader can delay frees.

The initial implementation supports Linux procfs, the native POSIX CPU-staging
backend and one worker rank. GDS, object storage and multi-rank TTL configurations
are rejected. Native correctness, HTTP transport and concurrent serving/cleanup have been
validated on GPFS. These finite experiments do not establish a universal latency
budget or production-traffic benefit. This draft does not enable TTL by default.

Configure the existing OffloadingConnector with:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SharedStorageOffloadingSpec",
    "spec_module_path": "llmd_fs_backend.spec",
    "shared_storage_path": "/kv-cache",
    "track_file_access": true,
    "storage_ttl_seconds": 3600
  }
}
```

The connector uses a separate `.llmd-ttl-v1-3600` namespace under the configured
root. Changing the configured duration creates a separate policy namespace;
legacy files are not retroactively reinterpreted. The lifetime origin is the
immutable file mtime established immediately before publication. Accurate access
tracking updates atime and preserves that mtime. A successful new publication
after expiry creates a new generation and deadline; TTL does not prohibit future
recaching of the same content.

A request may confirm the namespace policy through vLLM's existing transport:

```json
{"kv_transfer_params": {"llmd_storage_ttl_seconds": 3600}}
```

This first version does not merge arbitrary per-request lifetimes for a shared
prefix. A conflicting or invalid hint makes that request decline storage reuse
and persistence, with a warning; it does not throw inside the inference
scheduler. GPU cache reuse and inference continue according to vLLM's normal
behavior. Frontend HTTP validation is outside this backend feature. With no
configured TTL and no TTL hint, existing behavior remains the default.

Run the updated PVC evictor against the parent storage root with
`ENABLE_STORAGE_TTL=true`. It discovers expired namespace files and reclaims them
even when pressure eviction is off. Unexpired files are not removed below the
pressure threshold. Pressure removal of TTL files uses the same mutation
coordination. All writers/deleters touching these policy namespaces must use the
updated protocol; do not run an old uncoordinated deleter against them.

Readers pin an inode before accepting a hit. Its same-container native worker
uses `/proc/<scheduler-pid>/fd/<fd>` while the scheduler retains that descriptor.
Request completion releases speculative pins; accepted reads retain their pins
until completion, including late callbacks. This keeps data readable across
unlink or replacement without shared read locks across a distributed transfer.
Cache reset is refused while accepted TTL loads are in flight because this vLLM
API discards old callbacks before acknowledging the worker flush; drain loads
before reset.

Native publishers and deleters serialize pathname mutations with stable bucket
POSIX record locks. Native writers additionally serialize threads locally, since
record locks are process-owned. Lock identities are not removed by directory
cleanup. The GPFS test mount passed cross-node POSIX exclusion/release; its
`flock` probe did not. Other deployments need equivalent locking semantics and
shared writable lock-file permissions. Both local and GPFS native/end-to-end gates were exercised. Revalidate locking
and descriptor permissions for a different deployment.

Storage routing events remain best-effort hints, with the backend lookup
providing authoritative expiry. This does not invalidate GPU/CPU-resident copies
or promise globally ordered events across storage generations.

## Optional per-request lifetime allowlist

Fixed `storage_ttl_seconds` behavior remains the default. To let a caller choose
among explicit server policies, configure the existing spec with both a default
and an allowlist:

```json
{
  "spec_name": "SharedStorageOffloadingSpec",
  "spec_module_path": "llmd_fs_backend.spec",
  "shared_storage_path": "/cache",
  "storage_ttl_seconds": 60,
  "storage_ttl_allowlist": [30, 60, 90]
}
```

A caller supplies `kv_transfer_params.llmd_storage_ttl_seconds`. Missing hints use
the configured default. A supplied value outside the allowlist (including null,
booleans or strings) declines storage reuse and persistence for that request;
serving can still compute normally. Lifetimes remain positive integer seconds.
The default must belong to the allowlist. This option supports one worker rank,
native POSIX CPU staging, and disabled storage events. Policy-blind events cannot
advertise a lifetime-specific namespace safely, so enabling them is rejected.

`spec.py` copies trusted `FileMapper` instances into separate `.llmd-ttl-v1-N`
namespaces. `RequestStorageTTLManager` binds one route generation to the
scheduler-owned `ReqContext`; it delegates to the selected fixed-TTL manager.
Internal `SharedStorageLoadStoreSpec.source_paths` aliases carry store targets
or accepted read FDs through the existing worker transfer path. Clients cannot
supply these paths. Neither a late callback nor a changed request-parameter dict
can select another request's namespace. The route survives request finish until
all accepted load and store completions drain. A stale context cannot complete
work for a later request that reuses its ID.

Identical prefixes under two lifetimes have separate physical copies. A shorter
owner cannot shorten a longer copy; reading or an unexpired duplicate store does
not renew its publication deadline. This is isolation with duplication, not a
shared-file maximum-deadline ownership algorithm.

The PVC evictor still discovers files, reads the lifetime from each namespace,
rechecks generation metadata under the stable lock, and unlinks eligible files.
Set `ENABLE_STORAGE_TTL=true` and include the **common root** `/cache` in its
`CACHE_DIRECTORY` so it can see every allowed namespace. This port adds no private
quota, LRU policy or inline expiry deleter. Physical capacity feedback and cleanup
remain the evictor's responsibility. Earlier research utility comparisons used a
separate conservative private authority and must not be treated as performance
measurements of this integration. GPU/CPU-resident cache lifetimes remain outside
this storage policy. No automatic llm-d policy injector or TTL predictor is added.

## Scope of this draft

This branch adds TTL only on top of the recency/backpressure foundation in
PR #741. It contains no deletion work-rate or file-attempt ceilings, adaptive
capacity controller, or benchmark-only C++ batching helper. TTL remains opt-in.
