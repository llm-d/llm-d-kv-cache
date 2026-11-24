# Upstream versions

kvpoold imports vLLM internals on both sides: the server uses vLLM's `CPUOffloadingManager`, cache policies, `SharedOffloadRegion` and KV event publisher, and the adapter subclasses vLLM's `CPUOffloadingSpec`.

## vLLM

**Required: vLLM main at or after merge commit `b93d6e4c17`** ([vllm-project/vllm#67258](https://github.com/vllm-project/vllm/pull/67258)), or a `vllm/vllm-openai:nightly` image built from it. No vLLM release has it yet; vLLM 0.32.0 does not.

What kvpoold needs from it:

| vLLM API | Used by | Why |
|---|---|---|
| `SharedOffloadRegion(..., mmap_path=...)` | server, adapter | The region's files are named by the server, not derived from an engine id. |
| `REGION_SEGMENTS` | server, adapter | Server and engines split the region into the same segment files. |
| `CPUOffloadingManager(..., enable_events=True)`, `CachePolicyFactory` | server | The pool's records and cache policy. |
| `OffloadingEventsTracker`, `self_describing_kv_events` | adapter | The `BlockStored` an engine sends with its stores. |
| `EventPublisherFactory`, `KVEventsConfig` | server | Publishing the pool's KV events. |
| `StoreTrigger` (optional) | server, adapter | Passed to the store calls of a vLLM that has it; a vLLM without it is also supported. |

The server and every engine on a node must run the same vLLM, since both sides depend on the same segment split and chunk layout.

Check an install:

```bash
python3 -c 'import inspect
from vllm.v1.kv_offload.cpu.shared_offload_region import REGION_SEGMENTS, SharedOffloadRegion
assert "mmap_path" in inspect.signature(SharedOffloadRegion.__init__).parameters
import kvpoold.server.app, kvpoold.adapters.vllm.spec; print("ok")'
```
