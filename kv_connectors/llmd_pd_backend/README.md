# llmd-pd-backend: P/D NIXL Secondary Tier for vLLM

> **Status:** Out-of-tree secondary tier plugin for vLLM's `TieringOffloadingSpec`.
> Implements KV-cache block transfer between prefill (P) and decode (D)
> instances over NIXL/UCX.

## Overview

`llmd_pd_backend` implements a `SecondaryTierManager` that enables
disaggregated prefill/decode serving by transferring KV-cache blocks
between instances over NIXL. It plugs into vLLM's multi-tier offloading
system as an out-of-tree secondary tier.

**Data flow:**
```
GPU ↔ CPU Primary Tier (host DRAM) ↔ P/D Secondary Tier (NIXL/UCX)
                                          ↕
                                    Remote Instance
```

## Configuration

Add the P/D tier to your vLLM `kv_connector_extra_config`:

```json
{
  "kv_connector": "TieringOffloadingConnector",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 10737418240,
    "secondary_tiers": [
      {
        "type": "pd",
        "module_path": "llmd_pd_backend.manager",
        "host": "10.0.0.1",
        "port": 5710,
        "connect_timeout_s": 10.0,
        "transfer_timeout_s": 30.0,
        "max_retries": 3,
        "retry_backoff_s": 1.0
      }
    ]
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `host` | str | **required** | Routable IP of the NIXL endpoint |
| `port` | int | **required** | Port for NIXL connections |
| `connect_timeout_s` | float | 10.0 | Timeout for establishing peer connections |
| `transfer_timeout_s` | float | 30.0 | Timeout for individual block transfers |
| `max_retries` | int | 3 | Maximum connection retry attempts |
| `retry_backoff_s` | float | 1.0 | Base backoff between retries (exponential) |

## Installation

```bash
pip install -e kv_connectors/llmd_pd_backend
```

Requires:
- Python ≥ 3.12
- `nixl` (NVIDIA Inference Xfer Library)
- vLLM ≥ 0.23.0 (for the `TieringOffloadingSpec` interface)

## Architecture

### Prefill Side (Producer)
1. vLLM computes KV blocks and offloads them to the CPU primary tier
2. `submit_store()` registers the blocks as available for remote pull
3. NIXL agent exposes the registered memory region

### Decode Side (Consumer)
1. `lookup()` checks if blocks are available on the remote prefill peer
2. `submit_load()` initiates NIXL READ transfers from the prefill's memory
3. `get_finished_jobs()` polls for transfer completion

### Failure Handling
- **Connection failure:** Bounded retry with exponential backoff
- **Transfer timeout:** Configurable per-transfer deadline
- **Eviction race:** `evict()` removes keys atomically; concurrent lookups
  immediately return MISS
- **Shutdown:** `shutdown()` releases all NIXL handles and deregisters memory

## Testing

```bash
# Unit tests (no nixl/vllm required)
cd kv_connectors/llmd_pd_backend
pytest tests/test_config.py tests/test_manager.py -v

# Integration tests (requires nixl)
pytest tests/test_integration.py -v -m integration
```

## Relationship to Other Components

| Component | Scope |
|:---|:---|
| **This backend** | P/D KV handoff via NIXL (TieringOffloadingSpec secondary tier) |
| `llmd_nixl` (issue #409) | GPU ↔ S3 object storage offload |
| vLLM `NixlConnector` | Direct P/D disaggregation (KVConnectorBase, not tiering) |
| vLLM `P2PSecondaryTierManager` | Symmetric P2P cache sharing between arbitrary instances |

## References

- [llm-d v0.8.0 Roadmap (issue #519)](https://github.com/llm-d/llm-d-kv-cache/issues/519)
- [vLLM KV Offloading Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage.html)
- [NIXL Documentation](https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html)
