# PVC Evictor

Automatic disk space management for vLLM KV-cache storage on Kubernetes PVCs.

## Overview

The PVC Evictor is a multi-process Kubernetes deployment designed to automatically manage disk space on PVCs used for vLLM KV-cache storage offloading. It monitors PVC disk usage and automatically deletes old cache files when configured thresholds are exceeded, enabling continuous vLLM operation while resolving storage capacity exhaustion without manual intervention.

## Quick Start

The chart defaults to the latest PVC Evictor image:

```text
quay.io/pvc-evictor/pvc-evictor:latest
```

For reproducible llm-d v0.x deployments, pin `image.tag` to `llm-d-v0.x`.

```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-vllm-cache \
  --set securityContext.pod.fsGroup=1000960000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c31,c15" \
  --set securityContext.container.runAsUser=1000960000
```

See [QUICK_START.md](QUICK_START.md) for detailed deployment instructions.

## Architecture

The evictor uses an **N+2 process architecture** where N parallel crawler processes discover cache files, while two dedicated processes (activator and deleter) coordinate and execute the deletion workflow.

### Architecture Diagram

```mermaid
graph TB
    subgraph "PVC Evictor Pod"
        Main[Main Process<br/>evictor.py]
        
        subgraph "Crawler Processes"
            C1[Crawler 1<br/>hex range: 0-1]
            ...
            CN[Crawler N<br/>hex range: E-F]
        end
        
        Act[Activator Process<br/>monitors disk usage]
        Del[Deleter Process<br/>batch deletion]
        
        Queue[multiprocessing.Queue<br/>file paths FIFO]
        DelEvent[deletion_event<br/>Event flag]
    end
    
    PVC[PVC Mount<br/>/kv-cache]
    
    Main -->|spawns & monitors| C1
    Main -->|spawns & monitors| CN
    Main -->|spawns & monitors| Act
    Main -->|spawns & monitors| Del
    
    C1 -..->|scan files| PVC
    CN -..->|scan files| PVC
    
    C1 -->|put file paths| Queue
    CN -->|put file paths| Queue
    
    Act -..->|check usage| PVC
    Act -->|set/clear| DelEvent
    
    Del -->|get file paths| Queue
    Del -->|check flag| DelEvent
    Del -..->|delete files| PVC
```

### Process Roles

- **N Crawler Processes** - Discover and queue files for deletion (N configurable: 1, 2, 4, 8, or 16, default: 8)
- **Activator Process** - Monitors disk usage and controls deletion triggers
- **Deleter Process** - Performs batch file deletions
- **Main Process** - Spawns and monitors all child processes, aggregates logging

### Inter-Process Communication

- **multiprocessing.Queue** - FIFO queue for file paths (Crawlers → Deleter)
- **multiprocessing.Event** - Boolean flags for coordination:
  - `deletion_event` - Activator controls Deleter (ON when usage >= cleanup threshold)
  - `shutdown_event` - Main signals graceful shutdown to all processes

### Hot/Cold Cache Strategy

Files are classified as hot/cold based on access time (`st_atime`):
- **Hot files** - Accessed within threshold (default: 60 minutes) - **Protected from deletion**
- **Cold files** - Not accessed recently - **Eligible for deletion**

## Key Features

- **Automatic Threshold-Based Deletion** - Triggers at 85% usage, stops at 70% (configurable)
- **Hot Cache Protection** - Skips recently accessed files based on access time
- **Parallel File Discovery** - Configurable crawler processes (1-16) for multi-TB volumes
- **Batch Deletion** - Efficient deletion using `xargs rm -f`
- **Streaming Architecture** - No memory accumulation, works with multi-TB storage
- **Flat cache layout** - Path-based discovery matching llmd_fs_backend v0.20+ on-disk layout
- **Aggregated Logging** - Unified system status every 30 seconds

### Threshold Behavior

**Soft Thresholds (Current Implementation):**
- Deletion triggers at 85% usage (configurable)
- Deletion stops at 70% usage (configurable)
- Cold files (not accessed within threshold) are queued for deletion in discovery order

**What Happens if PVC Fills Completely:**
If the PVC reaches 100% before deletion frees space, vLLM cache writes will fail and new requests cannot offload to disk. The soft threshold design (85% trigger, 70% stop) maintains a safety buffer to prevent this. See [issue #218](https://github.com/llm-d/llm-d-kv-cache/issues/218) for future optimizations.

### Important Considerations

**Filesystem atime Tracking:**
Most filesystems use `relatime` (relative atime) which only updates access time if the file wasn't accessed in the past 24 hours. This means recently accessed files may appear "cold" and be deleted if they were last accessed more than 24 hours ago, even if accessed multiple times since then.

**Disk Usage Calculation:**
The evictor uses `statvfs()` for performance instead of the more accurate `du` scan. This provides real-time usage percentages but may differ slightly from `du` output due to filesystem metadata overhead and block allocation differences.

## Configuration

Key settings (see [CONFIGURATION.md](CONFIGURATION.md) for complete reference):

| Setting | Default | Description |
|---------|---------|-------------|
| `cleanupThreshold` | 85.0 | Disk usage % to trigger deletion |
| `targetThreshold` | 70.0 | Disk usage % to stop deletion |
| `numCrawlerProcesses` | 8 | Parallel file discovery (1, 2, 4, 8, or 16) |
| `cacheDirectory` | `kv/model-cache/models` | Cache path relative to PVC mount |
| `fileAccessTimeThresholdMinutes` | 60 | Protect files accessed within N minutes |

## Monitoring

### Log Patterns

**Deletion Events:**
```
DELETION_START:1234567890.123,85.42
DELETION_END:1234567890.456,69.87
```

**Aggregated System Status (every 30 seconds):**
```
=== System Status ===
Crawlers: 8 active
  Total files discovered: 50000
  Total files queued: 10000
  Total files skipped (hot): 5000
Activator P9:
  PVC Usage: 72.3% (144.60GB / 200.00GB)
  Deletion: OFF
  Thresholds: cleanup=85%, target=70%
Deleter P10:
  Files deleted: 15000
  Space freed: 30.50GB
=====================
```

### Monitoring Commands

```bash
# Watch logs
kubectl logs -f deployment/pvc-evictor-pvc-evictor

# Watch deletion events
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

## On-disk cache layout

Crawlers discover `.bin` files under the flat layout used by llmd_fs_backend (v0.20+). The crawler does not import vLLM or FileMapper. The image still bundles `llmd_fs_backend` for optional storage events when `STORAGE_EVENTS_ENDPOINT` is set (see #605).

```
<CACHE_DIRECTORY>/<model>_<digest>_r<rank>/
  <hhh>/<hh>_g<group>/<block-hash>.bin
```

Base fingerprint dirs (`<model>_<digest>/` with `config.json` only) are not scanned for `.bin` files.

## Development and tests

```bash
cd kv_connectors/pvc_evictor
pip install -r requirements-dev.txt
make test
```

Build image (from `kv_connectors/`):

```bash
make docker-build
```

By default, this builds `quay.io/pvc-evictor/pvc-evictor:latest`.
Override the target when testing a local or development tag:

```bash
make docker-build IMAGE_REPOSITORY=quay.io/pvc-evictor/pvc-evictor IMAGE_TAG=llm-d-v0.8
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Deployment guide
- **[CONFIGURATION.md](CONFIGURATION.md)** - Complete configuration reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and design
- **[helm/README.md](helm/README.md)** - Helm chart documentation

## Requirements

- Kubernetes 1.19+ or OpenShift 4.6+
- Helm 3.0+ (for Helm deployment)
- Bounded PVC
- Security context values (fsGroup, seLinuxOptions, runAsUser)
