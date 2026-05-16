# PVC Evictor Configuration Guide

This document provides comprehensive configuration reference for the PVC Evictor.

## Environment Variables

All configuration is done through environment variables. These can be set directly in the deployment YAML or through Helm chart values.

### Core Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PVC_MOUNT_PATH` | string | `/kv-cache` | Mount path of PVC in pod |
| `CACHE_DIRECTORY` | string | `kv/model-cache/models` | Subdirectory within PVC containing cache files |
| `CLEANUP_THRESHOLD` | float | `85.0` | Disk usage % to trigger deletion |
| `TARGET_THRESHOLD` | float | `70.0` | Disk usage % to stop deletion |

### Multi-Process & Sharding Configuration

| Variable | Type | Default | Valid Values | Description |
|----------|------|---------|--------------|-------------|
| `NUM_CRAWLER_PROCESSES` | int | `8` | > 0 | Number of crawler processes per pod (P1-PN) |
| `TOTAL_SHARDS` | int | `1` | > 0 | Total number of pods/shards across the deployment |
| `SHARD_INDEX` | int | `0` | 0 to `TOTAL_SHARDS-1` | Explicit shard index for this pod (auto-derived from hostname in StatefulSets) |
| `LOGGER_INTERVAL_SECONDS` | float | `0.5` | > 0 | Activator monitoring interval (seconds) |
| `FILE_QUEUE_MAXSIZE` | int | `10000` | > 0 | Max items in queue when deletion is ON |
| `FILE_QUEUE_MIN_SIZE` | int | `1000` | > 0 | Pre-fill queue to this size when deletion is OFF |
| `DELETION_BATCH_SIZE` | int | `100` | > 0 | Files per deletion batch (deleter process) |
| `FILE_ACCESS_TIME_THRESHOLD_MINUTES` | float | `60.0` | >= 0 | Skip files accessed within this time (minutes) |

### Safety Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DRY_RUN` | bool | `false` | If true, simulate deletion without actually deleting files |
| `LOG_LEVEL` | string | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |
| `LOG_FILE_PATH` | string | `None` | Optional file path to write logs to (in addition to stdout) |

**Note:** The evictor uses the canonical FileMapper structure from `llmd_fs_backend`, which matches the vLLM offloader's file organization.

## Helm Chart Configuration

When using the Helm chart, configuration is done through `values.yaml`:

```yaml
config:
  # Threshold Configuration
  cleanupThreshold: 85.0
  targetThreshold: 70.0
  
  # Cache Configuration
  cacheDirectory: "kv/model-cache/models"
  
  # Multi-Process Configuration
  numCrawlerProcesses: 8
  # Namespace Sharding Configuration (Multi-Pod)
  # totalShards: 1
  # shardIndex: 0
  loggerIntervalSeconds: 0.5
  fileQueueMaxsize: 10000
  fileQueueMinSize: 1000
  deletionBatchSize: 100
  fileAccessTimeThresholdMinutes: 60
  
  # Safety Configuration
  dryRun: false
  logLevel: INFO
  logFilePath: "/tmp/evictor_all_logs.txt"
  
```

See [helm/values.yaml](helm/values.yaml) for complete Helm configuration options.

## Configuration Defaults

All default values are defined as constants in [config.py](config.py):

```python
DEFAULT_PVC_MOUNT_PATH = "/kv-cache"
DEFAULT_CLEANUP_THRESHOLD = 85.0
DEFAULT_TARGET_THRESHOLD = 70.0
DEFAULT_CACHE_DIRECTORY = "kv/model-cache/models"
DEFAULT_DRY_RUN = False
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_CRAWLER_PROCESSES = 8
DEFAULT_LOGGER_INTERVAL = 0.5
DEFAULT_FILE_QUEUE_MAXSIZE = 10000
DEFAULT_FILE_QUEUE_MIN_SIZE = 1000
DEFAULT_DELETION_BATCH_SIZE = 100
DEFAULT_FILE_ACCESS_TIME_THRESHOLD_MINUTES = 60.0
```

## Advanced Configuration

### Tuning Crawler Processes & Multi-Pod Sharding

The number of crawler processes and shards affects:
- **Parallelism**: More processes/pods = faster file discovery
- **Resource usage**: Each process/pod consumes CPU and memory
- **Load balancing**: Processes use hex modulo filtering to divide work

**Scalability**: The sharding engine uses `HEX_MODULO_BASE = 4096` (matching the exact 3-hex-digit directory structure `000` to `fff` in FileMapper). This allows perfectly even load distribution across any number of total crawler processes (`TOTAL_SHARDS * NUM_CRAWLER_PROCESSES`), up to 4096. Non-power-of-2 crawler counts are fully supported.

**Recommendations (starting points - tune based on your workload)**:
- **Small deployments** (< 1TB): 1 pod, 1-2 processes
- **Medium deployments** (1-10TB): 1-2 pods, 4-8 processes each
- **Large deployments** (> 10TB): 2-4 pods, 8-16 processes each

Monitor CPU usage and file discovery rates to optimize for your specific deployment.


### Tuning Queue Sizes

**FILE_QUEUE_MAXSIZE**: Maximum queue size when deletion is active
- Higher values = more memory usage, smoother deletion
- Lower values = less memory usage, more queue blocking
- Recommended: 10000 for most deployments

**FILE_QUEUE_MIN_SIZE**: Pre-fill queue size when deletion is inactive
- Allows fast deletion start when threshold is reached
- Should be < MAXSIZE (typically 10% of MAXSIZE)
- Recommended: 1000 for most deployments

### Tuning Deletion Batch Size

**DELETION_BATCH_SIZE**: Number of files deleted per batch
- Higher values = fewer system calls, faster deletion
- Lower values = more granular progress reporting
- Recommended: 100-1000 depending on file sizes

### Tuning Access Time Threshold

**FILE_ACCESS_TIME_THRESHOLD_MINUTES**: Skip recently accessed files
- Prevents deletion of "hot" cache files
- **Important**: Requires filesystem with accurate atime tracking
- **Warning**: Many filesystems use `relatime` which may not update atime on every access
- Recommended: 60 minutes for most deployments

### Dry Run Mode

Enable dry run mode for testing:
```bash
helm install pvc-evictor ./helm --set config.dryRun=true
```

In dry run mode:
- Files are discovered and queued normally
- Deletion is simulated (no actual file removal)
- All logging and metrics work as normal
- Useful for testing configuration before production deployment

## See Also

- [README.md](README.md) - Overview and features
- [QUICK_START.md](QUICK_START.md) - Deployment guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [helm/README.md](helm/README.md) - Helm chart documentation