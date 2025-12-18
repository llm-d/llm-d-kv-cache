# PVC Evictor

## Overview

The PVC Evictor is multi-process Kubernetes deployment designed to automatically manage disk space on PVCs used for vLLM KV cache offloading. It monitors disk usage and automatically deletes old cache files when storage thresholds are exceeded, ensuring continuous operation of vLLM workloads without manual intervention.

## Architecture

### N+2 Process Design

- **P1-PN: Crawler Processes** - Discover and queue files for deletion (N configurable: 1, 2, 4, 8, or 16, default: 8)
- **P(N+1): Activator Process** - Monitors disk usage and controls deletion triggers
- **P(N+2): Deleter Process** - Performs batch file deletions

**Total Processes:** N + 2 (e.g., 8 crawlers + 1 activator + 1 deleter = 10 processes for default configuration)

**Inter Process Communication:** Processes communicate via `multiprocessing.Queue` (FIFO) and `multiprocessing.Event` (boolean flags). The queue uses pipes with automatic pickling/unpickling for data transfer. Each file path is pickled once on `put()` and unpickled once on `get()`, with minimal overhead compared to file I/O. Events (`deletion_event`, `shutdown_event`) use shared memory for efficient synchronization. This design provides process isolation (separate memory spaces, bypassing Python's GIL) while enabling efficient coordination.


**Hot/Cold Cache Eviction:** The evictor implements a cache management strategy that distinguishes between hot (actively used) and cold (inactive) cache files. It protects hot cache by checking file access time (`st_atime`) and skipping files accessed within the configured threshold (`FILE_ACCESS_TIME_THRESHOLD_MINUTES`, default: 60 minutes). Only cold cache files - those not accessed recently - are queued for deletion when disk usage exceeds the cleanup threshold. This ensures active cache entries remain available for vLLM workloads while automatically freeing space from unused cache files.

## Files

### `deploy.sh`

Bash deployment script that automates the deployment process with command-line arguments.

**Usage:**
```bash
./deploy.sh <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>]
```

**Arguments:**
- `<pvc-name>` - **Required**: Name of the PVC to manage
- `--namespace=<namespace>` - **Optional**: Kubernetes namespace (auto-detected from `kubectl config` context if not provided)
- `--fsgroup=<fsgroup>` - **Optional but Recommended**: Filesystem group ID. Auto-detected from existing pods/deployments in the namespace if not provided. **Note**: These values are namespace-specific. If auto-detection fails, you must provide the correct values for your namespace or the pod may fail to start.
- `--selinux-level=<level>` - **Optional but Recommended**: SELinux security level. Auto-detected from existing pods/deployments in the namespace if not provided. **Note**: These values are namespace-specific. If auto-detection fails, you must provide the correct values for your namespace or the pod may fail to start.
- `--runasuser=<user>` - **Optional but Recommended**: User ID to run containers as. Auto-detected from existing pods/deployments in the namespace if not provided. **Note**: These values are namespace-specific. If auto-detection fails, you must provide the correct values for your namespace or the pod may fail to start.
- `--num-crawlers=<n>` - **Optional**: Number of crawler processes (default: `8`, valid: 1, 2, 4, 8, 16)
- `--cleanup-threshold=<%>` - **Optional**: Disk usage % to trigger deletion (default: `85.0`)
- `--target-threshold=<%>` - **Optional**: Disk usage % to stop deletion (default: `70.0`)

**Features:**
- Substitutes all configuration values in the deployment YAML
- Auto-detects namespace from current `kubectl config` context and security context values from existing pods/deployments in the namespace
- Note: All arguments must use `--arg=value` format (e.g., `--namespace=e5`)

**See [QUICK_START.md](QUICK_START.md) for detailed usage and examples.**

### `deployment_evictor.yaml`

Kubernetes Deployment yaml for the evictor pod.

**Key Components:**
- Deployment with single replica (N+2 processes run inside one pod, where N is NUM_CRAWLER_PROCESSES)
- Uses Docker image: `ghcr.io/guygir/pvc-evictor:latest`
- Environment variables for all configuration
- Volume mount for PVC
- Health checks (liveness/readiness probes)

**Docker Image:**
The evictor runs from a Docker image (`ghcr.io/guygir/pvc-evictor:latest`) containing all Python modules. The image is built and maintained by the project maintainers. Users do not need to build the image - it's already available in the registry.



**Security Context (Namespace-Specific):**

Security context values (`fsGroup`, `seLinuxOptions.level`, `runAsUser`) are **namespace-specific** due to Security Context Constraints (SCCs) in OpenShift. Each namespace has unique SCC requirements that must be matched exactly, or the pod will fail to start.

**Auto-Detection:**
The `deploy.sh` script automatically detects these values from existing pods/deployments in the namespace. If auto-detection fails, you must provide them explicitly via command-line arguments. Without valid values, the pod will fail to start.

**Finding Your Namespace's Values:**
```bash
# Check an existing working pod in your namespace
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

**Security Context Fields:**
- `fsGroup`: Filesystem group ID - determines group ownership of volumes mounted to the pod
- `seLinuxOptions.level`: SELinux security level - namespace-specific security labeling
- `runAsUser`: User ID to run containers as - must match namespace SCC


**Recommended:** Use the `deploy.sh` script (see [QUICK_START.md](QUICK_START.md)) which automatically detects security context values and sets all configuration via command-line arguments.

### Python Module Structure

The codebase is organized into modular Python files:

- **`config.py`** - Configuration management (`Config` dataclass, `Config.from_env()`)
- **`evictor.py`** - Main entry point (`PVCEvictor` class, `main()` function)
- **`utils/system.py`** - System utilities (`DiskUsage` dataclass, `get_disk_usage_from_statvfs()`, `setup_logging()`)
- **`processes/crawler.py`** - Crawler process and helpers (`crawler_process()`, `stream_cache_files()`, `get_hex_modulo_ranges()`, etc.)
- **`processes/activator.py`** - Activator process (`activator_process()` - monitors disk usage and controls deletion)
- **`processes/deleter.py`** - Deleter process (`deleter_process()`, `delete_batch()`)

**Key Functions:**

#### `Config.from_env()`
Loads configuration from environment variables. Returns a `Config` dataclass with all settings. Uses environment variables (standard Kubernetes practice) instead of CLI arguments for seamless integration with ConfigMaps/Secrets.

#### `get_disk_usage_from_statvfs(mount_path: str)`
Gets disk usage using `os.statvfs()` - O(1) operation, critical for multi-TB volumes. Trade-off: less accurate than `du` (which would be O(n) and could take hours), but provides instant statistics.

#### `stream_cache_files(cache_path: Path, hex_modulo_range: Tuple[int, int])`
Generator function that streams cache files using `os.scandir()`. Filters files by hex folder modulo to partition work across crawlers. After finding `hash1` folder, uses `os.walk()` to recursively find all `.bin` files below it. Yields `Path` objects for `.bin` files.

**Cache Structure:**
```
/kv-cache/kv/model-cache/models/{model}/[optional {path}/]tp_{N}/rank_{M}/{X}/{hash1}/.../*.bin
```
where:
- `{model}` is the model name (e.g., `llama3-70b`)
- `{path}` is optional - may be present or missing
- `tp_{N}` is tensor parallelism directory (e.g., `tp_1`)
- `rank_{M}` is rank directory (e.g., `rank_0`)
- `{X}` can be any folder name (e.g., `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`, etc.)
- `{hash1}` is a hex folder (length may vary, e.g., `0ae`, `abc12345`) - used for partitioning and should be backwards compatible with 0.0.5 offloader.
- `...` represents any subdirectory structure below `{hash1}` (flexible, handles any depth/pattern)
- `*.bin` files are found recursively below `{hash1}` using `os.walk()`

**Hex Partitioning:**
- Each crawler (P1-PN) handles files where `hex_folder_name % 16` is in its assigned range
- Valid crawler counts: 1, 2, 4, 8, 16 (must evenly divide 16)
- Examples:
  - 1 crawler: handles %16 in [0, 15] (all values)
  - 2 crawlers: P1 handles %16 in [0, 7], P2 handles [8, 15]
  - 4 crawlers: P1 handles [0, 3], P2 handles [4, 7], P3 handles [8, 11], P4 handles [12, 15]
  - 8 crawlers: P1 handles [0, 1], P2 handles [2, 3], ..., P8 handles [14, 15]
  - 16 crawlers: Each handles one value [0], [1], ..., [15]
- Ensures (~) even distribution without coordination overhead

#### `crawler_process(process_id, hex_modulo_range, cache_path, config_dict, deletion_event, file_queue, shutdown_event)`
Main crawler process logic (P1-PN, where N is NUM_CRAWLER_PROCESSES).

**Queueing Strategy:**
- **When deletion is OFF**: Queue files until queue size >= `FILE_QUEUE_MIN_SIZE` (pre-fill for fast start)
- **When deletion is ON**: Queue files until queue size >= `FILE_QUEUE_MAXSIZE` (maximize throughput)

**Hot Cache Protection:**
- Checks `file_path.stat().st_atime` (last access time)
- Skips files accessed within `FILE_ACCESS_TIME_THRESHOLD_MINUTES`
- Prevents deletion of active cache entries

**Backpressure:**
- When queue is full, crawler sleeps 0.1s to avoid overwhelming the queue
- Continues discovering files even when queue is full (for monitoring)

#### `activator_process(mount_path, cleanup_threshold, target_threshold, logger_interval, deletion_event, shutdown_event)`
Activator process (P(N+1)) that monitors disk usage and controls deletion triggers.

**Monitoring:**
- Uses `get_disk_usage_from_statvfs()` every `logger_interval` seconds
- Logs usage percentage

**Deletion Control:**
- Sets `deletion_event` when usage >= `cleanup_threshold`
- Clears `deletion_event` when usage <= `target_threshold`
- Logs `DELETION_START` and `DELETION_END` events

#### `deleter_process(cache_path, config_dict, deletion_event, file_queue, result_queue, shutdown_event)`
Deleter process (P10) that performs actual file deletions.

**Deletion Logic:**
- Only processes files when `deletion_event.is_set()`
- Reads file paths from `file_queue`
- Batches files up to `deletion_batch_size`
- Calls `delete_batch()` for each batch

**Batch Deletion:**
- Uses `xargs -0 rm -f` for fastest deletion
- If `xargs` fails, logs error and skips batch (files will be retried in next cycle)
- Tracks files deleted and bytes freed

#### `delete_batch(file_paths: List[str], dry_run: bool, logger)`
Deletes a batch of files using `xargs rm -f`.

**Process:**
1. Validates paths and calculates total size
2. Uses null-terminated input (`\0`) for `xargs -0`
3. Runs `subprocess.run(['xargs', '-0', 'rm', '-f'], input=input_data)`
4. On failure: logs error and returns (0, 0) - files will be retried in next cycle

#### `PVCEvictor`
Main coordinator class that spawns and manages all processes.

**Initialization:**
- Waits for PVC mount to be ready (max 60s)
- Sets up signal handlers (SIGTERM, SIGINT)
- Creates shared IPC objects:
  - `deletion_event`: Multiprocessing Event
  - `file_queue`: Multiprocessing Queue
  - `result_queue`: Multiprocessing Queue
  - `shutdown_event`: Multiprocessing Event

**Process Management:**
- Spawns P1-PN crawlers with assigned hex ranges (N = NUM_CRAWLER_PROCESSES)
- Spawns P(N+1) activator
- Spawns P(N+2) deleter
- Monitors process health and restarts on failure

## Configuration

All configuration is done via environment variables in the deployment YAML.

### Required Configuration

#### `PVC_MOUNT_PATH`
- **Default**: `/kv-cache`
- **Description**: Mount path of the PVC in the pod

#### `CACHE_DIRECTORY`
- **Default**: `kv/model-cache/models`
- **Description**: Subdirectory within PVC containing cache files
- **Note**: This is appended to `PVC_MOUNT_PATH` to form the full cache path

### Threshold Configuration

#### `CLEANUP_THRESHOLD`
- **Default**: `85.0`
- **Description**: Disk usage percentage that triggers deletion (Activator sets deletion_event). Trigger deletion when usage >= 85%.

#### `TARGET_THRESHOLD`
- **Default**: `70.0`
- **Description**: Disk usage percentage that stops deletion (Activator clears deletion_event). Stop deletion when usage <= 70%.

**Large Storage Considerations:**
- On huge storage volumes (multi-TB), cache accumulates slowly in terms of percentage
- This means threshold crossings happen infrequently, allowing time for cleanup to complete
- The hysteresis buffer (15% in example) provides ample time for deletion to catch up
- Since `du` is too slow on huge storage, `statvfs()` is used instead (but may be not accurate)

### Hot Cache Configuration

#### `FILE_ACCESS_TIME_THRESHOLD_MINUTES`
- **Default**: `60.0`
- **Description**: Files accessed within this time (minutes) are skipped. Skip files accessed in last 60 minutes.

**File Access Time (atime) Considerations:**
- The cleanup pod uses `file_path.stat().st_atime` to determine last access time
- **Important**: Most filesystems use `relatime` (relative atime) by default, not `strictatime`
- **`relatime` behavior**: Updates atime only if:
  - The file was modified (`mtime` changed), OR
  - The atime is older than 24 hours, OR
  - The atime is older than the mtime/ctime
- **`strictatime` behavior**: Updates atime on every file access (more accurate but higher I/O overhead)
- **Impact**: With `relatime`, recently accessed files may not have updated atime, potentially causing false positives (deleting files that should be protected)
- **Current mitigation**: The cleanup pod still checks atime, but users should be aware that protection may be less reliable on `relatime` filesystems

### Multi-Process Configuration

#### `NUM_CRAWLER_PROCESSES`
- **Default**: `8`
- **Description**: Number of crawler processes (P1-PN). Valid values: 1, 2, 4, 8, 16

#### `LOGGER_INTERVAL_SECONDS`
- **Default**: `0.5`
- **Description**: How often Activator process checks disk usage (seconds)

### Queue Configuration

#### `FILE_QUEUE_MAXSIZE`
- **Default**: `10000`
- **Description**: Maximum items in file queue when deletion is ON.

**Memory Impact:**
- Each queue item is a file path string (~100-200 bytes)
- 10000 items ≈ 1-2 MB memory
- Larger queue = more memory, but better throughput

#### `FILE_QUEUE_MIN_SIZE`
- **Default**: `1000`
- **Description**: Pre-fill queue to this size when deletion is OFF.

**Rationale:**
- When deletion triggers, queue should already have files ready
- Pre-filling avoids delay waiting for crawlers to discover files
- 1000 items provides good buffer for immediate deletion

### Deletion Configuration

#### `DELETION_BATCH_SIZE`
- **Default**: `100`
- **Description**: Number of files per deletion batch (Deleter process).

**Performance:**
- Larger batches = fewer system calls, but longer batch duration
- Smaller batches = more system calls, but faster individual batches

### Safety Configuration

#### `DRY_RUN`
- **Default**: `false`
- **Description**: If `true`, simulate deletion without actually deleting files


#### `LOG_LEVEL`
- **Default**: `INFO`
- **Description**: Logging verbosity (DEBUG, INFO, WARNING, ERROR). Default `INFO` for production.

### Optional Configuration

#### `LOG_FILE_PATH`
- **Default**: `None`
- **Description**: Optional file path to also write logs to. Default `None` logs only to stdout.
- **Note**: Logs are always written to stdout; this adds file logging

#### `TIMING_FILE_PATH`
- **Default**: `/tmp/timing_analysis.txt`
- **Description**: Path for timing analysis file (currently not used)
- **Note**: Reserved for future timing analysis features

## Prerequisites

### Kubernetes/OpenShift Cluster
- Kubernetes 1.20+ or OpenShift 4.x+
- Access to create Deployments, PVCs
- Appropriate RBAC permissions
- `kubectl` CLI installed

### PVC Requirements
- PVC must exist and be bound
- PVC must be mounted to the pod at `PVC_MOUNT_PATH`
- PVC must have appropriate permissions for the pod's security context

### Security Context Constraints (SCC) - OpenShift
- Pod must run with appropriate SCC for the namespace
- SCC determines `fsGroup`, `seLinuxOptions`, `runAsUser`

**Finding SCC for Namespace:**
```bash
# Check pod's SCC
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.metadata.annotations.openshift\.io/scc}'

# Check SCC details
kubectl get scc <scc-name> -o yaml
```

### Docker Image

The evictor runs from a Docker image (`ghcr.io/guygir/pvc-evictor:latest`) that is built and maintained by the project maintainers. Users do not need to build the image - it's already available in the registry.

**Image Details:**
- Base image: `python:3.12-slim`
- Python 3.12+ required
- No additional dependencies (uses only standard library)
- All Python modules are included in the image

## Deployment

### Quick Deployment (Recommended)

For quick deployment with command-line configuration, use the `deploy.sh` script:

```bash
./deploy.sh <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>]
```

**Example:**
```bash
./deploy.sh test --namespace=e5 --fsgroup=1000960000 --selinux-level=s0:c31,c15 --runasuser=1000960000 --num-crawlers=16 --cleanup-threshold=25.0 --target-threshold=15.0
```

See [QUICK_START.md](QUICK_START.md) for detailed usage and examples.

## Monitoring

### Logs

The cleanup pod logs extensively to stdout (and optionally to a file). Key log patterns:

**Disk Usage Monitoring (Activator):**
```
PVC Usage: 85.42% (170.84GB / 200.00GB) [statvfs]
```

**Deletion Events:**
```
DELETION_START:1234567890.123,85.42
DELETION_END:1234567890.456,69.87
```

**Crawler Progress (P1-PN):**
```
Crawler P1: Queued 1000 files (discovered 5000, queue=5000/10000, deletion=ON)
```

**Deletion Progress (Deleter):**
```
Deleted batch: 100 files, 2.50GB freed (total: 1000 files, 25.00GB)
```

