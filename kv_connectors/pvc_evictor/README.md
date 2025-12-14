# PVC Evictor - Production Ready

## Overview

The PVC Evictor is multi-process Kubernetes deployment designed to automatically manage disk space on PVCs used for vLLM KV cache offloading. It monitors disk usage and automatically deletes old cache files when storage thresholds are exceeded, ensuring continuous operation of vLLM workloads without manual intervention.

## Architecture

### N+2 Process Design

- **P1-PN: Crawler Processes** - Discover and queue files for deletion (N configurable: 1, 2, 4, 8, or 16, default: 8)
- **P(N+1): Logger Process** - Monitors disk usage and controls deletion triggers
- **P(N+2): Deleter Process** - Performs batch file deletions

**Total Processes:** N + 2 (e.g., 8 crawlers + 1 logger + 1 deleter = 10 processes for default configuration)

### Key Features

1. **Streaming File Discovery** - Uses `os.scandir()` to avoid memory accumulation
2. **Bounded Queue** - Prevents memory overflow with configurable queue limits
3. **Event-Based Deletion Control** - Logger process triggers Deleter ON/OFF based on disk usage
4. **Fast Batch Deletion** - Uses `xargs rm -f` for efficient bulk operations
5. **Hot Cache Protection** - Skips recently accessed files to avoid deleting active cache
6. **Hex-Based Partitioning** - Distributes work across crawlers using hash folder modulo

## Files

### `deploy.sh`

Bash deployment script that automates the deployment process with command-line arguments.

**Usage:**
```bash
./deploy.sh <namespace> <pvc-name> [fsgroup] [selinux-level] [runasuser] [num-crawlers] [cleanup-threshold] [target-threshold]
```

**Features:**
- Creates or updates the ConfigMap automatically
- Substitutes all configuration values in the deployment YAML
- Supports all namespace-specific security contexts
- Configurable crawler count and thresholds via arguments
- All arguments are optional with sensible defaults

**See [QUICK_START.md](QUICK_START.md) for detailed usage and examples.**

### `deployment_evictor.yaml`

Kubernetes Deployment manifest for the evictor pod.

**Key Components:**
- Deployment with single replica (N+2 processes run inside one pod, where N is NUM_CRAWLER_PROCESSES)
- Security context configured for OpenShift SCCs (namespace-specific)
- Environment variables for all configuration
- Volume mounts for PVC and ConfigMap
- Health checks (liveness/readiness probes)
- Resource limits and requests

**Security Context (Namespace-Specific):**

The security context values are **namespace-specific** due to Security Context Constraints (SCCs) in OpenShift. Each namespace has its own SCC requirements that must be matched exactly, or the pod will fail to start.

**Default Values in deployment_evictor.yaml:**
- **e5 namespace:** `fsGroup=1000960000`, `seLinuxOptions.level="s0:c31,c15"`, `runAsUser=1000960000`
- **c3 namespace:** `fsGroup=1000940000`, `seLinuxOptions.level="s0:c31,c5"`, `runAsUser=1000940000`

**To find your namespace's values:**
```bash
# Check an existing working pod in your namespace
oc get pod <pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
oc get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

**Security Context Fields Explained:**
- `fsGroup`: Filesystem group ID - determines group ownership of volumes mounted to the pod
- `seLinuxOptions.level`: SELinux security level - namespace-specific security labeling
- `runAsUser`: User ID to run containers as - must match namespace SCC
- `allowPrivilegeEscalation: false`: Prevents privilege escalation attacks
- `capabilities.drop: ALL`: Drops all Linux capabilities (principle of least privilege)
- `runAsNonRoot: true`: Ensures containers run as non-root user for security
- `seccompProfile.type: RuntimeDefault`: Standard security profile for container runtime

**Important:** Before deploying, update the security context values in `deployment_evictor.yaml` to match your namespace's SCC requirements. 

**Recommended:** Use the `deploy.sh` script (see [QUICK_START.md](QUICK_START.md)) which can set all configuration values including security context, PVC name, number of crawlers, and thresholds via command-line arguments.

### `pvc_evictor.py`

Main Python script implementing the cleanup logic.

**Key Functions:**

#### `Config.from_env()`
Loads configuration from environment variables. Returns a `Config` dataclass with all settings.

#### `get_disk_usage_from_statvfs(mount_path: str)`
Gets disk usage using `os.statvfs()` - fastest method (~1ms). Returns `DiskUsage` dataclass with total, used, available bytes and usage percentage.

#### `stream_cache_files(cache_path: Path, hex_modulo_range: Tuple[int, int])`
Generator function that streams cache files using `os.scandir()`. Filters files by hex folder modulo to partition work across crawlers. Yields `Path` objects for `.bin` files.

**Cache Structure:**
```
/kv-cache/kv/model-cache/models/{model}/{path}/tp_{N}/rank_{M}/auto/{hash1}/{hash2}/.../*.bin
```

**Hex Partitioning:**
- Each crawler (P1-PN) handles files where `hex_folder_name % 16` is in its assigned range
- Valid crawler counts: 1, 2, 4, 8, 16 (must evenly divide 16)
- Examples:
  - 1 crawler: handles %16 in [0, 15] (all values)
  - 2 crawlers: P1 handles %16 in [0, 7], P2 handles [8, 15]
  - 4 crawlers: P1 handles [0, 3], P2 handles [4, 7], P3 handles [8, 11], P4 handles [12, 15]
  - 8 crawlers: P1 handles [0, 1], P2 handles [2, 3], ..., P8 handles [14, 15]
  - 16 crawlers: Each handles one value [0], [1], ..., [15]
- Ensures even distribution without coordination overhead

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

#### `logger_process(mount_path, cleanup_threshold, target_threshold, logger_interval, deletion_event, shutdown_event)`
Logger process (P(N+1)) that monitors disk usage and controls deletion.

**Monitoring:**
- Uses `get_disk_usage_from_statvfs()` every `logger_interval` seconds
- Logs usage percentage in format: `CACHE_PERCENT_LOG:{timestamp},{usage_percent},statvfs`

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
- Falls back to individual deletion if `xargs` fails
- Tracks files deleted and bytes freed

#### `delete_batch(file_paths: List[str], dry_run: bool, logger)`
Deletes a batch of files using `xargs rm -f`.

**Process:**
1. Validates paths and calculates total size
2. Uses null-terminated input (`\0`) for `xargs -0`
3. Runs `subprocess.run(['xargs', '-0', 'rm', '-f'], input=input_data)`
4. Falls back to `delete_individually()` on failure

#### `delete_individually(file_paths: List[str])`
Fallback deletion method using `file_path.unlink()`.

#### `PVCEvictor`
Main coordinator class that spawns and manages all processes.

**Initialization:**
- Waits for PVC mount to be ready (max 60s)
- Sets up signal handlers (SIGTERM, SIGINT)
- Creates shared IPC objects:
  - `deletion_event`: Multiprocessing Event (P9 controls, P1-P8/P10 check)
  - `file_queue`: Multiprocessing Queue (P1-P8 → P10)
  - `result_queue`: Multiprocessing Queue (P10 → Main)
  - `shutdown_event`: Multiprocessing Event (all processes check)

**Process Management:**
- Spawns P1-PN crawlers with assigned hex ranges (N = NUM_CRAWLER_PROCESSES)
- Spawns P(N+1) logger
- Spawns P(N+2) deleter
- Monitors process health and restarts on failure
- Handles graceful shutdown on signals

## Configuration

All configuration is done via environment variables in the deployment YAML.

### Required Configuration

#### `PVC_MOUNT_PATH`
- **Default**: `/kv-cache`
- **Description**: Mount path of the PVC in the pod
- **Example**: `/kv-cache`

#### `CACHE_DIRECTORY`
- **Default**: `kv/model-cache/models`
- **Description**: Subdirectory within PVC containing cache files
- **Example**: `kv/model-cache/models`
- **Note**: This is appended to `PVC_MOUNT_PATH` to form the full cache path

### Threshold Configuration

#### `CLEANUP_THRESHOLD`
- **Default**: `85.0`
- **Description**: Disk usage percentage that triggers deletion (Logger sets deletion_event)
- **Example**: `85.0` (trigger deletion when usage >= 85%)
- **Recommended**: `85.0` for production (allows headroom before full)

#### `TARGET_THRESHOLD`
- **Default**: `70.0`
- **Description**: Disk usage percentage that stops deletion (Logger clears deletion_event)
- **Example**: `70.0` (stop deletion when usage <= 70%)
- **Recommended**: `70.0` for production (provides buffer below cleanup threshold)

**Hysteresis Design:**
- Prevents oscillation by using different thresholds for ON/OFF
- Example: Cleanup at 85%, stop at 70% (15% buffer)

**Large Storage Considerations:**
- On huge storage volumes (multi-TB), cache accumulates slowly in terms of percentage
- This means threshold crossings happen infrequently, allowing time for cleanup to complete
- The hysteresis buffer (15% in example) provides ample time for deletion to catch up
- Since `du` is too slow on huge storage, `statvfs()` is used instead (O(1) operation)

### Hot Cache Configuration

#### `FILE_ACCESS_TIME_THRESHOLD_MINUTES`
- **Default**: `60.0`
- **Description**: Files accessed within this time (minutes) are skipped
- **Example**: `60.0` (skip files accessed in last 60 minutes)
- **Recommended**: `60.0` for production (protects active cache)

**Rationale:**
- vLLM may access cache files during inference
- Deleting active cache can cause performance degradation
- This threshold protects recently accessed files

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
- **For large storage systems**: Since cache accumulates slowly (in terms of %), the impact of false positives is minimal - files will be recreated if needed

### Multi-Process Configuration

#### `NUM_CRAWLER_PROCESSES`
- **Default**: `8`
- **Description**: Number of crawler processes (P1-PN). Valid values: 1, 2, 4, 8, 16
- **Example**: `8`
- **Recommended**: `8` (good balance of parallelism and resource usage)

**Valid Values:**
- Must be one of: 1, 2, 4, 8, 16 (must evenly divide 16 for hex partitioning)
- Invalid values will cause the evictor to exit with an error

**Scaling Considerations:**
- More crawlers = faster file discovery, but more CPU/memory
- Fewer crawlers = slower discovery, but less resource usage
- 8 crawlers typically handle 2TB PVCs efficiently
- 1-2 crawlers: Small PVCs (<500GB)
- 4 crawlers: Medium PVCs (500GB-1TB)
- 8 crawlers: Large PVCs (1TB-2TB)
- 16 crawlers: Very large PVCs (>2TB) or high file count

#### `LOGGER_INTERVAL_SECONDS`
- **Default**: `0.5`
- **Description**: How often Logger process checks disk usage (seconds)
- **Example**: `0.5` (check every 500ms)
- **Recommended**: `0.5` for responsive monitoring

**Trade-offs:**
- Lower interval = more responsive, but more CPU usage
- Higher interval = less CPU, but slower reaction to threshold changes

### Queue Configuration

#### `FILE_QUEUE_MAXSIZE`
- **Default**: `10000`
- **Description**: Maximum items in file queue when deletion is ON
- **Example**: `10000`
- **Recommended**: `10000` (good balance of memory and throughput)

**Memory Impact:**
- Each queue item is a file path string (~100-200 bytes)
- 10000 items ≈ 1-2 MB memory
- Larger queue = more memory, but better throughput

#### `FILE_QUEUE_MIN_SIZE`
- **Default**: `1000`
- **Description**: Pre-fill queue to this size when deletion is OFF
- **Example**: `1000`
- **Recommended**: `1000` (ensures fast start when deletion triggers)

**Rationale:**
- When deletion triggers, queue should already have files ready
- Pre-filling avoids delay waiting for crawlers to discover files
- 1000 items provides good buffer for immediate deletion

### Deletion Configuration

#### `DELETION_BATCH_SIZE`
- **Default**: `100`
- **Description**: Number of files per deletion batch (Deleter process)
- **Example**: `100`
- **Recommended**: `100` (good balance of efficiency and latency)

**Performance:**
- Larger batches = fewer system calls, but longer batch duration
- Smaller batches = more system calls, but faster individual batches
- 100 files per batch typically optimal for `xargs rm -f`

### Safety Configuration

#### `DRY_RUN`
- **Default**: `false`
- **Description**: If `true`, simulate deletion without actually deleting files
- **Example**: `true` for testing
- **Recommended**: `false` for production

**Use Cases:**
- Testing configuration without risk
- Validating file discovery and queueing logic
- Measuring performance without actual deletion

#### `LOG_LEVEL`
- **Default**: `INFO`
- **Description**: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- **Example**: `DEBUG` for troubleshooting
- **Recommended**: `INFO` for production

**Log Levels:**
- `DEBUG`: Very verbose, includes timing events
- `INFO`: Standard operational logs
- `WARNING`: Warnings only
- `ERROR`: Errors only

### Optional Configuration

#### `LOG_FILE_PATH`
- **Default**: `None` (logs only to stdout)
- **Description**: Optional file path to also write logs to
- **Example**: `/tmp/v4_cleanup_all_logs.txt`
- **Note**: Logs are always written to stdout; this adds file logging

#### `TIMING_FILE_PATH`
- **Default**: `/tmp/timing_analysis.txt`
- **Description**: Path for timing analysis file (currently not used in v4)
- **Note**: Reserved for future timing analysis features

## Prerequisites

### Kubernetes/OpenShift Cluster
- Kubernetes 1.20+ or OpenShift 4.x+
- Access to create Deployments, ConfigMaps, PVCs
- Appropriate RBAC permissions

### PVC Requirements
- PVC must exist and be bound
- PVC must be mounted to the pod at `PVC_MOUNT_PATH`
- PVC must have appropriate permissions for the pod's security context

### Security Context Constraints (SCC) - OpenShift
- Pod must run with appropriate SCC for the namespace
- Common SCCs:
  - `restricted-v2` (most restrictive)
  - `restricted` (standard)
- SCC determines `fsGroup`, `seLinuxOptions`, `runAsUser`

**Finding SCC for Namespace:**
```bash
# Check pod's SCC
oc get pod <pod-name> -n <namespace> -o jsonpath='{.metadata.annotations.openshift\.io/scc}'

# Check SCC details
oc get scc <scc-name> -o yaml
```

### ConfigMap Creation
The deployment expects a ConfigMap named `pvc-evictor-script` containing the Python script.

**Creating ConfigMap:**
```bash
oc create configmap pvc-evictor-script \
  --from-file=pvc_evictor.py \
  -n <namespace>
```

### Image Requirements
- Base image: `python:3.12-slim` (or compatible)
- Python 3.12+ required
- No additional dependencies (uses only standard library)

## Deployment

### Quick Deployment (Recommended)

For quick deployment with command-line configuration, use the `deploy.sh` script:

```bash
./deploy.sh <namespace> <pvc-name> [fsgroup] [selinux-level] [runasuser] [num-crawlers] [cleanup-threshold] [target-threshold]
```

**Example:**
```bash
./deploy.sh e5 test 1000960000 s0:c31,c15 1000960000 16 25.0 15.0
```

See [QUICK_START.md](QUICK_START.md) for detailed usage and examples.

### Manual Deployment

#### Step 1: Create ConfigMap

```bash
# From the PR directory
oc create configmap pvc-evictor-script \
  --from-file=pvc_evictor.py \
  -n <namespace>
```

#### Step 2: Update Deployment YAML

Edit `deployment_evictor.yaml`:

1. **Update namespace:**
   ```yaml
   metadata:
     namespace: <your-namespace>
   ```

2. **Update PVC name:**
   ```yaml
   volumes:
     - name: kv-cache-storage
       persistentVolumeClaim:
         claimName: <your-pvc-name>
   ```

3. **Update security context for your namespace:**
   ```yaml
   securityContext:
     fsGroup: <your-fsgroup>  # e.g., 1000940000 for c3, 1000960000 for e5
     seLinuxOptions:
       level: "<your-selinux-level>"  # e.g., "s0:c31,c5" for c3, "s0:c31,c15" for e5
   ```

4. **Update runAsUser:**
   ```yaml
   securityContext:
     runAsUser: <your-user-id>  # e.g., 1000940000 for c3, 1000960000 for e5
   ```

5. **Update environment variables:**
   - `PVC_MOUNT_PATH`: Mount path in pod (usually `/kv-cache`)
   - `CACHE_DIRECTORY`: Subdirectory containing cache files
   - `NUM_CRAWLER_PROCESSES`: Number of crawler processes (1, 2, 4, 8, or 16)
   - `CLEANUP_THRESHOLD`: Disk usage % to trigger deletion
   - `TARGET_THRESHOLD`: Disk usage % to stop deletion
   - `FILE_ACCESS_TIME_THRESHOLD_MINUTES`: Hot cache protection time
   - Other settings as needed

#### Step 3: Deploy

```bash
oc apply -f deployment_evictor.yaml
```

### Step 4: Verify

```bash
# Check pod status
oc get pods -n <namespace> | grep evictor

# Check logs
oc logs -f deployment/pvc-evictor -n <namespace>

# Check PVC usage (should be monitored by P9)
oc exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache
```

## Monitoring

### Logs

The cleanup pod logs extensively to stdout (and optionally to a file). Key log patterns:

**Disk Usage Monitoring (Logger):**
```
CACHE_PERCENT_LOG:1234567890.123,85.42,statvfs
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

### Metrics

Currently, metrics are available via logs. Future enhancements could include:
- Prometheus metrics endpoint
- PVC usage percentage gauge
- Files deleted counter
- Bytes freed counter
- Deletion duration histogram

### Health Checks

The deployment includes liveness and readiness probes:
- **Liveness**: Checks if `/kv-cache` mount exists
- **Readiness**: Checks if `/kv-cache` mount exists

Both probes use:
```python
python3 -c "import os; exit(0 if os.path.exists('/kv-cache') else 1)"
```

## Advantages

### Performance
1. **Multi-Process Parallelism**: 8 crawlers discover files in parallel
2. **Streaming Discovery**: No memory accumulation during file scanning
3. **Batch Deletion**: `xargs rm -f` is fastest deletion method
4. **Event-Based Control**: Minimal overhead when deletion is OFF
5. **Fast Disk Usage Monitoring**: Uses `statvfs()` instead of `du` (O(1) vs O(n))
   - **Critical for huge storage**: `du` can take hours on multi-TB volumes
   - `statvfs()` provides instant disk usage statistics regardless of volume size

### Reliability
1. **Hot Cache Protection**: Prevents deletion of active cache entries
2. **Hysteresis Design**: Prevents oscillation between ON/OFF states
3. **Graceful Shutdown**: Handles SIGTERM/SIGINT properly
4. **Process Health Monitoring**: Restarts failed processes automatically

### Scalability
1. **Bounded Memory**: Queue limits prevent memory overflow
2. **Configurable Parallelism**: Adjust crawler count based on PVC size
3. **Efficient Partitioning**: Hex-based work distribution avoids coordination

### Safety
1. **Dry Run Mode**: Test without risk
2. **Access Time Filtering**: Protects recently accessed files
3. **No File Locking**: Uses `os.scandir()` and `stat()` without locking
4. **Atomic Deletion**: `xargs rm -f` is atomic per file

## Disadvantages

### Limitations
1. **Single Pod**: All processes run in one pod (single point of failure)
   - **Mitigation**: Kubernetes will restart pod on failure
2. **No Distributed Coordination**: Cannot scale across multiple pods
   - **Mitigation**: Single pod is sufficient for most PVC sizes
3. **No Prometheus Metrics**: Currently logs-only monitoring
   - **Mitigation**: Can parse logs for metrics
4. **Python GIL**: Python's Global Interpreter Lock limits true parallelism
   - **Mitigation**: I/O-bound operations (file scanning) are not GIL-limited

### Known Issues
1. **Queue Size Estimation**: `qsize()` is approximate, not exact
   - **Impact**: Minor, queue limits are soft bounds
2. **File Access Time (atime) Reliability**: Most filesystems use `relatime` instead of `strictatime`
   - **Impact**: May delete files that should be protected (rare, but possible)
   - **Details**: 
     - `relatime` only updates atime under specific conditions (see Hot Cache Configuration)
     - `strictatime` updates atime on every access but requires filesystem mount option
     - On large storage systems, this is less critical since cache accumulates slowly and files can be regenerated
   - **Mitigation**: Set `FILE_ACCESS_TIME_THRESHOLD_MINUTES` conservatively, or use `strictatime` mount option (see Future Optimizations)
3. **Process Restart Delay**: Failed processes restart with delay
   - **Impact**: Temporary reduction in throughput
4. **Large Storage Limitations**: Commands like `du` are too slow on huge storage volumes
   - **Impact**: Cannot use `du` for disk usage monitoring
   - **Mitigation**: Uses `statvfs()` instead, which is O(1) and provides instant disk usage statistics

## Future Optimizations

### Short-Term
1. **Prometheus Metrics**: Add metrics endpoint for monitoring
2. **ConfigMap Hot Reload**: Reload configuration without pod restart
3. **Better Queue Size Tracking**: More accurate queue size estimation
4. **Deletion Rate Limiting**: Prevent overwhelming storage subsystem
5. **File Re-queuing Deduplication**: The crawler processes continuously rescan the cache directory and can queue the same file multiple times across scan cycles, wasting queue space and CPU. A hash-based in-memory deduplication cache (8-byte MD5 hashes, 10k entries per crawler) would prevent duplicate queue entries with minimal overhead (~640KB total memory, ~0.001ms per file). This optimization is particularly beneficial for large PVCs where duplicates accumulate over time, though the current implementation remains functional without it due to queue size limits and safe deletion handling.

### Medium-Term
1. **Distributed Mode**: Scale across multiple pods with coordination
2. **Smart Partitioning**: Dynamic hex range assignment based on load
3. **Predictive Deletion**: Delete files likely to be unused soon
4. **Compression**: Compress old cache files instead of deleting
5. **(NEW) Strictatime Support**: Enhanced atime-based hot cache protection
   - **Problem**: Most filesystems use `relatime` by default, which doesn't update atime on every access
   - **Solution**: Support for `strictatime` mount option to get accurate access time tracking
   - **Requirements**:
     - PVC must be mounted with `strictatime` option (requires storage class support or manual mount configuration)
     - May require privileged pod or init container to remount with `strictatime`
     - Alternative: Use init container to remount PVC with `mount -o remount,strictatime /kv-cache`
   - **Trade-offs**:
     - **Advantage**: More accurate hot cache protection, fewer false positives
     - **Disadvantage**: Higher I/O overhead (every file access updates atime metadata)
     - **For large storage**: Overhead is acceptable since cache accumulates slowly and protection accuracy is valuable
   - **Implementation**:
     - Add init container to check/remount PVC with `strictatime` if not already set
     - Add configuration option `ENABLE_STRICTATIME` (default: `false` for backward compatibility)
     - Enhance logging to indicate whether `strictatime` is active
   - **Considerations for huge storage**:
     - Atime updates are metadata-only operations (fast)
     - On large storage with slow cache accumulation, the I/O overhead is negligible
     - The benefit of accurate hot cache protection outweighs the minimal overhead

### Long-Term
1. **Machine Learning**: Predict cache access patterns
2. **Tiered Storage**: Move old cache to cheaper storage
3. **Cache Warming**: Pre-load frequently used cache entries
4. **Multi-Tenant Support**: Handle multiple models/namespaces
5. **(NEW) Alternative Hot Cache Detection**: Methods beyond atime for huge storage
   - **Problem**: Atime may be unreliable or unavailable on some storage backends
   - **Solutions**:
     - **File modification time (mtime) tracking**: Track when files were last written (more reliable than atime)
     - **LRU metadata file**: Maintain separate metadata file tracking access patterns (requires coordination with vLLM)
     - **Heuristic-based**: Use file age + size + location patterns to predict hot cache
     - **vLLM integration**: Query vLLM API for active cache entries (requires vLLM support)
   - **For huge storage**: Since cache accumulates slowly, simpler heuristics (e.g., "delete oldest files first") may be sufficient
   - **Performance**: Avoids filesystem metadata overhead, better suited for massive storage volumes

## Troubleshooting

### Pod Not Starting

**Check PVC mount:**
```bash
oc describe pod <pod-name> -n <namespace> | grep -A 5 "Mounts"
```

**Check security context:**
```bash
oc get pod <pod-name> -n <namespace> -o yaml | grep -A 10 securityContext
```

**Check ConfigMap:**
```bash
oc get configmap pvc-evictor-script -n <namespace>
```

### No Files Being Deleted

**Check thresholds:**
```bash
oc logs deployment/pvc-cleanup-v4-multiprocess -n <namespace> | grep "CACHE_PERCENT_LOG"
```

**Check deletion event:**
```bash
oc logs deployment/pvc-cleanup-v4-multiprocess -n <namespace> | grep "DELETION_START\|DELETION_END"
```

**Verify PVC usage:**
```bash
oc exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache
```

### High CPU Usage

**Reduce crawler count:**
```yaml
env:
  - name: NUM_CRAWLER_PROCESSES
    value: "4"  # Reduce from 8
```

**Increase logger interval:**
```yaml
env:
  - name: LOGGER_INTERVAL_SECONDS
    value: "1.0"  # Increase from 0.5
```

### Memory Issues

**Reduce queue sizes:**
```yaml
env:
  - name: FILE_QUEUE_MAXSIZE
    value: "5000"  # Reduce from 10000
  - name: FILE_QUEUE_MIN_SIZE
    value: "500"  # Reduce from 1000
```

### Files Not Being Discovered

**Check cache directory:**
```bash
oc exec -it deployment/pvc-evictor -n <namespace> -- ls -la /kv-cache/kv/model-cache/models
```

**Verify hex partitioning:**
```bash
oc logs deployment/pvc-evictor -n <namespace> | grep "Crawler P.*started"
```

**Check file structure:**
```bash
oc exec -it deployment/pvc-evictor -n <namespace> -- find /kv-cache -name "*.bin" | head -10
```

## Example Configurations

### Production (2TB PVC, High Performance)
```yaml
env:
  - name: CLEANUP_THRESHOLD
    value: "85.0"
  - name: TARGET_THRESHOLD
    value: "70.0"
  - name: FILE_ACCESS_TIME_THRESHOLD_MINUTES
    value: "60.0"
  - name: NUM_CRAWLER_PROCESSES
    value: "8"
  - name: LOGGER_INTERVAL_SECONDS
    value: "0.5"
  - name: FILE_QUEUE_MAXSIZE
    value: "10000"
  - name: DELETION_BATCH_SIZE
    value: "100"
```

### Testing (Small PVC, Dry Run)
```yaml
env:
  - name: CLEANUP_THRESHOLD
    value: "50.0"
  - name: TARGET_THRESHOLD
    value: "40.0"
  - name: DRY_RUN
    value: "true"
  - name: LOG_LEVEL
    value: "DEBUG"
  - name: NUM_CRAWLER_PROCESSES
    value: "4"
```

### Conservative (Protect More Cache)
```yaml
env:
  - name: CLEANUP_THRESHOLD
    value: "90.0"
  - name: TARGET_THRESHOLD
    value: "80.0"
  - name: FILE_ACCESS_TIME_THRESHOLD_MINUTES
    value: "120.0"  # 2 hours
```

## References

- **vLLM KV Cache Offloading**: [vLLM Documentation](https://docs.vllm.ai/)
- **Kubernetes Deployments**: [Kubernetes Docs](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- **OpenShift SCC**: [OpenShift Security Context Constraints](https://docs.openshift.com/container-platform/4.15/authentication/managing-security-context-constraints.html)
- **Python Multiprocessing**: [Python Docs](https://docs.python.org/3/library/multiprocessing.html)

## License

[Specify license if applicable]

## Contributors

[Add contributors if applicable]

