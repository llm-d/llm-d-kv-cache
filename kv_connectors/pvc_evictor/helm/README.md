# PVC Evictor Helm Chart

This Helm chart deploys the PVC Evictor, a multi-process Kubernetes deployment that automatically manages disk space on PVCs used for vLLM KV-cache storage.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PVC must exist and be bound
- Appropriate RBAC permissions to create deployments

## Installation

### Quick Start

```bash
# Install with required values
helm install pvc-evictor ./helm \
  --set pvc.name=my-kv-cache-pvc \
  --set securityContext.pod.fsGroup=1000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c123,c456" \
  --set securityContext.container.runAsUser=1000
```

### Installation with Custom Values

```bash
# Install with custom configuration
helm install pvc-evictor ./helm \
  --set pvc.name=my-kv-cache-pvc \
  --set securityContext.pod.fsGroup=1000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c123,c456" \
  --set securityContext.container.runAsUser=1000 \
  --set config.numCrawlerProcesses=16 \
  --set config.cleanupThreshold=90.0 \
  --set config.targetThreshold=75.0
```

### Installation with Values File

Create a `my-values.yaml` file:

```yaml
pvc:
  name: my-kv-cache-pvc

securityContext:
  pod:
    fsGroup: 1000
    seLinuxOptions:
      level: "s0:c123,c456"
  container:
    runAsUser: 1000

config:
  numCrawlerProcesses: 16
  cleanupThreshold: 90.0
  targetThreshold: 75.0
  logLevel: DEBUG
```

Then install:

```bash
helm install pvc-evictor ./helm -f my-values.yaml
```

## Configuration

### Required Values

These values **must** be set during installation:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `pvc.name` | Name of the PVC to manage | `my-kv-cache-pvc` |
| `securityContext.pod.fsGroup` | Filesystem group ID for volume ownership | `1000` |
| `securityContext.pod.seLinuxOptions.level` | SELinux security level | `"s0:c123,c456"` |
| `securityContext.container.runAsUser` | User ID to run container as | `1000` |

### Finding Security Context Values

To find the correct security context values for your namespace:

```bash
# Get values from an existing pod in your namespace
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

### All Configuration Parameters

#### Image Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `ghcr.io/guygir/pvc-evictor` |
| `image.tag` | Container image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |

#### PVC Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pvc.name` | Name of PVC to mount (REQUIRED) | `""` |
| `pvc.mountPath` | Mount path inside container | `/kv-cache` |
| `pvc.readOnly` | Mount as read-only | `false` |

#### Evictor Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.cleanupThreshold` | Disk usage % to trigger deletion | `85.0` |
| `config.targetThreshold` | Disk usage % to stop deletion | `70.0` |
| `config.cacheDirectory` | Cache directory relative to mount path | `kv/model-cache/models` |
| `config.numCrawlerProcesses` | Number of crawler processes (1,2,4,8,16) | `8` |
| `config.loggerIntervalSeconds` | Monitoring interval in seconds | `0.5` |
| `config.fileQueueMaxsize` | Max queue size when deletion is ON | `10000` |
| `config.fileQueueMinSize` | Pre-fill queue size when deletion is OFF | `1000` |
| `config.deletionBatchSize` | Files per deletion batch | `100` |
| `config.fileAccessTimeThresholdMinutes` | Skip files accessed within N minutes | `60` |
| `config.dryRun` | Enable dry-run mode (no actual deletions) | `false` |
| `config.logLevel` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

#### Cache Structure Configuration

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `config.cacheStructureMode` | Cache directory structure mode | `"file_mapper"` | `"file_mapper"`, `"vllm"` |

**Modes:**
- **`file_mapper`** (Recommended): Uses FileMapper class for canonical cache structure. Automatically discovers all configurations and matches the structure used by the vLLM offloader.
- **`vllm`** (Legacy): Uses hardcoded vLLM structure for backward compatibility with older deployments.

**Example:**
```yaml
config:
  cacheStructureMode: "file_mapper"  # Default
```

**Note:** The `customScanPattern` parameter has been removed. FileMapper provides the canonical structure, eliminating the need for custom patterns.

| `config.logFilePath` | Path to log file | `/tmp/evictor_all_logs.txt` |

#### Resource Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.requests.cpu` | CPU request | `1` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.limits.cpu` | CPU limit | `4000m` |
| `resources.limits.memory` | Memory limit | `2Gi` |

#### Health Check Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `livenessProbe.enabled` | Enable liveness probe | `true` |
| `livenessProbe.initialDelaySeconds` | Initial delay | `30` |
| `livenessProbe.periodSeconds` | Check period | `30` |
| `readinessProbe.enabled` | Enable readiness probe | `true` |
| `readinessProbe.initialDelaySeconds` | Initial delay | `10` |
| `readinessProbe.periodSeconds` | Check period | `10` |

#### Other Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `serviceAccountName` | Service account name | `default` |
| `labels` | Pod labels | See values.yaml |

**Note:** Additional Kubernetes scheduling options (annotations, nodeSelector, tolerations, affinity) can be added to values.yaml if needed for advanced deployment scenarios.

## Upgrading

```bash
# Upgrade with new values
helm upgrade pvc-evictor ./helm \
  --set config.cleanupThreshold=90.0 \
  --reuse-values
```

## Uninstalling

```bash
helm uninstall pvc-evictor
```

## Examples

### Enable Dry-Run Mode for Testing

```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-pvc \
  --set securityContext.pod.fsGroup=1000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c123,c456" \
  --set securityContext.container.runAsUser=1000 \
  --set config.dryRun=true
```

### High-Performance Configuration

```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-pvc \
  --set securityContext.pod.fsGroup=1000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c123,c456" \
  --set securityContext.container.runAsUser=1000 \
  --set config.numCrawlerProcesses=16 \
  --set resources.limits.cpu=8000m \
  --set resources.limits.memory=4Gi
```

### Debug Mode

```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-pvc \
  --set securityContext.pod.fsGroup=1000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c123,c456" \
  --set securityContext.container.runAsUser=1000 \
  --set config.logLevel=DEBUG \
  --set config.dryRun=true
```

## Troubleshooting

### Pod fails to start with permission errors

Check that your security context values match your namespace's Security Context Constraints (SCC):

```bash
# Get security context from existing pods
kubectl get pods -n <namespace> -o jsonpath='{.items[0].spec.securityContext}'
```

### Files are not being deleted

1. Check that `config.dryRun` is set to `false`
2. Verify disk usage exceeds `config.cleanupThreshold`
3. Check logs: `kubectl logs -f deployment/pvc-evictor`

### High CPU usage

Reduce `config.numCrawlerProcesses` or adjust `resources.limits.cpu`

## More Information

For detailed documentation, see:
- [Main README](../README.md)
- [Quick Start Guide](../QUICK_START.md)
- [GitHub Repository](https://github.com/foundation-model-stack/llm-d-kv-cache-manager)