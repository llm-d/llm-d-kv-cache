# Quick Start Guide

Deploy the PVC Evictor to automatically manage disk space on your vLLM KV-cache PVC.

## Prerequisites

- Kubernetes cluster with `kubectl` access
- Helm 3.0+ installed
- Existing PVC with vLLM cache files
- Security context values for your namespace (see below)

## Installation

### 1. Get Security Context Values

**Important:** The evictor must use the same security context as your vLLM pods to access the shared PVC. Find your vLLM pod's security context:

```bash
# Replace <vllm-pod> with your actual vLLM pod name
kubectl get pod <vllm-pod> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <vllm-pod> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

You need three values:
- **fsGroup**: Group ownership for mounted volumes (must match vLLM pods)
- **seLinuxOptions.level**: SELinux label for OpenShift multi-tenancy (must match vLLM pods)
- **runAsUser**: User ID for container processes (must match vLLM pods)

### 2. Install with Helm

**Quick install:**
```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-vllm-cache \
  --set securityContext.pod.fsGroup=1000960000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c31,c15" \
  --set securityContext.container.runAsUser=1000960000
```

**Using values file (recommended):**

Create `my-values.yaml`:
```yaml
pvc:
  name: my-vllm-cache

securityContext:
  pod:
    fsGroup: 1000960000
    seLinuxOptions:
      level: "s0:c31,c15"
  container:
    runAsUser: 1000960000

config:
  cleanupThreshold: 85.0
  targetThreshold: 70.0
  numCrawlerProcesses: 8
```

Install with the values file:
```bash
helm install pvc-evictor ./helm -f my-values.yaml
```

### 3. Verify Deployment

```bash
# Check status
helm status pvc-evictor
kubectl get pods -l app.kubernetes.io/name=pvc-evictor

# View logs
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

## Configuration

Key settings (all optional, defaults shown):

| Setting | Default | Description |
|---------|---------|-------------|
| `cleanupThreshold` | 85.0 | Disk usage % to trigger deletion |
| `targetThreshold` | 70.0 | Disk usage % to stop deletion |
| `numCrawlerProcesses` | 8 | Parallel file discovery (1, 2, 4, 8, or 16) |
| `cacheDirectory` | `kv/model-cache/models` | Cache path relative to PVC mount |
| `fileAccessTimeThresholdMinutes` | 60 | Protect files accessed within N minutes |


For complete configuration reference, see [CONFIGURATION.md](CONFIGURATION.md).

## Monitoring

Watch logs in real-time:
```bash
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

Key log patterns:
- `DELETION_START` - Deletion triggered (usage >= cleanup threshold)
- `DELETION_END` - Deletion stopped (usage <= target threshold)
- `System Status` - Aggregated statistics (every 30 seconds)

## Troubleshooting

**Pod not starting?**
- Verify PVC exists: `kubectl get pvc <pvc-name>`
- Check security context matches namespace SCC
- Review pod events: `kubectl describe pod <pod-name>`

**No files being deleted?**
- Check disk usage hasn't reached cleanup threshold
- Verify cache directory path is correct
- Check logs for crawler discovery: `kubectl logs <pod> | grep discovered`

**Need more help?**
- See [README.md](README.md) for architecture details
- See [CONFIGURATION.md](CONFIGURATION.md) for all settings
- See [helm/README.md](helm/README.md) for Helm chart details
