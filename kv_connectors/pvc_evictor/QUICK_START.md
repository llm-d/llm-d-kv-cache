# Quick Start Guide - PVC Evictor

## Prerequisites

1. OpenShift/Kubernetes cluster access
2. `kubectl` CLI installed
3. PVC exists and is bound
4. Appropriate RBAC permissions
5. **Docker image available** - The evictor uses `ghcr.io/guygir/pvc-evictor:latest` (already built and pushed by maintainers)

## Quick Deployment

### Option 1: Using deploy.sh (Recommended)

```bash
./deploy.sh <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>]
```

**Arguments:**
- `pvc-name`: Name of the PVC to manage - **Required** (first positional argument)
- `--namespace=<namespace>`: Kubernetes namespace - **Optional** (auto-detected from `kubectl config` context if not provided)
- `--fsgroup=<fsgroup>`: Filesystem group ID - **Optional but Recommended** (auto-detected from existing pods/deployments if not provided)
- `--selinux-level=<level>`: SELinux security level - **Optional but Recommended** (auto-detected from existing pods/deployments if not provided)
- `--runasuser=<user>`: User ID to run containers as - **Optional but Recommended** (auto-detected from existing pods/deployments if not provided)
- `--num-crawlers=<n>`: Number of crawler processes, valid: 1, 2, 4, 8, 16 (default: 8) - Optional
- `--cleanup-threshold=<%>`: Disk usage % to trigger deletion (default: 85.0) - Optional
- `--target-threshold=<%>`: Disk usage % to stop deletion (default: 70.0) - Optional

**Note:** Only `pvc-name` is required. `namespace` will be auto-detected from your current `kubectl config` context if not provided. Security context values (`fsgroup`, `selinux-level`, `runasuser`) are optional but recommended - they will be auto-detected from existing pods/deployments in the namespace if not provided. If auto-detection fails, you must provide these values explicitly or the pod may fail to start. Arguments can be specified in any order.

**Example with auto-detected namespace (all defaults):**
```bash
./deploy.sh test
```

**Example with explicit namespace:**
```bash
./deploy.sh test --namespace=e5
```

**Example with custom settings (16 crawlers, custom thresholds):**
```bash
./deploy.sh test --namespace=e5 --fsgroup=1000960000 --selinux-level=s0:c31,c15 --runasuser=1000960000 --num-crawlers=16 --cleanup-threshold=25.0 --target-threshold=15.0
```

**Example with partial custom settings (only specify what you need):**
```bash
./deploy.sh test --num-crawlers=16 --cleanup-threshold=25.0 --target-threshold=15.0
```

## Verify Deployment

```bash
# Check pod status
kubectl get pods -n <namespace> | grep evictor

# View logs
kubectl logs -f deployment/pvc-evictor -n <namespace>

# Check PVC usage
kubectl exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache
```

## Configurations

### Default
- `CLEANUP_THRESHOLD`: `85.0`
- `TARGET_THRESHOLD`: `70.0`
- `FILE_ACCESS_TIME_THRESHOLD_MINUTES`: `60.0`
- `NUM_CRAWLER_PROCESSES`: `8` (valid: 1, 2, 4, 8, 16)

## Finding Your Namespace's Security Context

```bash
# Check an existing working pod in your namespace
kubectl get pod <any-pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <any-pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

## Troubleshooting

**Pod not starting?**
- Check PVC mount: `kubectl describe pod <pod-name> -n <namespace>`
- Verify Docker image exists: `kubectl describe pod <pod-name> -n <namespace> | grep Image`
- Check security context matches namespace SCC (see README.md for details)
- Verify Docker image exists: `kubectl describe pod <pod-name> -n <namespace> | grep Image`

**No files being deleted?**
- Check PVC usage: `kubectl exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache`
- Check logs for `DELETION_START` events
- Verify thresholds are appropriate

**Files not being discovered?**
- Verify `CACHE_DIRECTORY` matches actual cache path (default: `kv/model-cache/models`)
- Check logs for crawler heartbeats: `kubectl logs deployment/pvc-evictor -n <namespace> | grep Heartbeat`

For more details, see [README.md](README.md).

