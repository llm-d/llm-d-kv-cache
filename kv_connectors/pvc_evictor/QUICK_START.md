# Quick Start Guide - PVC Evictor

## Prerequisites

1. OpenShift/Kubernetes cluster access
2. `oc` or `kubectl` CLI installed
3. PVC exists and is bound
4. Appropriate RBAC permissions

## Quick Deployment

### Option 1: Using deploy.sh (Recommended)

```bash
cd PR
./deploy.sh <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>]
```

**Arguments:**
- `pvc-name`: Name of the PVC to manage - **Required** (first positional argument)
- `--namespace=<namespace>`: Kubernetes namespace - **Optional** (auto-detected from `oc project` or `kubectl config` if not provided)
- `--fsgroup=<fsgroup>`: Filesystem group ID (default: 1000960000 for e5) - Optional
- `--selinux-level=<level>`: SELinux security level (default: s0:c31,c15 for e5) - Optional
- `--runasuser=<user>`: User ID to run containers as (default: 1000960000 for e5) - Optional
- `--num-crawlers=<n>`: Number of crawler processes, valid: 1, 2, 4, 8, 16 (default: 8) - Optional
- `--cleanup-threshold=<%>`: Disk usage % to trigger deletion (default: 85.0) - Optional
- `--target-threshold=<%>`: Disk usage % to stop deletion (default: 70.0) - Optional

**Note:** Only `pvc-name` is required. `namespace` will be auto-detected from your current OpenShift/Kubernetes context if not provided. All other arguments are optional and will use sensible defaults if not provided. Arguments can be specified in any order. For namespaces other than e5 or c3, you may need to find the security context values (see "Finding Your Namespace's Security Context" below).

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

**Example for c3 namespace:**
```bash
./deploy.sh kv-cache-pvc-ready-2tb --namespace=c3 --fsgroup=1000940000 --selinux-level=s0:c31,c5 --runasuser=1000940000
```

### Option 2: Manual Deployment

1. **Create ConfigMap:**
   ```bash
   oc create configmap pvc-evictor-script \
     --from-file=pvc_evictor.py \
     -n <namespace>
   ```

2. **Edit deployment_evictor.yaml:**
   - Update `namespace`
   - Update `claimName` (PVC name)
   - Update `fsGroup`, `seLinuxOptions.level`, `runAsUser` for your namespace
   - Optionally update `NUM_CRAWLER_PROCESSES`, `CLEANUP_THRESHOLD`, `TARGET_THRESHOLD` if needed

3. **Deploy:**
   ```bash
   oc apply -f deployment_evictor.yaml
   ```

## Verify Deployment

```bash
# Check pod status
oc get pods -n <namespace> | grep evictor

# View logs
oc logs -f deployment/pvc-evictor -n <namespace>

# Check PVC usage
oc exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache
```

## Common Configurations

### Production (2TB PVC)
- `CLEANUP_THRESHOLD`: `85.0`
- `TARGET_THRESHOLD`: `70.0`
- `FILE_ACCESS_TIME_THRESHOLD_MINUTES`: `60.0`
- `NUM_CRAWLER_PROCESSES`: `8` (valid: 1, 2, 4, 8, 16)

### Testing (Dry Run)
- `DRY_RUN`: `true`
- `LOG_LEVEL`: `DEBUG`
- `CLEANUP_THRESHOLD`: `50.0`

## Finding Your Namespace's Security Context

```bash
# Check an existing working pod in your namespace
oc get pod <any-pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
oc get pod <any-pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

## Troubleshooting

**Pod not starting?**
- Check PVC mount: `oc describe pod <pod-name> -n <namespace>`
- Check ConfigMap: `oc get configmap pvc-evictor-script -n <namespace>`
- Check security context matches namespace SCC (see README.md for details)

**No files being deleted?**
- Check PVC usage: `oc exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache`
- Check logs for `DELETION_START` events
- Verify thresholds are appropriate

**Files not being discovered?**
- Verify `CACHE_DIRECTORY` matches actual cache path (default: `kv/model-cache/models`)
- Check logs for crawler heartbeats: `oc logs deployment/pvc-evictor -n <namespace> | grep Heartbeat`

For more details, see [README.md](README.md).

