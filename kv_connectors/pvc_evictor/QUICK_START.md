# Quick Start Guide - PVC Evictor

This guide walks you through deploying the PVC Evictor to automatically manage disk space on your vLLM KV-cache PVC. You'll learn how to deploy using Helm (recommended), the automated bash script, or manual YAML configuration.

## Prerequisites

1. OpenShift/Kubernetes cluster access
2. `kubectl` CLI installed
3. Helm 3.0+ installed (for Helm deployment)
4. PVC exists and is bound
5. Appropriate RBAC permissions to create deployments
6. **Docker image available** - The evictor uses `ghcr.io/guygir/pvc-evictor:latest`
7. **Security context values** - Required for OpenShift/Kubernetes Security Context Constraints (SCC). These values are namespace-specific.

## Deployment Options

### Option 1: Using Helm Chart (Recommended)

Helm provides the most maintainable deployment method with built-in validation and templating.

**Basic installation:**
```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-vllm-cache \
  --set securityContext.pod.fsGroup=1000960000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c31,c15" \
  --set securityContext.container.runAsUser=1000960000
```

**Using a values file (recommended for complex configurations):**
```bash
# Create my-values.yaml
cat > my-values.yaml <<EOF
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
  numCrawlerProcesses: 16
  cleanupThreshold: 90.0
EOF

# Install using the values file
helm install pvc-evictor ./helm -f my-values.yaml
```

**Finding security context values:**
```bash
# Get values from an existing pod in your namespace
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

**Check deployment status:**
```bash
helm status pvc-evictor
kubectl get pods -l app.kubernetes.io/name=pvc-evictor
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

For complete Helm documentation, see [helm/README.md](helm/README.md).
### Option 2: Using deploy.sh (Legacy - Automated Script)

**Note:** The deploy.sh script is maintained for backward compatibility but Helm is now the recommended deployment method.

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

**Security Context Explained:**

The security context values are required for OpenShift/Kubernetes Security Context Constraints (SCC) compliance:

- **`fsGroup`**: Controls the group ownership of mounted volumes. Files created on the PVC will be owned by this group ID, ensuring the evictor can read and delete cache files.
- **`seLinuxOptions.level`**: SELinux security level label required for multi-tenant OpenShift clusters. This ensures proper isolation between namespaces.
- **`runAsUser`**: The user ID that container processes run as. Must match the namespace's SCC requirements to prevent permission denied errors.

**Auto-Detection Logic:**

The `deploy.sh` script automatically detects these values by:
1. Querying existing pods in the target namespace using `kubectl get pods`
2. Extracting security context values from the first pod found
3. If multiple deployments have different values, the script uses the first match
4. If no pods exist in the namespace, you must provide these values manually

**Note:** Only `pvc-name` is required. `namespace` will be auto-detected from your current `kubectl config` context if not provided. Security context values are namespace-specific - if auto-detection fails or no pods exist in the namespace, you must provide these values explicitly or the pod may fail to start due to SCC violations. Arguments can be specified in any order.

**Example - Comprehensive deployment with custom settings:**
```bash
# Deploy with all options specified
./deploy.sh my-vllm-cache \
  --namespace=my-namespace \
  --fsgroup=1000960000 \
  --selinux-level=s0:c31,c15 \
  --runasuser=1000960000 \
  --num-crawlers=16 \
  --cleanup-threshold=90.0 \
  --target-threshold=75.0

# Or use auto-detection for namespace and security context (recommended for most cases)
./deploy.sh my-vllm-cache --num-crawlers=16 --cleanup-threshold=90.0 --target-threshold=75.0

# Minimal deployment (all defaults, auto-detect everything)
./deploy.sh my-vllm-cache
```

For all available options, run:
```bash
./deploy.sh --help
```

### Option 3: Manual YAML Deployment

**Note:** Manual YAML deployment is the least maintainable option. Consider using Helm (Option 1) or deploy.sh (Option 2) instead.

If you prefer to manually edit and deploy the YAML configuration:

1. **Edit the deployment YAML:**
   ```bash
   # Copy the template
   cp deployment_evictor.yaml my-deployment.yaml
   
   # Edit the file and replace placeholders:
   # - {{PVC_NAME}} - Your PVC name
   # - {{NAMESPACE}} - Your namespace
   # - {{FS_GROUP}} - Your fsGroup value
   # - {{SELINUX_LEVEL}} - Your SELinux level
   # - {{RUN_AS_USER}} - Your runAsUser value
   # - {{NUM_CRAWLER_PROCESSES}} - Number of crawlers (1, 2, 4, 8, or 16)
   # - {{CLEANUP_THRESHOLD}} - Cleanup threshold percentage (e.g., 85.0)
   # - {{TARGET_THRESHOLD}} - Target threshold percentage (e.g., 70.0)
   ```

2. **Find your security context values:**
   ```bash
   # Check an existing pod in your namespace
   kubectl get pod <any-pod-name> -n <namespace> -o jsonpath='{.spec.securityContext}'
   ```

3. **Deploy:**
   ```bash
   kubectl apply -f my-deployment.yaml -n <namespace>
   ```

**Note:** Manual deployment requires you to know all security context values. Helm (Option 1) is recommended for maintainability, or use deploy.sh (Option 2) which handles auto-detection.
## Configuration Guide

The evictor's behavior can be customized via command-line arguments (deploy.sh) or environment variables (manual YAML deployment). Here are the most commonly adjusted settings:

### Common Configuration Parameters

**Number of Crawler Processes (`--num-crawlers` / `NUM_CRAWLER_PROCESSES`)**
- Default: `8`
- Valid values: 1, 2, 4, 8, 16
- More crawlers = faster file discovery on large directories
- Recommendation: Use 8-16 for multi-TB volumes, 1-4 for smaller volumes

**Cleanup Threshold (`--cleanup-threshold` / `CLEANUP_THRESHOLD`)**
- Default: `85.0` (%)
- Triggers deletion when disk usage reaches this percentage
- Recommendation: Set based on your storage size and growth rate
  - Large volumes (>10TB): Can use higher threshold (85-90%)
  - Smaller volumes (<1TB): May need lower threshold (70-80%)

**Target Threshold (`--target-threshold` / `TARGET_THRESHOLD`)**
- Default: `70.0` (%)
- Stops deletion when disk usage drops to this percentage
- Recommendation: Keep 10-20% gap from cleanup threshold for hysteresis

**File Access Time Threshold (`FILE_ACCESS_TIME_THRESHOLD_MINUTES`)**
- Default: `60` minutes
- Files accessed within this time are protected from deletion
- Recommendation: Adjust based on your workload patterns
  - Active workloads: 30-60 minutes
  - Batch workloads: 120-180 minutes

### Example Configurations

**High-throughput setup (large volume, many files):**
```bash
./deploy.sh my-cache --num-crawlers=16 --cleanup-threshold=90.0 --target-threshold=75.0
```

**Conservative setup (smaller volume, protect recent cache):**
```bash
./deploy.sh my-cache --num-crawlers=4 --cleanup-threshold=75.0 --target-threshold=60.0
```

**For complete configuration reference, see [README.md](README.md#configuration).**


## Verify Deployment

```bash
# Check pod status
kubectl get pods -n <namespace> | grep evictor

# View logs
kubectl logs -f deployment/pvc-evictor -n <namespace>

# Check PVC usage
kubectl exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache
```

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

**No files being deleted?**
- Check PVC usage: `kubectl exec -it deployment/pvc-evictor -n <namespace> -- df -h /kv-cache`
- Check logs for `DELETION_START` events
- Verify thresholds are appropriate

**Files not being discovered?**
- Verify `CACHE_DIRECTORY` matches actual cache path (default: `kv/model-cache/models`)
- Check logs for crawler heartbeats: `kubectl logs deployment/pvc-evictor -n <namespace> | grep Heartbeat`

For more details, see [README.md](README.md).

