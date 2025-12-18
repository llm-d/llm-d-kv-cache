#!/bin/bash
# Deployment helper script for PVC Evictor
# Usage: ./deploy.sh <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>] [--log-level=<level>]
# Note: All arguments must use --arg=value format (e.g., --namespace=e5)
# Note: Docker image must be built and pushed to ghcr.io/guygir/pvc-evictor:latest before deployment

set -e

# Default values
FS_GROUP=""
SELINUX_LEVEL=""
RUNAS_USER=""
NUM_CRAWLERS="8"
CLEANUP_THRESHOLD="85.0"
TARGET_THRESHOLD="70.0"
LOG_LEVEL="INFO"
NAMESPACE=""

# PVC name is required (first positional argument)
if [ -z "$1" ] || [[ "$1" == --* ]]; then
    echo "Error: PVC name is required as the first argument"
    echo "Usage: $0 <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>] [--log-level=<level>]"
    echo "Note: All arguments must use --arg=value format (e.g., --namespace=e5)"
    exit 1
fi

PVC_NAME="$1"
shift  # Remove PVC_NAME from arguments

# Parse named arguments (only --arg=value format supported)
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace=*)
            NAMESPACE="${1#*=}"
            shift
            ;;
        --fsgroup=*)
            FS_GROUP="${1#*=}"
            shift
            ;;
        --selinux-level=*)
            SELINUX_LEVEL="${1#*=}"
            shift
            ;;
        --runasuser=*)
            RUNAS_USER="${1#*=}"
            shift
            ;;
        --num-crawlers=*)
            NUM_CRAWLERS="${1#*=}"
            shift
            ;;
        --cleanup-threshold=*)
            CLEANUP_THRESHOLD="${1#*=}"
            shift
            ;;
        --target-threshold=*)
            TARGET_THRESHOLD="${1#*=}"
            shift
            ;;
        --log-level=*)
            LOG_LEVEL="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 <pvc-name> [--namespace=<namespace>] [--fsgroup=<fsgroup>] [--selinux-level=<level>] [--runasuser=<user>] [--num-crawlers=<n>] [--cleanup-threshold=<%>] [--target-threshold=<%>] [--log-level=<level>]"
            echo "Note: All arguments must use --arg=value format (e.g., --namespace=e5)"
            exit 1
            ;;
    esac
done

# Validate num-crawlers (must be 1, 2, 4, 8, or 16)
valid_crawler_counts=(1 2 4 8 16)
if [[ ! " ${valid_crawler_counts[@]} " =~ " ${NUM_CRAWLERS} " ]]; then
    echo "Error: --num-crawlers must be one of: ${valid_crawler_counts[*]}, got: ${NUM_CRAWLERS}"
    exit 1
fi

# Auto-detect namespace if not provided
if [ -z "$NAMESPACE" ]; then
    if command -v kubectl >/dev/null 2>&1; then
        NAMESPACE=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || echo "")
    fi
    
    if [ -z "$NAMESPACE" ]; then
        echo "Error: Namespace is required and could not be auto-detected"
        echo "Please provide namespace with --namespace=<namespace>"
        exit 1
    fi
    echo "Auto-detected namespace: $NAMESPACE"
fi

# Auto-detect security context values if not provided
# Strategy: 1) Try existing evictor deployment, 2) Try evictor pod, 3) Try any pod with security context, 4) Use defaults
AUTO_DETECTED_FS_GROUP=""
AUTO_DETECTED_SELINUX_LEVEL=""
AUTO_DETECTED_RUNAS_USER=""

if [ -z "$FS_GROUP" ] && [ -z "$SELINUX_LEVEL" ] && [ -z "$RUNAS_USER" ]; then
    # Only auto-detect if all three are empty (user didn't provide any)
    
    # First, try to get values from existing evictor deployment (most reliable for redeployments)
    if command -v kubectl >/dev/null 2>&1; then
        if kubectl get deployment pvc-evictor -n "$NAMESPACE" >/dev/null 2>&1; then
            AUTO_DETECTED_FS_GROUP=$(kubectl get deployment pvc-evictor -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.securityContext.fsGroup}' 2>/dev/null || echo "")
            AUTO_DETECTED_SELINUX_LEVEL=$(kubectl get deployment pvc-evictor -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.securityContext.seLinuxOptions.level}' 2>/dev/null || echo "")
            AUTO_DETECTED_RUNAS_USER=$(kubectl get deployment pvc-evictor -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].securityContext.runAsUser}' 2>/dev/null || echo "")
        fi
    fi
    
    # If deployment didn't have all values, try evictor pod
    if [ -z "$AUTO_DETECTED_FS_GROUP" ] || [ -z "$AUTO_DETECTED_SELINUX_LEVEL" ] || [ -z "$AUTO_DETECTED_RUNAS_USER" ]; then
        DETECT_POD=""
        if command -v kubectl >/dev/null 2>&1; then
            DETECT_POD=$(kubectl get pod -l app=pvc-evictor -n "$NAMESPACE" --no-headers 2>/dev/null | head -1 | awk '{print $1}')
        fi
        
        if [ -n "$DETECT_POD" ]; then
            if [ -z "$AUTO_DETECTED_FS_GROUP" ]; then
                AUTO_DETECTED_FS_GROUP=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.fsGroup}' 2>/dev/null || echo "")
            fi
            if [ -z "$AUTO_DETECTED_SELINUX_LEVEL" ]; then
                AUTO_DETECTED_SELINUX_LEVEL=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.seLinuxOptions.level}' 2>/dev/null || echo "")
            fi
            if [ -z "$AUTO_DETECTED_RUNAS_USER" ]; then
                AUTO_DETECTED_RUNAS_USER=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.runAsUser}' 2>/dev/null || echo "")
            fi
        fi
    fi
    
    # Try any pod in namespace only if we still don't have all values and runAsUser is not 0 (root)
    if [ -z "$AUTO_DETECTED_FS_GROUP" ] || [ -z "$AUTO_DETECTED_SELINUX_LEVEL" ] || [ -z "$AUTO_DETECTED_RUNAS_USER" ] || [ "$AUTO_DETECTED_RUNAS_USER" = "0" ]; then
        DETECT_POD=""
        if command -v kubectl >/dev/null 2>&1; then
            # Get pods that have security context set (prefer pods with fsGroup)
            DETECT_POD=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | head -1 | awk '{print $1}')
        fi
        
        if [ -n "$DETECT_POD" ]; then
            TEMP_FS_GROUP=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.fsGroup}' 2>/dev/null || echo "")
            TEMP_SELINUX_LEVEL=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.seLinuxOptions.level}' 2>/dev/null || echo "")
            TEMP_RUNAS_USER=$(kubectl get pod "$DETECT_POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.runAsUser}' 2>/dev/null || echo "")
            
            # Only use if runAsUser is not 0 (root) and we have meaningful values
            if [ -n "$TEMP_FS_GROUP" ] && [ "$TEMP_RUNAS_USER" != "0" ] && [ -n "$TEMP_RUNAS_USER" ]; then
                if [ -z "$AUTO_DETECTED_FS_GROUP" ]; then
                    AUTO_DETECTED_FS_GROUP="$TEMP_FS_GROUP"
                fi
                if [ -z "$AUTO_DETECTED_SELINUX_LEVEL" ] && [ -n "$TEMP_SELINUX_LEVEL" ]; then
                    AUTO_DETECTED_SELINUX_LEVEL="$TEMP_SELINUX_LEVEL"
                fi
                if [ "$AUTO_DETECTED_RUNAS_USER" = "0" ] || [ -z "$AUTO_DETECTED_RUNAS_USER" ]; then
                    AUTO_DETECTED_RUNAS_USER="$TEMP_RUNAS_USER"
                fi
            fi
        fi
    fi
    
    # Use auto-detected values if found, otherwise keep defaults
    SOURCE="defaults"
    if [ -n "$AUTO_DETECTED_FS_GROUP" ] || [ -n "$AUTO_DETECTED_SELINUX_LEVEL" ] || ([ -n "$AUTO_DETECTED_RUNAS_USER" ] && [ "$AUTO_DETECTED_RUNAS_USER" != "0" ]); then
        SOURCE="existing pod/deployment"
    fi
    
    if [ -n "$AUTO_DETECTED_FS_GROUP" ]; then
        FS_GROUP="$AUTO_DETECTED_FS_GROUP"
        echo "Auto-detected fsGroup ($SOURCE): $FS_GROUP"
    fi
    if [ -n "$AUTO_DETECTED_SELINUX_LEVEL" ]; then
        SELINUX_LEVEL="$AUTO_DETECTED_SELINUX_LEVEL"
        echo "Auto-detected seLinuxLevel ($SOURCE): $SELINUX_LEVEL"
    fi
    if [ -n "$AUTO_DETECTED_RUNAS_USER" ] && [ "$AUTO_DETECTED_RUNAS_USER" != "0" ]; then
        RUNAS_USER="$AUTO_DETECTED_RUNAS_USER"
        echo "Auto-detected runAsUser ($SOURCE): $RUNAS_USER"
    fi
fi

echo "=== PVC Evictor Deployment ==="
echo "Namespace: $NAMESPACE"
echo "PVC Name: $PVC_NAME"
echo "FS Group: $FS_GROUP"
echo "SELinux Level: $SELINUX_LEVEL"
echo "RunAs User: $RUNAS_USER"
echo "Num Crawlers: $NUM_CRAWLERS"
echo "Cleanup Threshold: $CLEANUP_THRESHOLD%"
echo "Target Threshold: $TARGET_THRESHOLD%"
echo "Log Level: $LOG_LEVEL"
echo ""

# Create temporary deployment file with substitutions
# Note: Only replaces placeholders (e.g., <your_namespace>). If values are manually set in the YAML,
# they will NOT be replaced - only placeholders are substituted.
TEMP_DEPLOYMENT=$(mktemp)
sed -e "s/namespace: <your_namespace>/namespace: $NAMESPACE/" \
    -e "s/claimName: <your_pvc_name>/claimName: $PVC_NAME/" \
    -e "s/fsGroup: <your_fsgroup>/fsGroup: $FS_GROUP/" \
    -e "s/level: \"<your_selinux_level>\"/level: \"$SELINUX_LEVEL\"/" \
    -e "s/runAsUser: <your_runasuser>/runAsUser: $RUNAS_USER/" \
    -e "s/value: \"8\"  # Number of crawler processes (valid: 1, 2, 4, 8, 16). Default: 8/value: \"$NUM_CRAWLERS\"  # Number of crawler processes (valid: 1, 2, 4, 8, 16). Default: 8/" \
    -e "s/value: \"85.0\"  # Trigger deletion above this %/value: \"$CLEANUP_THRESHOLD\"  # Trigger deletion above this %/" \
    -e "s/value: \"70.0\"  # Stop deletion below this %/value: \"$TARGET_THRESHOLD\"  # Stop deletion below this %/" \
    -e "s/value: \"INFO\"  # DEBUG, INFO, WARNING, ERROR/value: \"$LOG_LEVEL\"  # DEBUG, INFO, WARNING, ERROR/" \
    deployment_evictor.yaml > "$TEMP_DEPLOYMENT"

echo ""
echo "Deploying evictor pod..."
kubectl apply -f "$TEMP_DEPLOYMENT" -n "$NAMESPACE"
rm "$TEMP_DEPLOYMENT"

echo "Deployment applied"
echo ""
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod -l app=pvc-evictor -n "$NAMESPACE" --timeout=120s || true

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Check pod status:"
echo "  kubectl get pods -n $NAMESPACE | grep evictor"
echo ""
echo "View logs:"
echo "  kubectl logs -f deployment/pvc-evictor -n $NAMESPACE"
echo ""
echo "Check PVC usage:"
echo "  kubectl exec -it deployment/pvc-evictor -n $NAMESPACE -- df -h /kv-cache"
