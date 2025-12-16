# KV-Cache Setup Guide

This guide provides a complete walkthrough for setting up and testing the example llm-d-kv-cache system. You will deploy a vLLM with LMCache and Redis, then run an example application that demonstrates KV cache indexing capabilities.

By following this guide, you will:

1. **Deploy the Infrastructure**: Use Helm to set up:
   - vLLM nodes with LMCache CPU offloading (4 replicas) serving Llama 3.1 8B Instruct model
   - Redis server
2. **Test with Example Application**: Run a Go application that:
   - Connects to your deployed vLLM and Redis infrastructure,
   - Demonstrates KV cache indexing by processing a sample prompt

The demonstrated KV-cache indexer is utilized for AI-aware routing to accelerate inference across the system through minimizing redundant computation.

## vLLM Deployment

The llm-d-kv-cache repository includes a Helm chart for deploying vLLM with CPU offloading (LMCache) and KV-events indexing (Redis). This section describes how to use this Helm chart for a complete deployment.

*Note*: Ensure that the Kubernetes node designated for running vLLM supports GPU workloads.

### Prerequisites

- Kubernetes cluster with GPU support
- Helm 3.x
- HuggingFace token for accessing models
- kubectl configured to access your cluster

### Installation

1. Set environment variables:

```bash
export HF_TOKEN=<your-huggingface-token>
export NAMESPACE=<your-namespace>
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export VLLM_POOLLABEL="vllm-model-pool"
```

> Note that both the Helm deployment and the example application use the same `MODEL_NAME` environment variable,
> ensuring alignment between the vLLM deployment configuration and the KV cache indexer.
> Set this variable once during initial setup and both components will use the same model configuration.

2. Deploy using Helm:

```bash
helm upgrade --install vllm-stack ./vllm-setup-helm \
  --namespace $NAMESPACE \
  --create-namespace \
  --set secret.create=true \
  --set secret.hfTokenValue=$HF_TOKEN \
  --set vllm.model.name=$MODEL_NAME \
  --set vllm.poolLabelValue=$VLLM_POOLLABEL \
  -f ./vllm-setup-helm/values.yaml
```

**Note:**

- Adjust the resource and limit allocations for vLLM and Redis in `values.yaml` to match your cluster's capacity.
- By default, the chart uses a `PersistentVolume` to cache the model. To disable this, set `.persistence.enabled` to `false`.

3. Verify the deployment:

```bash
kubectl get deployments -n $NAMESPACE
```

You should see:

- vLLM pods (default: 4 replicas)
- Redis lookup server pod

### Configuration Options

The Helm chart supports various configuration options. See [values.yaml](../../../vllm-setup-helm/values.yaml) for all available options.

Key configuration parameters:

- `vllm.model.name`: The HuggingFace model to use (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `vllm.replicaCount`: Number of vLLM replicas (default: 4)
- `vllm.poolLabelValue`: Label value for the inference pool (used by scheduler)
- `redis.enabled`: Whether to deploy Redis for KV cache indexing (default: true)
- `persistence.enabled`: Enable persistent storage for model cache (default: true)
- `secret.create`: Create HuggingFace token secret (default: true)

## Using the KV Cache Indexer Example

### Prerequisites

Ensure you have a running deployment with vLLM and Redis as described above.

### Running the Example

The vLLM node can be tested with the prompt found in `examples/kv_cache_index/main.go`.

Then, set the required environment variables and run example:

```bash
export HF_TOKEN=<token>
export REDIS_ADDR=<redis://$user:$pass@localhost:6379/$db> # optional, defaults to localhost:6379
export MODEL_NAME=<model_name_used_in_vllm_deployment> # optional, defaults to meta-llama/Llama-3.1-8B-Instruct

make run-example kv_cache_index
```

Environment variables:

- `HF_TOKEN` (required): HuggingFace access token
- `REDIS_ADDR` (optional): Redis address; defaults to localhost:6379.
- `MODEL_NAME` (optional): The model name used in vLLM deployment; defaults to meta-llama/Llama-3.1-8B-Instruct. Use the same value you set during Helm deployment.
