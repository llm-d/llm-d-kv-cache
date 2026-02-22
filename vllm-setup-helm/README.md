# vLLM Deployment Chart

This Helm chart assists with deploying vLLMs with optionally:
- KVEvents publishing and a demo KV Cache indexing deployment
- LMCache for KV-cache offloading and deprecated Redis indexing

## Prerequisites

- A Kubernetes cluster with NVIDIA GPU resources.
- Helm 3 installed.
- `kubectl` configured to connect to your cluster.

## Installation

1.  **Set Environment Variables**

    If you are using a private model from Hugging Face, you will need an access token.

    ```bash
    export HF_TOKEN="your-huggingface-token"
    export NAMESPACE="default" # Or your desired namespace
    ```

2.  **Deploy the Chart**

    Install the chart with a release name (e.g., `my-vllm`):

    ```bash
    helm upgrade --install my-vllm ./vllm-setup-helm \
      --namespace $NAMESPACE \
      --set secret.hfTokenValue=$HF_TOKEN 
    ```

    You can customize the deployment by creating your own `values.yaml` file or by using `--set` flags for other parameters.

## Configuration

The most important configuration parameters are listed below. For a full list of options, see the [`values.yaml`](./values.yaml) file.

| Parameter | Description                                                        | Default                            |
| --- |--------------------------------------------------------------------|------------------------------------|
| `vllm.model.name` | The Hugging Face model to deploy.                                  | `meta-llama/Llama-3.1-8B-Instruct` |
| `vllm.replicaCount` | Number of vLLM replicas.                                           | `1`                                |
| `vllm.resources.limits` | GPU and other resource limits for the vLLM container.              | `nvidia.com/gpu: '1'`              |
| `lmcache.enabled` | Enable LMCache for KV-cache offloading.                            | `false`                            |
| `lmcache.redis.enabled` | Deploy a Redis instance for KV-cache indexing through LMCache.     | `false`                            |
| `kvCacheManager.enabled` | Deploy the KV Cache for event indexing DEMO.                | `true`                             |
| `kvCacheManager.externalTokenizer.enabled` | Enable external tokenizer as a sidecar for KV Cache.        | `true`                             |
| `persistence.enabled` | Enable persistent storage for model weights to speed up restarts.  | `true`                             |
| `secret.create` | Set to `true` to automatically create the secret for `hfTokenValue`. | `true`                             |
| `secret.hfTokenValue` | The Hugging Face token. Required if `secret.create` is `true`.     | `""`                               |

### External Tokenizer on UDS

The KV Cache can optionally use an external tokenizer deployed as a sidecar container. This external tokenizer is implemented in Python using the same transformers library as vLLM for tokenization logic, which helps achieve better compatibility, especially for models like the DeepSeek series. See [the example UDS tokenizer service](../services/uds_tokenizer/README.md) for more details.

To enable the external tokenizer, set `kvCacheManager.externalTokenizer.enabled` to `true` in your `values.yaml`. The external tokenizer will be deployed as a sidecar container alongside the KV Cache.

You can customize the external tokenizer image and resources through the following parameters:

- `kvCacheManager.externalTokenizer.image.repository`: The repository for the tokenizer image
- `kvCacheManager.externalTokenizer.image.tag`: The tag for the tokenizer image
- `kvCacheManager.externalTokenizer.image.pullPolicy`: The image pull policy
- `kvCacheManager.externalTokenizer.resources`: Resource requests and limits for the tokenizer container

## Cleanup

To uninstall the chart and clean up all associated resources, run the following command, replacing `my-vllm` with your release name:

```bash
helm uninstall my-vllm --namespace $NAMESPACE
```