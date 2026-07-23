# Model Caching in Tokenizer Service

The UDS tokenizer service implements a caching mechanism to improve performance and reduce network dependencies. When a model is requested, the service follows this priority order:

1. **Local Cache**: Check if the model files already exist in the directory configured by `TOKENIZERS_DIR`
2. **Remote Download**: If not cached, download from ModelScope or Hugging Face (based on `USE_MODELSCOPE` environment variable)
3. **Local Storage**: Save downloaded files to the configured tokenizer cache directory for future use

When `TOKENIZERS_DIR` is unset, local source checkouts default to the `tokenizers/` directory shown below. The container image sets `TOKENIZERS_DIR=/tmp/tokenizers` so restricted Kubernetes runtimes can write the cache.

## Directory Structure

The models are organized by provider and model name:
```
tokenizers/
├── Qwen/
│   └── Qwen3-8B/
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
└── deepseek-ai/
    └── DeepSeek-R1-Distill-Qwen-1.5B/
        ├── config.json
        ├── generation_config.json
        ├── merges.txt
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
```

## Pre-populating the Cache

You can pre-download model files to avoid runtime downloads:

### Using ModelScope CLI

```bash
# Install modelscope
pip install modelscope

# Download tokenizer files for a specific model
python -c "
from modelscope import snapshot_download
snapshot_download(
    'Qwen/Qwen3-8B',
    local_dir='./tokenizers/Qwen/Qwen3-8B',
    allow_patterns=[
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
        'config.json',
        'generation_config.json'
    ]
)
"
```

### Using Hugging Face CLI

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download tokenizer files
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-8B',
    local_dir='./tokenizers/Qwen/Qwen3-8B',
    allow_patterns=[
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
        'config.json',
        'generation_config.json'
    ]
)
"
```

## Kubernetes Deployment with Model Caching

In Kubernetes environments, you can use a PersistentVolume to store model files across pod restarts:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-cache-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  hostPath:
    path: /data/model-cache
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: tokenizer-service
        image: your-tokenizer-service:latest
        volumeMounts:
        - name: model-cache
          mountPath: /tmp/tokenizers
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
```
