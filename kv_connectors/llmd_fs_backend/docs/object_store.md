# Object Storage Guide

Object storage enables usage of an object storage backend for caching kv data. 

The object store support is built on top of NIXL.

## Requirements

- NIXL library (will be installed by the python wheel during the build)
- an object store (S3, Ceph, Noobaa, MinIO, etc.)

## Build

Follow the standard [build instructions](../README.md#installation).

## Configuration

Add the object store configuration and specify the object store backend in  `kv_connector_extra_config` in your vLLM config:

```yaml
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SharedStorageOffloadingSpec",
    "spec_module_path": "llmd_fs_backend.spec",
    "shared_storage_path": "/mnt/nvme/kv-cache/",
    "block_size": 256,
    "threads_per_gpu": "64",
    "bucket": "testing1", 
    "scheme": "https",
    "endpoint_override": "172.30.228.75:9000", 
    "access_key": "minioadmin", 
    "secret_key": "minioadmin",
    "ca_bundle": "/root/tls.crt",
    "backend": "OBJ",
  }
}'
```

## Limitations

Note that currently the gpu block size (set as block_size in vllm) must match the object store blocksize (set as block_size in kv_connector_extra_config).
