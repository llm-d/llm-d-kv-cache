# GPUDirect Storage (GDS) Guide

GPU Direct Storage (GDS) enables direct DMA transfers between GPU memory and NVMe/NVMe-oF storage,
bypassing the CPU staging buffer entirely. This can significantly improve throughput for KV-cache offloading.

GDS uses the **cuFile** library by default for all GPU↔storage transfers. cuFile behavior (I/O threads,
bounce buffer sizes, RDMA settings, filesystem-specific tuning) can be customized via a `cufile_rdma.json`
config file — see [Tuning cuFile](#tuning-cufile-cufilejson) for an example.

## Requirements

- NVIDIA GPU with GDS support (`cudaDevAttrGPUDirectRDMASupported`)
- cuFile library installed (`nvidia-fs` kernel module + `libcufile`)
- Filesystem that supports O_DIRECT (e.g., ext4, xfs, NVMe-oF — **not** most NFS/FUSE mounts)
- cuFile detected at build time (see [Build with GDS support](#build-with-gds-support))

> **Default**: GDS is **disabled** by default. The connector works without GDS using CPU staging buffers.
> Only enable GDS if your hardware and filesystem support it.

## Build with GDS support

GDS support is detected automatically at build time. If `libcufile.so` is present on the system,
the extension is compiled with `USE_CUFILE` and linked against `libcufile`:

```bash
pip install -e .
```

To install cuFile: `apt-get install -y cuda-cufile-12-9` (adjust version to match your CUDA install).

## GDS modes

| `gds_mode` value | Read | Write | Staging buffer |
|---|---|---|---|
| `disabled` (default) | CPU staging | CPU staging | full size |
| `read_only` | GDS direct | CPU staging | full size |
| `write_only` | CPU staging | GDS direct | full size |
| `read_write` | GDS direct | GDS direct | **none** (16 MB floor) |
| `bb_read_only` | GDS + Bounce Buffer | CPU staging | full size |
| `bb_write_only` | CPU staging | GDS + Bounce Buffer | full size |
| `bb_read_write` | GDS + Bounce Buffer | GDS + Bounce Buffer | **none** (16 MB floor) |

**Bounce Buffer (BB) modes** use GDS with an intermediate GPU-side buffer.
Use BB modes when the filesystem does not support full O_DIRECT GDS but you still want GPU-accelerated I/O.

## Configuration

Add `gds_mode` to `kv_connector_extra_config` in your vLLM config:

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
    "gds_mode": "read_write"
  }
}'
```

If you are unsure which mode to use, start with `read_write` for NVMe local disks or `bb_read_write` for shared storage.

## Verify GDS is active

Set `STORAGE_LOG_LEVEL=info` (the default) and look for these lines at startup:

```
GdsFileIO: GPUDirect Storage (GDS) enabled
```

If GDS initialization fails, the connector automatically falls back to CPU staging and logs:

```
StorageOffloadEngine: GDS initialization failed, falling back to CPU_BUFFER_STAGE for both READ and WRITE
StorageOffloadEngine: READ=CPU WRITE=CPU
```

For more detail during startup, use `STORAGE_LOG_LEVEL=debug`:

```bash
STORAGE_LOG_LEVEL=debug vllm serve ...
```

## Tuning cuFile (`cufile_rdma.json`)

cuFile reads its configuration from a JSON file. By default it looks in `/etc/cufile.json`.
To use a custom path:

```bash
export CUFILE_ENV_PATH_JSON=/path/to/cufile_rdma.json
```

An example config  is provided at [`docs/cufile_rdma.json`](./cufile_rdma.json).
All settings are documented inline in that file. For RDMA storage, fill in your NIC IPs:

```json
"rdma_dev_addr_list": [ "<RDMA_NIC_IP_1>", "<RDMA_NIC_IP_2>" ]
```

To find your RDMA NIC IPs:
```bash
ibdev2netdev    # maps InfiniBand devices to network interfaces
ip addr show    # note the IPs of those interfaces
```

> Your local `cufile_rdma.json` (with deployment-specific IPs) should **not** be committed to version control.



## Troubleshooting

### `cuFileDriverOpen` fails

- Check that the `nvidia-fs` kernel module is loaded: `lsmod | grep nvidia_fs`
- Install the cuFile package: `apt-get install -y cuda-cufile-12-9`

### GDS not supported on this GPU

- GDS requires `cudaDevAttrGPUDirectRDMASupported = 1`
- Check: `nvidia-smi --query-gpu=name --format=csv` — A100, H100, and most data-center GPUs support it
- Consumer GPUs (RTX series) typically do not

### Filesystem does not support O_DIRECT

- NFS, FUSE-based mounts, and tmpfs do not support O_DIRECT
- Use `bb_read_write` (Bounce Buffer mode) as a fallback for these filesystems
- For full GDS performance, use a local NVMe or NVMe-oF mount

### GDS falls back silently to CPU

- The connector falls back automatically if GDS init fails
- Always check startup logs for `READ=GDS` / `READ=CPU` to confirm the active mode

