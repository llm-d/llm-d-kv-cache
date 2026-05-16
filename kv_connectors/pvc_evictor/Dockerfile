FROM us-central1-artifactregistry.gcr.io/gke-release/gke-release/lustre-csi-driver:v0.3.17-gke.1 AS lustre-client

FROM python:3.12-slim

# Install required system libraries for lfs binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreadline8 libjson-c5 libyaml-0-2 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy lfs binary and Lustre libraries from lustre-client stage
COPY --from=lustre-client /usr/bin/lfs /usr/bin/lfs
COPY --from=lustre-client /lib/liblustreapi.so.1 /lib/liblustreapi.so.1
COPY --from=lustre-client /lib/liblnetconfig.so.4 /lib/liblnetconfig.so.4

# Set working directory
WORKDIR /app

# Copy llmd_fs_backend Python package (for FileMapper)
COPY llmd_fs_backend/llmd_fs_backend/ /app/llmd_fs_backend/

# Copy PVC Evictor files
COPY pvc_evictor/config.py pvc_evictor/evictor.py pvc_evictor/lfs_evictor.py ./
COPY pvc_evictor/utils/ ./utils/
COPY pvc_evictor/processes/ ./processes/

# Set PYTHONPATH to enable imports from llmd_fs_backend
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run the LFS evictor
CMD ["python3", "lfs_evictor.py"]

