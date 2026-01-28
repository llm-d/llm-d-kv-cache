FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy llmd_fs_backend Python package (for FileMapper)
# Only copy the Python package, not the C++/CUDA build infrastructure
# Note: Build from kv_connectors directory: podman build -f pvc_evictor/Dockerfile -t pvc-evictor:tag .
COPY llmd_fs_backend/llmd_fs_backend/ /app/llmd_fs_backend/

# Copy PVC Evictor files
COPY pvc_evictor/config.py pvc_evictor/evictor.py ./
COPY pvc_evictor/utils/ ./utils/
COPY pvc_evictor/processes/ ./processes/

# Set PYTHONPATH to enable imports from llmd_fs_backend
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run the evictor
CMD ["python3", "evictor.py"]

