FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy llmd_fs_backend Python package (for FileMapper)
# Only copy the Python package, not the C++/CUDA build infrastructure
COPY ../llmd_fs_backend/llmd_fs_backend/ /app/llmd_fs_backend/

# Copy PVC Evictor files
COPY config.py evictor.py ./
COPY utils/ ./utils/
COPY processes/ ./processes/

# Set PYTHONPATH to enable imports from llmd_fs_backend
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run the evictor
CMD ["python3", "evictor.py"]

