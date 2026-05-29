FROM python:3.12-slim

WORKDIR /app

# llmd_fs_backend: required when STORAGE_EVENTS_ENDPOINT is set (#605 BlockRemoved).
# Crawler discovery is path-only (#611) and does not import FileMapper/vLLM.
# Build from kv_connectors/: podman build -f pvc_evictor/Dockerfile -t pvc-evictor:tag .
COPY llmd_fs_backend/llmd_fs_backend/ /app/llmd_fs_backend/

COPY pvc_evictor/config.py pvc_evictor/evictor.py ./
COPY pvc_evictor/utils/ ./utils/
COPY pvc_evictor/processes/ ./processes/

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python3", "evictor.py"]
