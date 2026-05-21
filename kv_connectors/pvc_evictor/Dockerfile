FROM python:3.12-slim

WORKDIR /app

# Copy PVC Evictor files (build from kv_connectors/: podman build -f pvc_evictor/Dockerfile -t pvc-evictor:tag .)
COPY pvc_evictor/config.py pvc_evictor/evictor.py ./
COPY pvc_evictor/utils/ ./utils/
COPY pvc_evictor/processes/ ./processes/

CMD ["python3", "evictor.py"]
