FROM python:3.12-slim

# Set working directory first
WORKDIR /app

# Copy all Python files and directories
COPY config.py evictor.py ./
COPY utils/ ./utils/
COPY processes/ ./processes/

# Run the evictor
CMD ["python3", "evictor.py"]

