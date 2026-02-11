# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build stage
FROM --platform=$TARGETPLATFORM python:3.10-slim as builder

# Set build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM --platform=$TARGETPLATFORM python:3.10-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/cache/apt/

# Copy installed dependencies from build stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy executables from build stage
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy project files into the image
COPY run_grpc_server.py /app/
COPY requirements.txt /app/requirements.txt
COPY tokenizer_grpc_service.py /app/tokenizer_grpc_service.py
COPY models/ /app/models/
COPY utils/ /app/utils/
COPY tokenizer_service/ /app/tokenizer_service/
COPY tokenizerpb/ /app/tokenizerpb/

# Create directory for UDS socket
RUN mkdir -p /tmp/tokenizer && chown 65532:65532 /tmp/tokenizer

# Create model cache directories and set permissions
RUN mkdir -p /app/models && chown -R 65532:65532 /app/models
# Create and set permissions for ModelScope directory
RUN mkdir -p /.modelscope && chown -R 65532:65532 /.modelscope
# Create and set permissions for Hugging Face cache directory
RUN mkdir -p /.cache && chown -R 65532:65532 /.cache

# Switch to non-root user
USER 65532:65532

# Startup command: run direct gRPC server
CMD ["python", "/app/run_grpc_server.py"]