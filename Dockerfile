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

FROM python:3.12-slim AS python-builder

ARG TARGETOS=linux
ARG TARGETARCH=amd64

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY Makefile Makefile
COPY pkg/preprocessing/chat_completions/ pkg/preprocessing/chat_completions/
RUN TARGETOS=${TARGETOS} TARGETARCH=${TARGETARCH} make install-python-deps

# Build Stage: using Go 1.24.1 image
FROM quay.io/projectquay/golang:1.24 AS builder
ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

# Install system-level dependencies first. This layer is very stable.
USER root
# Install EPEL repository directly and then ZeroMQ, as epel-release is not in default repos.
# Install all necessary dependencies including Python 3.12 for chat-completions templating.
# The builder is based on UBI8, so we need epel-release-8.
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm' && \
    dnf install -y gcc-c++ libstdc++ libstdc++-devel clang zeromq-devel pkgconfig python3.12-devel python3.12-pip && \
    dnf clean all

# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the source code.
COPY . .

# Copy this project's own Python source code into the final image
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions /workspace/pkg/preprocessing/chat_completions
RUN make setup-venv
COPY --from=python-builder /workspace/build/venv/lib/python3.12/site-packages /workspace/build/venv/lib/python3.12/site-packages

# Set the PYTHONPATH. This mirrors the Makefile's export, ensuring both this project's
# Python code and the installed libraries (site-packages) are found at runtime.
ENV PYTHONPATH=/workspace/pkg/preprocessing/chat_completions:/workspace/build/venv/lib/python3.12/site-packages
RUN python3.12 -c "import tokenizer_wrapper"

RUN make build

# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM registry.access.redhat.com/ubi9/ubi:latest
WORKDIR /
# Install zeromq runtime library needed by the manager.
# The final image is UBI9, so we need epel-release-9.
USER root
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm' && \
    dnf install -y zeromq libxcrypt-compat python3.12 python3.12-pip && \
    dnf clean all

# Copy this project's own Python source code into the final image
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions /app/pkg/preprocessing/chat_completions
COPY --from=python-builder /workspace/build/venv/lib/python3.12/site-packages /workspace/build/venv/lib/python3.12/site-packages

# Set the PYTHONPATH. This mirrors the Makefile's export, ensuring both this project's
# Python code and the installed libraries (site-packages) are found at runtime.
ENV PYTHONPATH=/app/pkg/preprocessing/chat_completions:/workspace/build/venv/lib/python3.12/site-packages
RUN python3.12 -c "import tokenizer_wrapper"

# Copy the compiled Go application
COPY --from=builder /workspace/bin/llm-d-kv-cache /app/kv-cache-manager
USER 65532:65532

# Set the entrypoint to the kv-cache-manager binary
ENTRYPOINT ["/app/kv-cache-manager"]
