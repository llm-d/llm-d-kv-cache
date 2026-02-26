#!/bin/bash
# vLLM installation script for macOS/Apple Silicon
# https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html

set -e

# 1. Skip if vllm is already installed
PYTHON_BIN=$(which python3 || which python)
if $PYTHON_BIN -c "import vllm" &> /dev/null; then
    echo "[SKIP] vllm is already installed. Exiting."
    exit 0
fi

# 2. Check for macOS/Apple Silicon
ARCH=$(uname -m)
OS=$(uname)

if [[ "$ARCH" != "arm64" || "$OS" != "Darwin" ]]; then
    echo "[ERROR] This script is for Apple Silicon (arm64) on macOS only."
    echo "Current system: $OS $ARCH"
    exit 1
fi

VLLM_REPO=https://github.com/vllm-project/vllm.git
VLLM_TAG=v0.14.0

# 3. Check and install Python requirements (runtime)
REQUIRED_PKGS=(cmake wheel packaging ninja setuptools-scm numpy)
TO_INSTALL=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    # Try pip show, then fallback to checking if the binary exists in PATH
    if ! $PYTHON_BIN -m pip show "$pkg" &> /dev/null; then
        # Some packages like cmake, ninja may be installed as binaries
        if ! command -v "$pkg" &> /dev/null; then
            TO_INSTALL+=("$pkg")
        fi
    fi
done
$PYTHON_BIN -m pip install --upgrade pip
if [[ ${#TO_INSTALL[@]} -gt 0 ]]; then
    $PYTHON_BIN -m pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
else
    echo "[SKIP] python runtime packages already installed."
fi

# 4. Clone vllm source and install requirements/cpu.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC_DIR="$SCRIPT_DIR/vllm_source"
if [ ! -d "$VLLM_SRC_DIR" ]; then
    git clone $VLLM_REPO "$VLLM_SRC_DIR"
fi
cd "$VLLM_SRC_DIR"
git fetch --tags
git checkout tags/$VLLM_TAG

$PYTHON_BIN -m pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Build and install vLLM for Apple Silicon
$PYTHON_BIN -m pip install -e .

echo "✅ vLLM CPU build and installation completed for Apple Silicon."