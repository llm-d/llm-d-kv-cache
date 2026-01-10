#!/bin/bash
# https://docs.vllm.ai/en/v0.8.4/getting_started/installation/cpu.html

set -e

# 1. Skip if vllm is already installed
PYTHON_BIN=$(which python3 || which python)
if $PYTHON_BIN -c "import vllm" &> /dev/null; then
    echo "[SKIP] vllm is already installed. Exiting."
    exit 0
fi

# 2. Architecture check (Only Intel/AMD x86, ARM AArch64, Apple Silicon supported)
ARCH=$(uname -m)
OS=$(uname)
VLLM_REPO=https://github.com/vllm-project/vllm.git
VLLM_TAG=v0.11.1

if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_TYPE="x86_64"
elif [[ "$ARCH" == "aarch64" ]]; then
    ARCH_TYPE="aarch64"
elif [[ "$ARCH" == "arm64" && "$OS" == "Darwin" ]]; then
    ARCH_TYPE="apple_silicon"
else
    echo "[ERROR] Only Intel/AMD x86_64, ARM AArch64 (aarch64), and Apple Silicon (arm64, macOS) are supported."
    exit 1
fi

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

# 4. Check and install build dependencies (system packages) per architecture
if [[ "$ARCH_TYPE" == "x86_64" || "$ARCH_TYPE" == "aarch64" ]]; then
    SYS_PKGS=(git gcc-12 g++-12 libnuma-dev)
    INSTALL_SYS_PKGS=()
    for pkg in "${SYS_PKGS[@]}"; do
        if ! dpkg -s "$pkg" &> /dev/null; then
            INSTALL_SYS_PKGS+=("$pkg")
        fi
    done
    if [[ ${#INSTALL_SYS_PKGS[@]} -gt 0 ]]; then
        if command -v apt-get &> /dev/null; then
            apt-get update
            apt-get install -y "${INSTALL_SYS_PKGS[@]}"
        elif command -v dnf &> /dev/null; then
            dnf install -y "${INSTALL_SYS_PKGS[@]}"
        elif command -v yum &> /dev/null; then
            yum install -y "${INSTALL_SYS_PKGS[@]}"
        else
            echo "[ERROR] No supported package manager found (apt-get, dnf, yum). Please install build dependencies manually: ${SYS_PKGS[*]}"
            exit 1
        fi
    else
        echo "[SKIP] gcc-12, g++-12, libnuma-dev already installed."
    fi
    # Ensure gcc-12 is set as the default gcc (Debian/Ubuntu only)
    if command -v update-alternatives &> /dev/null && ! gcc --version | grep -q 'gcc-12'; then
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
    fi
fi

# 5. Clone vllm source and install requirements/cpu.txt (common)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC_DIR="$SCRIPT_DIR/vllm_source"
if [ ! -d "$VLLM_SRC_DIR" ]; then
    git clone $VLLM_REPO "$VLLM_SRC_DIR"
fi
cd "$VLLM_SRC_DIR"
git fetch --tags
git checkout tags/$VLLM_TAG

$PYTHON_BIN -m pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Build wheel from source (actual build)
if [[ "$ARCH_TYPE" == "x86_64" || "$ARCH_TYPE" == "aarch64" ]]; then
    VLLM_TARGET_DEVICE=cpu $PYTHON_BIN setup.py install
elif [[ "$ARCH_TYPE" == "apple_silicon" ]]; then
    $PYTHON_BIN -m pip install -e .
fi

echo "vLLM CPU build and installation completed."