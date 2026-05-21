#!/bin/bash
# https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html

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
VLLM_REPO=${VLLM_REPO:-https://github.com/vllm-project/vllm.git}
# VLLM_TAG can be passed from Makefile, defaults to v0.15.0
VLLM_TAG=${VLLM_TAG:-v0.15.0}
# VLLM_BRANCH can be passed from Makefile to checkout a branch instead of a tag
VLLM_BRANCH=${VLLM_BRANCH:-}

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

# Use sudo for root-only operations when running as non-root.
SUDO=""
if [ "$(id -u)" != "0" ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    fi
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
    # Detect package manager and set appropriate package names
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt"
        SYS_PKGS=(git gcc-12 g++-12 libnuma-dev)
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        SYS_PKGS=(git gcc gcc-c++ numactl-devel)
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        SYS_PKGS=(git gcc gcc-c++ numactl-devel)
    else
        echo "[ERROR] No supported package manager found (apt-get, dnf, yum). Please install build dependencies manually: git, gcc, g++, libnuma-dev (or numactl-devel)"
        exit 1
    fi

    # Check which packages need to be installed
    INSTALL_SYS_PKGS=()
    for pkg in "${SYS_PKGS[@]}"; do
        if [[ "$PKG_MANAGER" == "apt" ]]; then
            if ! dpkg -s "$pkg" &> /dev/null; then
                INSTALL_SYS_PKGS+=("$pkg")
            fi
        else
            # For dnf/yum, use rpm to check installed packages
            if ! rpm -q "$pkg" &> /dev/null; then
                INSTALL_SYS_PKGS+=("$pkg")
            fi
        fi
    done

    if [[ ${#INSTALL_SYS_PKGS[@]} -gt 0 ]]; then
        echo "[INFO] Installing system packages: ${INSTALL_SYS_PKGS[*]}"
        if [[ "$PKG_MANAGER" == "apt" ]]; then
            $SUDO apt-get update
            $SUDO apt-get install -y "${INSTALL_SYS_PKGS[@]}"
        elif [[ "$PKG_MANAGER" == "dnf" ]]; then
            $SUDO dnf install -y "${INSTALL_SYS_PKGS[@]}"
        elif [[ "$PKG_MANAGER" == "yum" ]]; then
            $SUDO yum install -y "${INSTALL_SYS_PKGS[@]}"
        fi
    else
        echo "[SKIP] Build dependencies already installed."
    fi

    # Ensure gcc-12 is set as the default gcc (Debian/Ubuntu only).
    # Per https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#build-wheel-from-source
    if [[ "$PKG_MANAGER" == "apt" ]] && command -v update-alternatives &> /dev/null; then
        GCC_MAJOR=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
        if [[ "$GCC_MAJOR" != "12" ]]; then
            $SUDO update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
        else
            echo "[SKIP] gcc is already version $GCC_MAJOR (default)."
        fi
    fi
fi

# 5. Clone vllm source and install requirements/cpu.txt (common)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC_DIR="$SCRIPT_DIR/vllm_source"
if [ ! -d "$VLLM_SRC_DIR" ]; then
    if [ -n "$VLLM_BRANCH" ]; then
        git clone --depth=1 --single-branch --branch "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_SRC_DIR"
    else
        git clone --depth=1 --single-branch --branch "$VLLM_TAG" "$VLLM_REPO" "$VLLM_SRC_DIR"
    fi
fi
cd "$VLLM_SRC_DIR"

if [ -f requirements/build/cpu.txt ]; then
    $PYTHON_BIN -m pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi
$PYTHON_BIN -m pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Build wheel from source (actual build)
if [[ "$ARCH_TYPE" == "x86_64" ]]; then
    # Check if CPU supports AVX-512
    if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
        echo "[INFO] AVX-512 supported, building with full optimizations"
        VLLM_TARGET_DEVICE=cpu $PYTHON_BIN -m pip install --no-build-isolation .
    else
        echo "[INFO] AVX-512 NOT supported, building with AVX2 only (VLLM_CPU_DISABLE_AVX512=true)"
        VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512=true $PYTHON_BIN -m pip install --no-build-isolation .
    fi
elif [[ "$ARCH_TYPE" == "aarch64" ]]; then
    VLLM_TARGET_DEVICE=cpu $PYTHON_BIN -m pip install --no-build-isolation .
elif [[ "$ARCH_TYPE" == "apple_silicon" ]]; then
    $PYTHON_BIN -m pip install -e .
fi

# 7. Install vLLM audio extra (required for audio_url / input_audio multimodal inputs)
# vLLM's base install omits soundfile/resampy/av; they live under the '[audio]' extra.
# Re-run pip install against the already-built source tree to pull the extra only.
if [[ "$ARCH_TYPE" == "x86_64" || "$ARCH_TYPE" == "aarch64" ]]; then
    VLLM_TARGET_DEVICE=cpu $PYTHON_BIN -m pip install --no-build-isolation '.[audio]'
else
    $PYTHON_BIN -m pip install -e '.[audio]'
fi

echo "✅ vLLM CPU build and installation completed for $OS $ARCH."