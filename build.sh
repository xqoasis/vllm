#!/bin/bash
set -euo pipefail

retry() {
  local max_attempts="${CUSTOM_RETRY_MAX_ATTEMPTS:-5}"
  local sleep_seconds="${CUSTOM_RETRY_SLEEP_SECONDS:-20}"
  local attempt=1

  while true; do
    echo "Running command, attempt ${attempt}/${max_attempts}: $*"
    if "$@"; then
      return 0
    fi

    if [ "${attempt}" -ge "${max_attempts}" ]; then
      echo "Command failed after ${max_attempts} attempts: $*"
      return 1
    fi

    echo "Command failed. Retrying in ${sleep_seconds}s..."
    sleep "${sleep_seconds}"
    attempt=$((attempt + 1))
  done
}

ls -l

# =====================
# custom env and defaults
# =====================
: "${CUSTOM_TARGET_ARCH:=${ARCH:-$(uname -m)}}"
: "${CUSTOM_CUDA_TAG:=cu129}"

: "${CUSTOM_TORCH_VERSION:=2.10.0}"
: "${CUSTOM_TORCHAUDIO_VERSION:=2.10.0}"
: "${CUSTOM_TORCHVISION_VERSION:=0.25.0}"
: "${CUSTOM_TORCH_ABI_TAG:=th210}"

: "${CUSTOM_PYPI_INDEX_URL:=https://bytedpypi.byted.org/simple}"
: "${CUSTOM_PYPI_EXTRA_INDEX_URL:=https://bytedpypi.byted.org/simple}"

: "${CUSTOM_TRITON_VERSION:=v3.5.0}"

# 默认别开太多 arch，避免 nvcc OOM。
# 如需 sm_120，可以外部覆盖：
# export CUSTOM_TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
: "${CUSTOM_TORCH_CUDA_ARCH_LIST:=9.0;10.0}"

: "${CUSTOM_TOSUTIL_BASE_URL:=https://m645b3e1bb36e-mrap.mrap.accesspoint.tos-global.volces.com}"
: "${CUSTOM_TOS_ENDPOINT:=tos-cn-beijing.volces.com}"
: "${CUSTOM_TOS_REGION:=cn-beijing}"
: "${CUSTOM_TOS_BUCKET:=tos://xllm-vllm-whl}"

# Retry / timeout defaults
: "${CUSTOM_RETRY_MAX_ATTEMPTS:=5}"
: "${CUSTOM_RETRY_SLEEP_SECONDS:=20}"
: "${CUSTOM_PIP_RETRIES:=10}"
: "${CUSTOM_PIP_TIMEOUT:=300}"
: "${CUSTOM_WGET_TIMEOUT:=300}"
: "${CUSTOM_WGET_TRIES:=5}"

# CUDA runfile URLs
# CUDA 12.9.1
: "${CUSTOM_CUDA_129_X86_64_RUNFILE_URL:=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run}"
: "${CUSTOM_CUDA_129_SBSA_RUNFILE_URL:=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux_sbsa.run}"

# CUDA 13.0.2
: "${CUSTOM_CUDA_130_X86_64_RUNFILE_URL:=https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run}"
: "${CUSTOM_CUDA_130_SBSA_RUNFILE_URL:=https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux_sbsa.run}"

: "${CUSTOM_VERSION:?CUSTOM_VERSION is required}"

: "${CUSTOM_TOS_AK:?CUSTOM_TOS_AK is required}"
: "${CUSTOM_TOS_SK:?CUSTOM_TOS_SK is required}"

# =====================
# resolve arch
# =====================
case "${CUSTOM_TARGET_ARCH}" in
  x86_64|amd64)
    PLATFORM_TAG="x86_64"
    TOSUTIL_ARCH="amd64"
    RUNFILE_ARCH="x86_64"
    : "${CUSTOM_MAX_JOBS:=8}"
    ;;
  aarch64|arm64)
    PLATFORM_TAG="aarch64"
    TOSUTIL_ARCH="arm64"
    RUNFILE_ARCH="sbsa"
    : "${CUSTOM_MAX_JOBS:=4}"
    ;;
  *)
    echo "Unsupported CUSTOM_TARGET_ARCH=${CUSTOM_TARGET_ARCH}"
    echo "Supported: x86_64, amd64, aarch64, arm64"
    exit 1
    ;;
esac

# =====================
# resolve cuda
# =====================
CUDA_INSTALL_METHOD="runfile"

case "${CUSTOM_CUDA_TAG}" in
  cu129)
    CUDA_VERSION="12.9"
    CUDA_FULL_VERSION="12.9.1"
    CUSTOM_CUDA_HOME="${CUSTOM_CUDA_HOME:-/usr/local/cuda-12.9}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"

    case "${RUNFILE_ARCH}" in
      x86_64)
        CUDA_RUNFILE_URL="${CUSTOM_CUDA_129_X86_64_RUNFILE_URL}"
        ;;
      sbsa)
        CUDA_RUNFILE_URL="${CUSTOM_CUDA_129_SBSA_RUNFILE_URL}"
        ;;
      *)
        echo "Unsupported RUNFILE_ARCH=${RUNFILE_ARCH}"
        exit 1
        ;;
    esac
    ;;

  cu130)
    CUDA_VERSION="13.0"
    CUDA_FULL_VERSION="13.0.2"
    CUSTOM_CUDA_HOME="${CUSTOM_CUDA_HOME:-/usr/local/cuda-13.0}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"

    case "${RUNFILE_ARCH}" in
      x86_64)
        CUDA_RUNFILE_URL="${CUSTOM_CUDA_130_X86_64_RUNFILE_URL}"
        ;;
      sbsa)
        CUDA_RUNFILE_URL="${CUSTOM_CUDA_130_SBSA_RUNFILE_URL}"
        ;;
      *)
        echo "Unsupported RUNFILE_ARCH=${RUNFILE_ARCH}"
        exit 1
        ;;
    esac
    ;;

  *)
    echo "Unsupported CUSTOM_CUDA_TAG=${CUSTOM_CUDA_TAG}"
    echo "Supported: cu129, cu130"
    exit 1
    ;;
esac

# =====================
# install base system deps
# =====================
retry apt-get update

retry apt-get install -y \
  wget \
  git \
  gnupg \
  ca-certificates \
  build-essential \
  ninja-build \
  cmake \
  python3-dev \
  python3-pip

# =====================
# install cuda toolkit by runfile if missing
# =====================
if [ ! -x "${CUSTOM_CUDA_HOME}/bin/nvcc" ]; then
  echo "nvcc not found at ${CUSTOM_CUDA_HOME}/bin/nvcc"
  echo "Installing CUDA Toolkit ${CUDA_FULL_VERSION} for ${CUSTOM_TARGET_ARCH}..."
  echo "CUDA install method: ${CUDA_INSTALL_METHOD}"
  echo "CUDA_HOME: ${CUSTOM_CUDA_HOME}"
  echo "CUDA runfile URL: ${CUDA_RUNFILE_URL}"

  rm -f cuda-runfile.run

  retry wget \
    --tries="${CUSTOM_WGET_TRIES}" \
    --timeout="${CUSTOM_WGET_TIMEOUT}" \
    --read-timeout="${CUSTOM_WGET_TIMEOUT}" \
    --continue \
    -O cuda-runfile.run \
    "${CUDA_RUNFILE_URL}"

  chmod +x cuda-runfile.run

  retry sh cuda-runfile.run \
    --silent \
    --toolkit \
    --override \
    --installpath="${CUSTOM_CUDA_HOME}"

  rm -f cuda-runfile.run
else
  echo "Found nvcc at ${CUSTOM_CUDA_HOME}/bin/nvcc"
fi

# =====================
# export tool-required envs
# =====================
export CUDA_HOME="${CUSTOM_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export MAX_JOBS="${CUSTOM_MAX_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CUSTOM_CMAKE_BUILD_PARALLEL_LEVEL:-$CUSTOM_MAX_JOBS}"
export TORCH_CUDA_ARCH_LIST="${CUSTOM_TORCH_CUDA_ARCH_LIST}"
export GIT_TERMINAL_PROMPT=0

echo "CUSTOM_TARGET_ARCH=${CUSTOM_TARGET_ARCH}"
echo "PLATFORM_TAG=${PLATFORM_TAG}"
echo "RUNFILE_ARCH=${RUNFILE_ARCH}"
echo "CUSTOM_CUDA_TAG=${CUSTOM_CUDA_TAG}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "CUDA_FULL_VERSION=${CUDA_FULL_VERSION}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_INSTALL_METHOD=${CUDA_INSTALL_METHOD}"
echo "CUDA_RUNFILE_URL=${CUDA_RUNFILE_URL}"
echo "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
echo "CUSTOM_TORCH_VERSION=${CUSTOM_TORCH_VERSION}"
echo "CUSTOM_TORCHAUDIO_VERSION=${CUSTOM_TORCHAUDIO_VERSION}"
echo "CUSTOM_TORCHVISION_VERSION=${CUSTOM_TORCHVISION_VERSION}"
echo "CUSTOM_TORCH_ABI_TAG=${CUSTOM_TORCH_ABI_TAG}"
echo "CUSTOM_MAX_JOBS=${CUSTOM_MAX_JOBS}"
echo "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "CUSTOM_RETRY_MAX_ATTEMPTS=${CUSTOM_RETRY_MAX_ATTEMPTS}"
echo "CUSTOM_RETRY_SLEEP_SECONDS=${CUSTOM_RETRY_SLEEP_SECONDS}"
echo "CUSTOM_PIP_RETRIES=${CUSTOM_PIP_RETRIES}"
echo "CUSTOM_PIP_TIMEOUT=${CUSTOM_PIP_TIMEOUT}"
echo "CUSTOM_WGET_TIMEOUT=${CUSTOM_WGET_TIMEOUT}"
echo "CUSTOM_WGET_TRIES=${CUSTOM_WGET_TRIES}"

# =====================
# cuda toolkit check
# =====================
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "Error: nvcc still not found at ${CUDA_HOME}/bin/nvcc after installation"
  exit 1
fi

if [ ! -f "${CUDA_HOME}/include/cuda_runtime.h" ]; then
  echo "Error: cuda_runtime.h not found at ${CUDA_HOME}/include/cuda_runtime.h"
  exit 1
fi

"${CUDA_HOME}/bin/nvcc" --version

# =====================
# python deps
# =====================
retry pip install \
  --no-cache-dir \
  --retries "${CUSTOM_PIP_RETRIES}" \
  --timeout "${CUSTOM_PIP_TIMEOUT}" \
  --progress-bar off \
  numpy loguru build packaging setuptools wheel \
  -i "${CUSTOM_PYPI_INDEX_URL}"

# Do not use --no-cache-dir here.
# Torch wheels are large; keeping pip cache helps retries and unstable networks.
retry pip install \
  --retries "${CUSTOM_PIP_RETRIES}" \
  --timeout "${CUSTOM_PIP_TIMEOUT}" \
  --progress-bar off \
  torch=="${CUSTOM_TORCH_VERSION}" \
  torchaudio=="${CUSTOM_TORCHAUDIO_VERSION}" \
  torchvision=="${CUSTOM_TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}" \
  --extra-index-url "${CUSTOM_PYPI_EXTRA_INDEX_URL}"

python3 - <<PY
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
expected = "${CUDA_VERSION}"
assert torch.version.cuda == expected, f"expected cuda {expected}, got {torch.version.cuda}"
PY

python3 use_existing_torch.py

export VLLM_VERSION_OVERRIDE="${CUSTOM_VERSION}-$(git rev-parse --short=6 HEAD)-${CUSTOM_TORCH_ABI_TAG}-${CUSTOM_CUDA_TAG}"

if [ -f requirements/build.txt ]; then
  BUILD_REQ=requirements/build.txt
elif [ -f requirements/build/cuda.txt ]; then
  BUILD_REQ=requirements/build/cuda.txt
else
  echo "Error: neither requirements/build.txt nor requirements/build/cuda.txt exists"
  exit 1
fi

echo "Using build requirements: ${BUILD_REQ}"

retry pip install \
  --no-cache-dir \
  --retries "${CUSTOM_PIP_RETRIES}" \
  --timeout "${CUSTOM_PIP_TIMEOUT}" \
  --progress-bar off \
  -r "${BUILD_REQ}" \
  --no-build-isolation

# =====================
# triton source
# =====================
mkdir -p .deps
rm -rf .deps/triton-src

retry git clone --depth 1 --branch "${CUSTOM_TRITON_VERSION}" \
  "https://github.com/triton-lang/triton.git" \
  .deps/triton-src

export TRITON_KERNELS_SRC_DIR="$PWD/.deps/triton-src/python/triton_kernels/triton_kernels"
test -f "${TRITON_KERNELS_SRC_DIR}/__init__.py"

echo "TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR}"
echo "VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE}"

# =====================
# build
# =====================
rm -rf output
mkdir -p output

python3 -m build --wheel --no-isolation

cp dist/*.whl output/

whl_name="$(basename "$(ls output/*.whl | head -n1)")"

echo "Wheel output: output/${whl_name}"

# =====================
# upload
# =====================
retry wget \
  --tries="${CUSTOM_WGET_TRIES}" \
  --timeout="${CUSTOM_WGET_TIMEOUT}" \
  --read-timeout="${CUSTOM_WGET_TIMEOUT}" \
  -O tosutil \
  "${CUSTOM_TOSUTIL_BASE_URL}/linux/${TOSUTIL_ARCH}/tosutil"

chmod a+x tosutil

./tosutil version

./tosutil config \
  -i "${CUSTOM_TOS_AK}" \
  -k "${CUSTOM_TOS_SK}" \
  -e "${CUSTOM_TOS_ENDPOINT}" \
  -re "${CUSTOM_TOS_REGION}"

retry ./tosutil cp "output/${whl_name}" \
  "${CUSTOM_TOS_BUCKET}/${CUSTOM_CUDA_TAG}/${PLATFORM_TAG}/${whl_name}" \
  -acl public-read
