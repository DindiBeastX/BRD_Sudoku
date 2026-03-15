#!/usr/bin/env bash
# build_blackwell.sh — Build KeyHunt-Cuda optimised for Blackwell consumer
#                      (RTX 50xx, sm_120)
#
# Requirements:
#   CUDA >= 12.8
#   NVIDIA Driver >= 570
#
# Usage:
#   ./build_blackwell.sh
#   CUDA=/usr/local/cuda-12.8 ./build_blackwell.sh

set -euo pipefail

# ── Detect CUDA installation ──────────────────────────────────────────────────
if [[ -z "${CUDA:-}" ]]; then
    for candidate in /usr/local/cuda-12.8 /usr/local/cuda-12.9 \
                     /usr/local/cuda-13.0 /usr/local/cuda; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            CUDA="$candidate"
            break
        fi
    done
fi

if [[ -z "${CUDA:-}" ]]; then
    echo "[ERROR] CUDA not found. Set CUDA=/path/to/cuda and re-run."
    exit 1
fi

NVCC="$CUDA/bin/nvcc"
CUDA_VER=$("$NVCC" --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "[INFO] Using CUDA $CUDA_VER at $CUDA"

# sm_120 requires CUDA 12.8+
CUDA_MAJOR=${CUDA_VER%%.*}
CUDA_MINOR=${CUDA_VER##*.}
if (( CUDA_MAJOR < 12 || (CUDA_MAJOR == 12 && CUDA_MINOR < 8) )); then
    echo "[ERROR] Blackwell consumer (sm_120) requires CUDA >= 12.8. Found: $CUDA_VER"
    echo "        Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# ── Check driver version ──────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRIVER_MAJOR=${DRIVER_VER%%.*}
    echo "[INFO] NVIDIA driver: $DRIVER_VER"
    if (( DRIVER_MAJOR < 570 )); then
        echo "[WARN] sm_120 requires driver >= 570. Found: $DRIVER_VER"
        echo "       The binary may fail to run on this system."
    fi
fi

echo "[INFO] Building for Blackwell consumer (sm_120) ..."
make -j"$(nproc)" gpu=1 CCAP=120 CUDA="$CUDA" all

echo ""
echo "[OK] Build complete: ./keyhunt"
echo ""
echo "Recommended thread config for RTX 5090 (170 SMs):"
echo "  --gpux 1360x256  (170 SMs × 8 blocks × 256 threads)"
echo ""
echo "Note: sm_120 (RTX 50xx) does NOT have TMEM or wgmma — only sm_100"
echo "datacenter Blackwell (B200/GB200) has those features."
