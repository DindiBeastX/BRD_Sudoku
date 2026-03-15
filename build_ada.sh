#!/usr/bin/env bash
# build_ada.sh — Build KeyHunt-Cuda optimised for Ada Lovelace (RTX 40xx, sm_89)
#
# Usage:
#   ./build_ada.sh            # auto-detect CUDA path
#   CUDA=/usr/local/cuda-12.4 ./build_ada.sh

set -euo pipefail

# ── Detect CUDA installation ──────────────────────────────────────────────────
if [[ -z "${CUDA:-}" ]]; then
    for candidate in /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 \
                     /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8 \
                     /usr/local/cuda; do
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

# sm_89 requires CUDA 11.8+
CUDA_MAJOR=${CUDA_VER%%.*}
CUDA_MINOR=${CUDA_VER##*.}
if (( CUDA_MAJOR < 11 || (CUDA_MAJOR == 11 && CUDA_MINOR < 8) )); then
    echo "[ERROR] Ada Lovelace (sm_89) requires CUDA >= 11.8. Found: $CUDA_VER"
    exit 1
fi

echo "[INFO] Building for Ada Lovelace (sm_89) ..."
make -j"$(nproc)" gpu=1 CCAP=89 CUDA="$CUDA" all

echo ""
echo "[OK] Build complete: ./keyhunt"
echo ""
echo "Quick benchmark (1 second):"
echo "  ./keyhunt -m address -f puzzles/puzzle66.txt -n 0x1000000 --gpux 1x256 -t 0"
echo ""
echo "Recommended thread config for RTX 4090:"
echo "  --gpux 1024x256  (128 SMs × 8 blocks × 256 threads)"
