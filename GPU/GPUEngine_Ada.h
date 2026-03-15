/*
 * GPUEngine_Ada.h — host-facing header for the Ada/Blackwell GPU engine.
 *
 * Mirrors the interface of GPUEngine.h from KeyHunt-Cuda so this file
 * can be substituted without changing caller code.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct FoundKey {
    uint32_t key32[8];   // 256-bit private key (8 × uint32, little-endian)
    int      blockIdx;   // which GPU block found it
};

// Arguments forwarded to the CUDA kernel each iteration.
// All device pointers must be allocated before the first call to search().
struct KernelArgs {
    uint32_t* d_startKey;   // device pointer: 8-limb starting private key
    uint32_t* d_GTable;     // device pointer: precomputed generator multiples
    uint8_t*  d_bloom;      // device pointer: bloom filter bytes
    uint32_t  bloomBits;    // log2(bloom filter bit-width)
    uint64_t  keyCount;     // number of keys to check this iteration
};

// ─────────────────────────────────────────────────────────────────────────────
// Engine class
// ─────────────────────────────────────────────────────────────────────────────

class GPUEngineAda {
public:
    // deviceId   — CUDA device index (0 for first GPU)
    // gridSize   — number of blocks; 0 = auto (8 × SM count)
    // blockSize  — threads per block; 0 = use compiled TPB default (256)
    GPUEngineAda(int deviceId = 0, int gridSize = 0, int blockSize = 0);
    ~GPUEngineAda();

    // Upload the bloom filter to GPU memory and enable L2 persistence.
    // Call once after construction (or whenever the target set changes).
    void setBloom(const uint8_t* hostBloom, size_t bytes);

    // Run one search iteration.  Returns true if ≥1 key was found.
    // outKeys must point to a buffer of at least gridSize FoundKey entries.
    bool search(const KernelArgs& args, FoundKey* outKeys, int* outCount);

    int gridSize()  const { return gridSize_; }
    int blockSize() const { return blockSize_; }

private:
    void launchKernel_(const KernelArgs& args, cudaStream_t stream);
    void invalidateGraph_();
    void updateGraphArgs_(const KernelArgs& args);
    bool parseResults_(FoundKey* outKeys, int* outCount);

    int                 deviceId_;
    int                 gridSize_;
    int                 blockSize_;
    cudaDeviceProp      prop_;
    cudaStream_t        streams_[2];

    uint8_t*            d_bloom_;
    size_t              bloomBytes_;

    uint64_t*           d_results_;
    size_t              resultBufSize_;
    uint64_t            h_resultsBuf_[/* max blocks */ 4096 * 5] = {};

    cudaGraph_t         graph_;
    cudaGraphExec_t     graphExec_;
    bool                graphCaptured_;
    int                 iterCount_ = 0;
};
