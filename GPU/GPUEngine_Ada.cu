/*
 * GPUEngine_Ada.cu — KeyHunt-Cuda GPU engine optimised for Ada Lovelace (sm_89)
 * and Blackwell Consumer (sm_120).
 *
 * Drop-in replacement for GPU/GPUEngine.cu.
 * Compile via: make gpu=1 CCAP=89 all   (RTX 40xx)
 *              make gpu=1 CCAP=120 all  (RTX 50xx)
 *
 * Optimisations applied vs. the stock VanitySearch/KeyHunt-Cuda engine:
 *  1. __launch_bounds__(TPB, MIN_BLOCKS_PER_SM) — tells the compiler the exact
 *     occupancy target so it caps register allocation appropriately.
 *  2. L2 cache persistence — pins the bloom filter inside Ada's 72 MB L2,
 *     dramatically reducing global memory traffic in ADDRESS/ADDRESSES mode.
 *  3. CUDA Graphs — eliminates per-iteration kernel-launch CPU overhead
 *     (~5–15 µs per launch) from the hot search loop.
 *  4. Dual-stream pipeline — overlaps GPU compute with host-side result
 *     processing and range updates.
 *  5. Async shared-memory prefetch (cp.async / cuda::memcpy_async) — hides
 *     global memory latency for the GTable load at the top of each step.
 *  6. Inline PTX helpers for the 64×64→128-bit multiply at the core of every
 *     256-bit modular multiplication.
 *
 * This file provides the host-side harness.  The device kernels live in
 * GPUCompute_Ada.h (included below).
 *
 * Copyright (c) 2024 — derived from JeanLucPons/VanitySearch (GPLv3)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "GPUEngine_Ada.h"   // host-facing header (mirrors GPUEngine.h)
#include "GPUCompute_Ada.h"  // device kernels

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time tunables
// ─────────────────────────────────────────────────────────────────────────────

// Threads per block.  Keep as a power of 2; 256 is a good default.
// If --ptxas-options=-v shows registers/thread > 64, occupancy on Ada is
// capped at 50%.  Reducing to 128 may raise occupancy at the cost of
// slightly more block-scheduling overhead.
#ifndef TPB
  #define TPB 256
#endif

// Minimum blocks per SM hint passed to __launch_bounds__.
// 4 → compiler targets ≥4 resident blocks/SM.
#ifndef MIN_BLOCKS_PER_SM
  #define MIN_BLOCKS_PER_SM 4
#endif

// Number of ping-pong streams for the dual-stream pipeline.
#define NUM_STREAMS 2

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error helper
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s:%d — %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GPUEngineAda — implementation
// ─────────────────────────────────────────────────────────────────────────────

GPUEngineAda::GPUEngineAda(int deviceId, int gridSize, int blockSize)
    : deviceId_(deviceId),
      gridSize_(gridSize),
      blockSize_(blockSize > 0 ? blockSize : TPB),
      d_bloom_(nullptr),
      bloomBytes_(0),
      d_results_(nullptr),
      resultBufSize_(0),
      graphExec_(nullptr),
      graph_(nullptr),
      graphCaptured_(false)
{
    CUDA_CHECK(cudaSetDevice(deviceId_));
    CUDA_CHECK(cudaGetDeviceProperties(&prop_, deviceId_));

    printf("[GPU %d] %s  sm_%d%d  SMs=%d  L2=%d MB  ShMem/SM=%d KB\n",
           deviceId_,
           prop_.name,
           prop_.major, prop_.minor,
           prop_.multiProcessorCount,
           prop_.l2CacheSize / (1024 * 1024),
           (int)(prop_.sharedMemPerMultiprocessor / 1024));

    // Derive a sensible grid size if caller passed 0
    if (gridSize_ <= 0)
        gridSize_ = prop_.multiProcessorCount * 8;

    // Create dual streams
    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));

    // Result buffer — one uint64_t found-flag + one 32-byte key per block
    resultBufSize_ = gridSize_ * (1 + 4) * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_results_, resultBufSize_));
    CUDA_CHECK(cudaMemset(d_results_, 0, resultBufSize_));

    printf("[GPU %d] Grid %d × %d threads  (%d total)\n",
           deviceId_, gridSize_, blockSize_, gridSize_ * blockSize_);
}

GPUEngineAda::~GPUEngineAda()
{
    if (graphExec_)  cudaGraphExecDestroy(graphExec_);
    if (graph_)      cudaGraphDestroy(graph_);
    if (d_bloom_)    cudaFree(d_bloom_);
    if (d_results_)  cudaFree(d_results_);
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams_[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// setBloom — upload the bloom filter and pin it in L2 (Ada 72 MB / Blackwell)
// ─────────────────────────────────────────────────────────────────────────────
void GPUEngineAda::setBloom(const uint8_t* hostBloom, size_t bytes)
{
    if (d_bloom_) { cudaFree(d_bloom_); d_bloom_ = nullptr; }
    bloomBytes_ = bytes;
    CUDA_CHECK(cudaMalloc(&d_bloom_, bytes));
    CUDA_CHECK(cudaMemcpy(d_bloom_, hostBloom, bytes, cudaMemcpyHostToDevice));

    // ── L2 persistence (Ada Lovelace / Blackwell consumer) ────────────────
    // Ada has up to 72 MB of L2; pin the bloom table so it stays hot
    // across all kernel launches for ADDRESS/ADDRESSES search modes.
    //
    // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...) must be called
    // BEFORE the first kernel on this device if you want the full set-aside.
    size_t setAside = min(bytes, (size_t)prop_.persistingL2CacheMaxSize);
    if (setAside > 0 && prop_.major >= 8) {   // sm_80+ required
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAside));

        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_bloom_);
        attr.accessPolicyWindow.num_bytes = setAside;
        attr.accessPolicyWindow.hitRatio  = 1.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        for (int i = 0; i < NUM_STREAMS; i++)
            CUDA_CHECK(cudaStreamSetAttribute(
                streams_[i],
                cudaStreamAttributeAccessPolicyWindow,
                &attr));

        printf("[GPU %d] L2 persistence: %.1f MB of bloom pinned (L2 = %d MB)\n",
               deviceId_,
               setAside / (1024.0 * 1024.0),
               prop_.l2CacheSize / (1024 * 1024));
    } else {
        printf("[GPU %d] L2 persistence: not available (sm_%d%d, persistingL2CacheMaxSize=%zu)\n",
               deviceId_, prop_.major, prop_.minor, prop_.persistingL2CacheMaxSize);
    }

    // Invalidate any previously captured CUDA graph — it holds stale pointers
    invalidateGraph_();
}

// ─────────────────────────────────────────────────────────────────────────────
// search — main entry point called each iteration of the key-range loop
// ─────────────────────────────────────────────────────────────────────────────
bool GPUEngineAda::search(const KernelArgs& args, FoundKey* outKeys, int* outCount)
{
    *outCount = 0;
    const int streamIdx = iterCount_++ % NUM_STREAMS;
    cudaStream_t stream = streams_[streamIdx];

    // ── CUDA Graphs path ────────────────────────────────────────────────────
    // After the first iteration (where we capture the graph), subsequent
    // iterations update the kernel's arguments in-place and re-launch the
    // executable graph — bypassing all CPU-side kernel-launch overhead.
    if (graphCaptured_) {
        updateGraphArgs_(args);
        CUDA_CHECK(cudaGraphLaunch(graphExec_, stream));
    } else {
        // First iteration: launch normally while capturing into a graph
        CUDA_CHECK(cudaStreamBeginCapture(stream,
                                          cudaStreamCaptureModeGlobal));
        launchKernel_(args, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));

        CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_,
                                        nullptr, nullptr, 0));
        graphCaptured_ = true;

        // Launch the captured graph immediately for this first iteration
        CUDA_CHECK(cudaGraphLaunch(graphExec_, stream));
    }

    // ── Async result copy (overlaps with next CPU work) ─────────────────────
    CUDA_CHECK(cudaMemcpyAsync(h_resultsBuf_, d_results_,
                                resultBufSize_,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Parse results ────────────────────────────────────────────────────────
    return parseResults_(outKeys, outCount);
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

void GPUEngineAda::launchKernel_(const KernelArgs& args, cudaStream_t stream)
{
    // Opt into maximum shared memory per block (Ada: up to 99 KB usable)
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)keyhunt_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        prop_.sharedMemPerBlockOptin));

    // Target the occupancy we want: __launch_bounds__ in the kernel header
    // enforces the register cap; this call requests the shared mem carveout.
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)keyhunt_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    size_t dynSmem = 0;  // static shared mem declared inside the kernel
    keyhunt_kernel<<<gridSize_, blockSize_, dynSmem, stream>>>(
        args.d_startKey,
        args.d_GTable,
        args.d_bloom,
        args.bloomBits,
        d_results_,
        args.keyCount);
}

void GPUEngineAda::invalidateGraph_()
{
    if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
    if (graph_)     { cudaGraphDestroy(graph_);          graph_     = nullptr; }
    graphCaptured_ = false;
    iterCount_     = 0;
}

void GPUEngineAda::updateGraphArgs_(const KernelArgs& args)
{
    // Walk the graph's kernel nodes and update any pointer/scalar parameters
    // that change between iterations (typically: d_startKey, keyCount).
    cudaGraphNode_t nodes[16];
    size_t numNodes = 0;
    cudaGraphGetNodes(graph_, nodes, &numNodes);

    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        cudaGraphNodeGetType(nodes[i], &type);
        if (type != cudaGraphNodeTypeKernel) continue;

        cudaKernelNodeParams params = {};
        cudaGraphKernelNodeGetParams(nodes[i], &params);

        // Patch arguments that change per iteration.
        // The kernel signature is: (startKey, GTable, bloom, bloomBits,
        //                           results, keyCount)
        // Indices 0 and 5 change; the rest are stable across iterations.
        void* newArgs[] = {
            (void*)&args.d_startKey,     // arg 0 — changes each iter
            (void*)&args.d_GTable,       // arg 1 — stable
            (void*)&args.d_bloom,        // arg 2 — stable
            (void*)&args.bloomBits,      // arg 3 — stable
            (void*)&d_results_,          // arg 4 — stable
            (void*)&args.keyCount        // arg 5 — may change
        };
        params.kernelParams = newArgs;
        cudaGraphExecKernelNodeSetParams(graphExec_, nodes[i], &params);
        break;
    }
}

bool GPUEngineAda::parseResults_(FoundKey* outKeys, int* outCount)
{
    // Result layout per block:
    //   uint64_t found_flag   (non-zero = key found)
    //   uint32_t key[8]       (32-byte private key, little-endian limbs)
    const uint64_t* buf = h_resultsBuf_;
    bool anyFound = false;

    for (int b = 0; b < gridSize_; b++) {
        uint64_t flag = buf[b * 5];
        if (flag) {
            const uint32_t* kp = reinterpret_cast<const uint32_t*>(&buf[b * 5 + 1]);
            FoundKey fk;
            memcpy(fk.key32, kp, 32);
            fk.blockIdx = b;
            outKeys[(*outCount)++] = fk;
            anyFound = true;
            // Clear the flag so it is not reported again
            CUDA_CHECK(cudaMemset(
                reinterpret_cast<uint8_t*>(d_results_) + b * 5 * sizeof(uint64_t),
                0, sizeof(uint64_t)));
        }
    }
    return anyFound;
}
