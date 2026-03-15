/*
 * l2_persist.h — L2 cache persistence helpers for Ada Lovelace / Blackwell.
 *
 * Ada (sm_89) L2 sizes:
 *   RTX 4090 (AD102): 72 MB usable (96 MB physical, ~75% available to CUDA)
 *   RTX 4080 (AD103): 64 MB
 *   RTX 4070 Ti (AD104): 48 MB
 *
 * Blackwell consumer (sm_120) L2:
 *   RTX 5090: ~96 MB (exact figure TBC by NVIDIA)
 *
 * L2 persistence pins a device buffer inside the L2 cache so it survives
 * across kernel launches.  For KeyHunt-Cuda's bloom filter (typically 4–64 MB
 * depending on target-set size), this eliminates repeated global-memory
 * fetches on every bloom probe — the dominant memory-bandwidth consumer.
 *
 * Usage:
 *   L2PersistScope scope(stream, d_bloom, bloom_bytes);
 *   // all kernels on `stream` will find d_bloom in L2
 *   // destructor resets persistence when scope exits
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
// l2_persist_set — pin `buf` (size `bytes`) in the L2 of device `deviceId`.
//
// Call ONCE before your kernel launch loop.  The effect persists for all
// kernels on all streams until l2_persist_clear() or program exit.
// ─────────────────────────────────────────────────────────────────────────────
inline bool l2_persist_set(cudaStream_t   stream,
                            void*          buf,
                            size_t         bytes,
                            int            deviceId = 0)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    if (prop.major < 8) {
        // L2 persistence requires sm_80+ (Ampere and newer)
        return false;
    }

    // Cap to what the device exposes as the maximum set-aside
    size_t setAside = bytes;
    if (setAside > prop.persistingL2CacheMaxSize)
        setAside = prop.persistingL2CacheMaxSize;

    // Reserve the L2 set-aside region
    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAside);
    if (err != cudaSuccess) {
        fprintf(stderr, "[L2] cudaDeviceSetLimit failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    // Attach the access policy to the stream
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = buf;
    attr.accessPolicyWindow.num_bytes = setAside;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    err = cudaStreamSetAttribute(stream,
                                  cudaStreamAttributeAccessPolicyWindow,
                                  &attr);
    if (err != cudaSuccess) {
        fprintf(stderr, "[L2] cudaStreamSetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    printf("[L2] %.1f MB pinned in L2 (device L2 = %d MB, max set-aside = %zu MB)\n",
           setAside / (1024.0 * 1024.0),
           prop.l2CacheSize / (1024 * 1024),
           prop.persistingL2CacheMaxSize / (1024 * 1024));
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// l2_persist_clear — release the L2 set-aside on the given stream.
// Call this when you no longer need persistence (e.g., target set changes).
// ─────────────────────────────────────────────────────────────────────────────
inline void l2_persist_clear(cudaStream_t stream)
{
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream,
                           cudaStreamAttributeAccessPolicyWindow,
                           &attr);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// RAII wrapper
// ─────────────────────────────────────────────────────────────────────────────
class L2PersistScope {
public:
    L2PersistScope(cudaStream_t stream, void* buf, size_t bytes, int deviceId = 0)
        : stream_(stream), active_(false)
    {
        active_ = l2_persist_set(stream, buf, bytes, deviceId);
    }
    ~L2PersistScope() {
        if (active_) l2_persist_clear(stream_);
    }
    bool active() const { return active_; }
private:
    cudaStream_t stream_;
    bool         active_;
};
