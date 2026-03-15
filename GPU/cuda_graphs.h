/*
 * cuda_graphs.h — CUDA Graph helpers for KeyHunt-Cuda's main search loop.
 *
 * Problem: KeyHunt-Cuda's search loop calls cudaMemcpyAsync + kernel launch
 * hundreds of times per second.  Each launch costs 5–15 µs of CPU overhead.
 * At 1 ms per kernel, that's up to 1.5% of wall time wasted — and it gets
 * worse with faster GPUs where kernel duration shrinks.
 *
 * Solution: capture the launch sequence into a CUDA Graph once, then call
 * cudaGraphLaunch() each iteration.  The graph launch overhead is < 1 µs.
 *
 * How it works:
 *  1. StreamGraphCapture captures everything launched on a stream into a graph.
 *  2. Instantiate once → cudaGraphExec_t (compiled, reusable).
 *  3. Update only the changing parameters (start key, key count) via
 *     cudaGraphExecKernelNodeSetParams each iteration.
 *  4. cudaGraphLaunch replaces the individual kernel + memcpy calls.
 *
 * Usage:
 *   StreamGraphCapture cap(stream);
 *   // ... your kernel launches and memcpy calls here ...
 *   auto exec = cap.instantiate();
 *
 *   while (searching) {
 *       exec.update_start_key(new_key);
 *       exec.launch(stream);
 *       cudaStreamSynchronize(stream);
 *       process_results();
 *   }
 */

#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK_GR(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDAGraph] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            throw std::runtime_error(cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// SearchGraphExec — a compiled CUDA graph that runs one search iteration.
// ─────────────────────────────────────────────────────────────────────────────
class SearchGraphExec {
public:
    SearchGraphExec() = default;

    explicit SearchGraphExec(cudaGraph_t graph) {
        CUDA_CHECK_GR(cudaGraphInstantiate(&exec_, graph,
                                           nullptr, nullptr, 0));
        // Cache the kernel node so we can update it per-iteration
        findKernelNode_(graph);
    }

    ~SearchGraphExec() {
        if (exec_) cudaGraphExecDestroy(exec_);
    }

    // Non-copyable, moveable
    SearchGraphExec(const SearchGraphExec&) = delete;
    SearchGraphExec& operator=(const SearchGraphExec&) = delete;
    SearchGraphExec(SearchGraphExec&& o) noexcept
        : exec_(o.exec_), kernelNode_(o.kernelNode_), params_(o.params_)
    { o.exec_ = nullptr; }

    // Update the first kernel argument (start key pointer) before each launch.
    // Extend this method for other per-iteration arguments as needed.
    void update_kernel_args(void** newArgs, int numArgs) {
        if (!kernelNode_) return;
        params_.kernelParams = newArgs;
        CUDA_CHECK_GR(cudaGraphExecKernelNodeSetParams(exec_, kernelNode_, &params_));
    }

    // Launch the graph — replaces all individual kernel/memcpy calls.
    void launch(cudaStream_t stream) {
        CUDA_CHECK_GR(cudaGraphLaunch(exec_, stream));
    }

    bool valid() const { return exec_ != nullptr; }

private:
    void findKernelNode_(cudaGraph_t graph) {
        size_t numNodes = 0;
        cudaGraphGetNodes(graph, nullptr, &numNodes);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        cudaGraphGetNodes(graph, nodes.data(), &numNodes);

        for (auto& node : nodes) {
            cudaGraphNodeType t;
            cudaGraphNodeGetType(node, &t);
            if (t == cudaGraphNodeTypeKernel) {
                kernelNode_ = node;
                cudaGraphKernelNodeGetParams(node, &params_);
                return;
            }
        }
        kernelNode_ = nullptr;
    }

    cudaGraphExec_t        exec_       = nullptr;
    cudaGraphNode_t        kernelNode_ = nullptr;
    cudaKernelNodeParams   params_     = {};
};

// ─────────────────────────────────────────────────────────────────────────────
// StreamGraphCapture — RAII stream capture scope.
//
// Example:
//   cudaGraph_t graph;
//   {
//       StreamGraphCapture cap(stream, &graph);
//       myKernel<<<g,b,0,stream>>>(args);
//       cudaMemcpyAsync(dst, src, n, D2H, stream);
//   }  // capture ends here, graph is populated
//   SearchGraphExec exec(graph);
//   cudaGraphDestroy(graph);
// ─────────────────────────────────────────────────────────────────────────────
class StreamGraphCapture {
public:
    StreamGraphCapture(cudaStream_t stream, cudaGraph_t* outGraph)
        : stream_(stream), outGraph_(outGraph)
    {
        CUDA_CHECK_GR(cudaStreamBeginCapture(stream_,
                                              cudaStreamCaptureModeGlobal));
    }

    ~StreamGraphCapture() {
        if (outGraph_)
            cudaStreamEndCapture(stream_, outGraph_);
    }

    // Cancel capture without producing a graph
    void cancel() {
        cudaStreamEndCapture(stream_, outGraph_);
        outGraph_ = nullptr;
    }

private:
    cudaStream_t  stream_;
    cudaGraph_t*  outGraph_;
};
