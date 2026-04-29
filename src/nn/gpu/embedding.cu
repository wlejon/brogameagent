#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

constexpr int EMB_BLOCK = 256;

// One thread per (b, j). Gather row d_idx[b] of `table` into out[b, :].
__global__ void embedding_lookup_forward_kernel(const float* __restrict__ table,
                                                const int* __restrict__ idx,
                                                float* __restrict__ out,
                                                int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        out[t] = table[row * D + j];
    }
}

// Scatter-accumulate via atomicAdd. Multiple lookups of the same row sum.
__global__ void embedding_lookup_backward_kernel(const float* __restrict__ dOut,
                                                 const int* __restrict__ idx,
                                                 float* __restrict__ dTable,
                                                 int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        atomicAdd(&dTable[row * D + j], dOut[t]);
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return b;
}

} // namespace

void embedding_lookup_forward_gpu(const GpuTensor& table,
                                  const int32_t* d_idx, int B,
                                  GpuTensor& out) {
    const int D = table.cols;
    if (out.rows != B || out.cols != D) out.resize(B, D);
    const int total = B * D;
    if (total == 0) return;
    embedding_lookup_forward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
        table.data, reinterpret_cast<const int*>(d_idx), out.data, B, D);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void embedding_lookup_backward_gpu(const GpuTensor& dOut,
                                   const int32_t* d_idx, int B,
                                   GpuTensor& dTable) {
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;
    embedding_lookup_backward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
        dOut.data, reinterpret_cast<const int*>(d_idx), dTable.data, B, D);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
