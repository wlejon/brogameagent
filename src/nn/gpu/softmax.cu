#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

constexpr int SM_BLOCK = 256;

// One block computes softmax over a vector of length N.
// Stable: subtract max over valid (mask==1) entries, exp, normalise.
__global__ void softmax_forward_kernel(const float* __restrict__ logits,
                                       float* __restrict__ probs,
                                       const float* __restrict__ mask,
                                       int n) {
    __shared__ float sdata[SM_BLOCK];

    const int tid = threadIdx.x;

    // Phase 1: find max over valid entries.
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] == 0.0f) continue;
        const float v = logits[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    // Phase 2: write exp(x - m) into probs (zero on masked), accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] == 0.0f) {
            probs[i] = 0.0f;
            continue;
        }
        const float e = expf(logits[i] - m);
        probs[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] = probs[i] * inv;
    }
}

// dL/dz_i = p_i * (dL/dp_i - sum_j dL/dp_j * p_j).
__global__ void softmax_backward_kernel(const float* __restrict__ probs,
                                        const float* __restrict__ dProbs,
                                        float* __restrict__ dLogits,
                                        int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += dProbs[i] * probs[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int i = tid; i < n; i += blockDim.x) {
        dLogits[i] = probs[i] * (dProbs[i] - dot);
    }
}

} // namespace

void softmax_forward_gpu(const GpuTensor& logits, GpuTensor& probs,
                         const float* d_mask) {
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols) {
        probs.resize(logits.rows, logits.cols);
    }
    if (n == 0) return;
    softmax_forward_kernel<<<1, SM_BLOCK>>>(logits.data, probs.data, d_mask, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void softmax_backward_gpu(const GpuTensor& probs, const GpuTensor& dProbs,
                          GpuTensor& dLogits) {
    const int n = probs.size();
    if (dLogits.rows != probs.rows || dLogits.cols != probs.cols) {
        dLogits.resize(probs.rows, probs.cols);
    }
    if (n == 0) return;
    softmax_backward_kernel<<<1, SM_BLOCK>>>(probs.data, dProbs.data,
                                             dLogits.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
