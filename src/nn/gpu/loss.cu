#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

constexpr int LOSS_BLOCK = 256;

// Block-reduce sum over a flat length-N buffer of (pred-target)^2.
// Single-block launch — N ≤ a few thousand for our use cases. Threads stride
// over the buffer.
__global__ void mse_forward_kernel(const float* __restrict__ pred,
                                   const float* __restrict__ target,
                                   float* __restrict__ out_sum, int n) {
    __shared__ float sdata[LOSS_BLOCK];
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = pred[i] - target[i];
        local += d * d;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out_sum = sdata[0];
}

__global__ void mse_backward_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    float* __restrict__ dPred, int n,
                                    float scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dPred[i] = scale * (pred[i] - target[i]);
    }
}

// Fused softmax + cross-entropy. Single block, length-N vector.
//   probs[i]   = softmax(logits)[i]   (0 on masked)
//   dLogits[i] = probs[i] - target[i] (0 on masked)
//   loss       = -sum_{valid i} target[i] * log(max(probs[i], 1e-12))
__global__ void softmax_xent_fused_kernel(const float* __restrict__ logits,
                                          const float* __restrict__ target,
                                          const float* __restrict__ mask,
                                          float* __restrict__ probs,
                                          float* __restrict__ dLogits,
                                          float* __restrict__ out_loss,
                                          int n) {
    __shared__ float sdata[LOSS_BLOCK];
    const int tid = threadIdx.x;

    // Phase 1: max over valid.
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

    // Phase 2: exp(x - m) into probs (zero on masked), accumulate sum.
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

    // Phase 3: normalise probs, compute loss + dLogits.
    float local_loss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] == 0.0f) {
            dLogits[i] = 0.0f;
            continue;
        }
        const float p = probs[i] * inv;
        probs[i] = p;
        const float t = target[i];
        if (t > 0.0f) {
            const float pc = p > 1e-12f ? p : 1e-12f;
            local_loss -= t * logf(pc);
        }
        dLogits[i] = p - t;
    }
    sdata[tid] = local_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out_loss = sdata[0];
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return b;
}

} // namespace

float mse_vec_forward_gpu(const GpuTensor& pred, const GpuTensor& target) {
    const int n = pred.size();
    if (n == 0) return 0.0f;
    float* d_sum = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    mse_forward_kernel<<<1, LOSS_BLOCK>>>(pred.data, target.data, d_sum, n);
    BGA_CUDA_CHECK(cudaGetLastError());
    float h_sum = 0.0f;
    BGA_CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_sum);
    return h_sum / static_cast<float>(n);
}

void mse_vec_backward_gpu(const GpuTensor& pred, const GpuTensor& target,
                          GpuTensor& dPred) {
    const int n = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols) {
        dPred.resize(pred.rows, pred.cols);
    }
    if (n == 0) return;
    const float scale = 2.0f / static_cast<float>(n);
    mse_backward_kernel<<<grid_for(n, LOSS_BLOCK), LOSS_BLOCK>>>(
        pred.data, target.data, dPred.data, n, scale);
    BGA_CUDA_CHECK(cudaGetLastError());
}

float softmax_xent_fused_gpu(const GpuTensor& logits, const GpuTensor& target,
                             const float* d_mask,
                             GpuTensor& probs, GpuTensor& dLogits) {
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols) {
        probs.resize(logits.rows, logits.cols);
    }
    if (dLogits.rows != logits.rows || dLogits.cols != logits.cols) {
        dLogits.resize(logits.rows, logits.cols);
    }
    if (n == 0) return 0.0f;

    float* d_loss = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    softmax_xent_fused_kernel<<<1, LOSS_BLOCK>>>(
        logits.data, target.data, d_mask,
        probs.data, dLogits.data, d_loss, n);
    BGA_CUDA_CHECK(cudaGetLastError());

    float h_loss = 0.0f;
    BGA_CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
    return h_loss;
}

} // namespace brogameagent::nn::gpu
