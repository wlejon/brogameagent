#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

// NOTE on signature: per Subagent 1's spec, mean_out/rstd_out are host-side
// floats. We honour that: the kernel writes the two scalars to a tiny device
// scratch buffer, and we cudaMemcpy them back synchronously at the end of
// layernorm_forward_gpu. Backward consumes rstd as a host float (same).
// gamma/beta/xhat are device tensors as declared.

namespace {

constexpr int LN_BLOCK = 256;

__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         float* __restrict__ xhat,
                                         float* __restrict__ scratch, // [mean, rstd]
                                         int n, float eps) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Mean.
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += x[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);

    // Variance.
    float local_v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = x[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var = sdata[0] / static_cast<float>(n);
    const float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }

    for (int i = tid; i < n; i += blockDim.x) {
        const float xh = (x[i] - mean) * rstd;
        xhat[i] = xh;
        y[i] = gamma[i] * xh + beta[i];
    }
}

// Backward.
//   dGamma_i += dY_i * xhat_i
//   dBeta_i  += dY_i
//   dxh_i = dY_i * gamma_i
//   sum_dxh = sum dxh
//   sum_dxh_xhat = sum dxh * xhat
//   dX_i = (rstd / N) * (N * dxh_i - sum_dxh - xhat_i * sum_dxh_xhat)
__global__ void layernorm_backward_kernel(const float* __restrict__ dY,
                                          const float* __restrict__ xhat,
                                          const float* __restrict__ gamma,
                                          float rstd,
                                          float* __restrict__ dX,
                                          float* __restrict__ dGamma,
                                          float* __restrict__ dBeta,
                                          int n) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Accumulate dGamma, dBeta in place.
    for (int i = tid; i < n; i += blockDim.x) {
        const float g = dY[i];
        dGamma[i] += g * xhat[i];
        dBeta[i]  += g;
    }

    // sum_dxh
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += dY[i] * gamma[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh = sdata[0];

    // sum_dxh_xhat
    float local2 = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local2 += dY[i] * gamma[i] * xhat[i];
    }
    sdata[tid] = local2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh_xhat = sdata[0];

    const float nf = static_cast<float>(n);
    const float scale = rstd / nf;
    for (int i = tid; i < n; i += blockDim.x) {
        const float dxh = dY[i] * gamma[i];
        dX[i] = scale * (nf * dxh - sum_dxh - xhat[i] * sum_dxh_xhat);
    }
}

} // namespace

void layernorm_forward_gpu(const GpuTensor& x,
                           const GpuTensor& gamma, const GpuTensor& beta,
                           GpuTensor& y, GpuTensor& xhat,
                           float& mean_out, float& rstd_out,
                           float eps) {
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (xhat.rows != x.rows || xhat.cols != x.cols) xhat.resize(x.rows, x.cols);
    if (n == 0) {
        mean_out = 0.0f;
        rstd_out = 0.0f;
        return;
    }

    // Scratch buffer for [mean, rstd] on device.
    float* d_scratch = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                              2 * sizeof(float)));

    layernorm_forward_kernel<<<1, LN_BLOCK>>>(x.data, gamma.data, beta.data,
                                              y.data, xhat.data, d_scratch,
                                              n, eps);
    BGA_CUDA_CHECK(cudaGetLastError());

    float h[2] = {0.0f, 0.0f};
    BGA_CUDA_CHECK(cudaMemcpy(h, d_scratch, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_scratch);
    mean_out = h[0];
    rstd_out = h[1];
}

// Inference-only batched forward. Processes R independent rows of length D
// (R = rows, D = cols). One block per row; threads cooperate on mean/var
// reductions in shared memory. Does NOT cache xhat or write mean/rstd —
// no host syncs. Use when caches/backward aren't needed (e.g., inference
// pipeline through a TransformerEncoder).
namespace {
__global__ void layernorm_forward_inference_batched_kernel(
        const float* __restrict__ x,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ y,
        int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const float* xrow = x + static_cast<size_t>(row) * D;
    float*       yrow = y + static_cast<size_t>(row) * D;

    // Mean.
    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += xrow[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);

    // Variance.
    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = xrow[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(D);
    const float rstd = rsqrtf(var + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (xrow[i] - mean) * rstd;
        yrow[i] = xh * gamma[i] + beta[i];
    }
}
} // namespace

void layernorm_forward_inference_batched_gpu(const GpuTensor& X_RD,
                                              const GpuTensor& gamma,
                                              const GpuTensor& beta,
                                              GpuTensor& Y_RD,
                                              float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D) Y_RD.resize(R, D);
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    layernorm_forward_inference_batched_kernel<<<R, block, shmem>>>(
        X_RD.data, gamma.data, beta.data, Y_RD.data, R, D, eps);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward_gpu(const GpuTensor& dY, const GpuTensor& xhat,
                            const GpuTensor& gamma, float rstd,
                            GpuTensor& dX,
                            GpuTensor& dGamma, GpuTensor& dBeta) {
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols) dX.resize(dY.rows, dY.cols);
    if (n == 0) return;

    layernorm_backward_kernel<<<1, LN_BLOCK>>>(dY.data, xhat.data, gamma.data,
                                               rstd, dX.data,
                                               dGamma.data, dBeta.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
