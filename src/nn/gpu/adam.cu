#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>
#include <math.h>

namespace brogameagent::nn::gpu {

namespace {

// Matches CPU semantics in src/nn/circuits.cpp::adam_step_cpu:
//   m  = beta1 * m + (1 - beta1) * g
//   v  = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^step)
//   v_hat = v / (1 - beta2^step)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
__global__ void adam_step_kernel(float* __restrict__ param,
                                 const float* __restrict__ grad,
                                 float* __restrict__ m,
                                 float* __restrict__ v,
                                 float lr, float beta1, float beta2, float eps,
                                 float inv_bc1, float inv_bc2,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = grad[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    const float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    const float m_hat = mi * inv_bc1;
    const float v_hat = vi * inv_bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

} // namespace

void adam_step_gpu(GpuTensor& param, const GpuTensor& grad,
                   GpuTensor& m, GpuTensor& v,
                   float lr, float beta1, float beta2, float eps, int step) {
    const int n = param.size();
    if (n == 0) return;
    const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    constexpr int BLOCK = 256;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    adam_step_kernel<<<blocks, BLOCK>>>(param.data, grad.data, m.data, v.data,
                                        lr, beta1, beta2, eps,
                                        inv_bc1, inv_bc2, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
