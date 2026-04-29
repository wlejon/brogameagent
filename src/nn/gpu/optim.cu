#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

// Matches CPU semantics in src/nn/attention.cpp::sgd_mat:
//   v[i] = momentum * v[i] + g[i];
//   w[i] -= lr * v[i];
__global__ void sgd_step_kernel(float* __restrict__ param,
                                const float* __restrict__ grad,
                                float* __restrict__ velocity,
                                float lr, float momentum, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = momentum * velocity[i] + grad[i];
    velocity[i] = v;
    param[i] -= lr * v;
}

} // namespace

void sgd_step_gpu(GpuTensor& param, GpuTensor& grad, GpuTensor& velocity,
                  float lr, float momentum) {
    const int n = param.size();
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    sgd_step_kernel<<<blocks, BLOCK>>>(param.data, grad.data, velocity.data,
                                       lr, momentum, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
