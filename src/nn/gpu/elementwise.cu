#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

constexpr int EW_BLOCK = 256;

__global__ void relu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
    }
}

__global__ void tanh_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = tanhf(x[i]);
    }
}

__global__ void tanh_backward_kernel(const float* __restrict__ y,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * (1.0f - yv * yv);
    }
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ y,
                                        const float* __restrict__ dY,
                                        float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * yv * (1.0f - yv);
    }
}

__global__ void add_inplace_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

__global__ void add_scalar_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += s;
    }
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

} // anonymous namespace

void relu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void relu_backward_gpu(const GpuTensor& x, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_backward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, dY.data, dX.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    tanh_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    tanh_backward_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, dY.data, dX.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, dY.data, dX.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    const int n = y.size();
    if (n == 0) return;
    add_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, x.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void add_scalar_inplace_gpu(GpuTensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    add_scalar_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, s, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
