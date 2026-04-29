#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

constexpr int RED_BLOCK = 256;

// One thread per output column. Each thread sums its column over valid rows
// and divides by num_valid. If num_valid == 0, writes 0.
__global__ void masked_mean_pool_forward_kernel(const float* __restrict__ X,
                                                const float* __restrict__ mask,
                                                float* __restrict__ y,
                                                int K, int D) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    // Count num_valid (cheap; D threads each redundantly count K — fine for
    // the K we care about; if K grows large we'd hoist this).
    int num_valid = 0;
    if (mask) {
        for (int k = 0; k < K; ++k) num_valid += (mask[k] != 0.0f);
    } else {
        num_valid = K;
    }

    if (num_valid == 0) { y[j] = 0.0f; return; }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float m = mask ? mask[k] : 1.0f;
        if (m != 0.0f) acc += X[k * D + j];
    }
    y[j] = acc / static_cast<float>(num_valid);
}

// Distribute dY/num_valid to valid rows; zero invalid rows.
__global__ void masked_mean_pool_backward_kernel(const float* __restrict__ dY,
                                                 const float* __restrict__ mask,
                                                 float* __restrict__ dX,
                                                 int K, int D, int num_valid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * D;
    if (idx >= total) return;
    const int k = idx / D;
    const int j = idx - k * D;
    const float m = mask ? mask[k] : 1.0f;
    if (num_valid == 0 || m == 0.0f) {
        dX[idx] = 0.0f;
    } else {
        dX[idx] = dY[j] / static_cast<float>(num_valid);
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

} // namespace

void masked_mean_pool_forward_gpu(const GpuTensor& X, const float* d_mask,
                                  GpuTensor& y) {
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1) y.resize(D, 1);
    if (D == 0) return;
    masked_mean_pool_forward_kernel<<<grid_for(D, RED_BLOCK), RED_BLOCK>>>(
        X.data, d_mask, y.data, K, D);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void masked_mean_pool_backward_gpu(const GpuTensor& dY, const float* d_mask,
                                   int K, GpuTensor& dX) {
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    const int total = K * D;
    if (total == 0) return;

    // Need num_valid on host for the divisor. We could reduce on device, but
    // K is small (slot counts of a few dozen) — just download the mask.
    int num_valid = 0;
    if (d_mask) {
        std::vector<float> hmask(K);
        BGA_CUDA_CHECK(cudaMemcpy(hmask.data(), d_mask, sizeof(float) * K,
                                  cudaMemcpyDeviceToHost));
        for (int k = 0; k < K; ++k) num_valid += (hmask[k] != 0.0f);
    } else {
        num_valid = K;
    }

    masked_mean_pool_backward_kernel<<<grid_for(total, RED_BLOCK), RED_BLOCK>>>(
        dY.data, d_mask, dX.data, K, D, num_valid);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
