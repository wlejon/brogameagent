// CPU↔GPU parity tests for masked_mean_pool_forward / _backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/tensor.h>

#include <cuda_runtime.h>

#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

#define LOCAL_CUDA_OK(expr) do {                                              \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
        std::printf("cuda err: %s\n", cudaGetErrorString(_e));                \
        throw 0;                                                              \
    }                                                                         \
} while (0)

// CPU reference: matches DeepSetsEncoder/SetTransformerEncoder masked mean
// pool behaviour.
void masked_mean_pool_cpu(const Tensor& X, const std::vector<float>& mask,
                          Tensor& y) {
    const int K = X.rows;
    const int D = X.cols;
    y.resize(D, 1);
    y.zero();
    int nv = 0;
    for (int k = 0; k < K; ++k) if (mask[k] != 0.0f) ++nv;
    if (nv == 0) return;
    for (int k = 0; k < K; ++k) {
        if (mask[k] == 0.0f) continue;
        for (int j = 0; j < D; ++j) y[j] += X(k, j);
    }
    const float inv = 1.0f / static_cast<float>(nv);
    for (int j = 0; j < D; ++j) y[j] *= inv;
}

void masked_mean_pool_backward_cpu(const Tensor& dY,
                                   const std::vector<float>& mask,
                                   Tensor& dX) {
    const int K = dX.rows;
    const int D = dX.cols;
    int nv = 0;
    for (int k = 0; k < K; ++k) if (mask[k] != 0.0f) ++nv;
    dX.zero();
    if (nv == 0) return;
    const float inv = 1.0f / static_cast<float>(nv);
    for (int k = 0; k < K; ++k) {
        if (mask[k] == 0.0f) continue;
        for (int j = 0; j < D; ++j) dX(k, j) = dY[j] * inv;
    }
}

void run_pool(int K, int D, uint64_t seed, const std::vector<float>& mask) {
    SplitMix64 rng(seed);
    Tensor X(K, D), dY(D, 1);
    fill_random(X, rng);
    fill_random(dY, rng);

    Tensor y_cpu, dX_cpu(K, D);
    masked_mean_pool_cpu(X, mask, y_cpu);
    masked_mean_pool_backward_cpu(dY, mask, dX_cpu);

    GpuTensor gX, gdY, gy, gdX;
    upload(X, gX);
    upload(dY, gdY);

    float* d_mask = nullptr;
    LOCAL_CUDA_OK(cudaMalloc(&d_mask, sizeof(float) * K));
    LOCAL_CUDA_OK(cudaMemcpy(d_mask, mask.data(), sizeof(float) * K,
                             cudaMemcpyHostToDevice));

    // Pre-fill dX with garbage to confirm overwrite semantics.
    Tensor dX_garbage(K, D);
    fill_random(dX_garbage, rng);
    upload(dX_garbage, gdX);

    brogameagent::nn::gpu::masked_mean_pool_forward_gpu(gX, d_mask, gy);
    brogameagent::nn::gpu::masked_mean_pool_backward_gpu(gdY, d_mask, K, gdX);

    Tensor y_gpu = download_to_host(gy);
    Tensor dX_gpu = download_to_host(gdX);
    cudaFree(d_mask);

    compare_tensors(y_cpu, y_gpu, "masked_mean_pool.y");
    compare_tensors(dX_cpu, dX_gpu, "masked_mean_pool.dX");
}

std::vector<float> mask_all(int K)  { return std::vector<float>(K, 1.0f); }
std::vector<float> mask_half(int K) {
    std::vector<float> m(K, 0.0f);
    for (int k = 0; k < K; ++k) m[k] = (k < K / 2) ? 1.0f : 0.0f;
    if (K / 2 == 0) m[0] = 1.0f;
    return m;
}
std::vector<float> mask_one(int K) {
    std::vector<float> m(K, 0.0f);
    m[K / 2] = 1.0f;
    return m;
}

} // namespace

BGA_PARITY_TEST(pool_all_K4_D8)    { auto m = mask_all(4);   run_pool(4,  8,  0x300ull, m); }
BGA_PARITY_TEST(pool_all_K8_D32)   { auto m = mask_all(8);   run_pool(8,  32, 0x301ull, m); }
BGA_PARITY_TEST(pool_half_K8_D32)  { auto m = mask_half(8);  run_pool(8,  32, 0x310ull, m); }
BGA_PARITY_TEST(pool_half_K16_D64) { auto m = mask_half(16); run_pool(16, 64, 0x311ull, m); }
BGA_PARITY_TEST(pool_one_K8_D32)   { auto m = mask_one(8);   run_pool(8,  32, 0x320ull, m); }
BGA_PARITY_TEST(pool_one_K16_D64)  { auto m = mask_one(16);  run_pool(16, 64, 0x321ull, m); }

int main() { return run_all("gpu masked_mean_pool parity"); }
