// CPU↔GPU parity tests for mha_forward / mha_backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/multi_head_attention.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::MultiHeadAttention;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_mha(int K, int D, int H, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor X(K, D), dO(K, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    Tensor Wq(D, D), Wk(D, D), Wv(D, D), Wo(D, D);
    fill_random(Wq, rng, 0.5f);
    fill_random(Wk, rng, 0.5f);
    fill_random(Wv, rng, 0.5f);
    fill_random(Wo, rng, 0.5f);

    // Pre-fill dW* to validate accumulation behavior.
    Tensor dWq_init(D, D), dWk_init(D, D), dWv_init(D, D), dWo_init(D, D);
    fill_random(dWq_init, rng, 0.1f);
    fill_random(dWk_init, rng, 0.1f);
    fill_random(dWv_init, rng, 0.1f);
    fill_random(dWo_init, rng, 0.1f);

    // CPU path.
    MultiHeadAttention att;
    uint64_t s = seed ^ 0xDEADBEEFull;
    att.init(K, D, H, s);
    att.Wq() = Wq; att.Wk() = Wk; att.Wv() = Wv; att.Wo() = Wo;
    att.dWq() = dWq_init; att.dWk() = dWk_init;
    att.dWv() = dWv_init; att.dWo() = dWo_init;

    Tensor O_cpu(K, D), dX_cpu(K, D);
    att.forward(X, mask ? mask->data() : nullptr, O_cpu);
    att.backward(dO, dX_cpu);
    Tensor dWq_cpu = att.dWq(); Tensor dWk_cpu = att.dWk();
    Tensor dWv_cpu = att.dWv(); Tensor dWo_cpu = att.dWo();

    // GPU path.
    GpuTensor gX, gWq, gWk, gWv, gWo;
    upload(X, gX);
    upload(Wq, gWq); upload(Wk, gWk); upload(Wv, gWv); upload(Wo, gWo);

    GpuTensor gQh, gKh, gVh, gAttnh, gYconcat, gO;
    const int dh = D / H;
    gQh.resize(H * K, dh); gKh.resize(H * K, dh); gVh.resize(H * K, dh);
    gAttnh.resize(H * K, K); gYconcat.resize(K, D); gO.resize(K, D);

    float* d_mask = nullptr;
    if (mask) {
        cudaMalloc(&d_mask, sizeof(float) * K);
        cudaMemcpy(d_mask, mask->data(), sizeof(float) * K, cudaMemcpyHostToDevice);
    }
    brogameagent::nn::gpu::mha_forward_gpu(
        gX, gWq, gWk, gWv, gWo, d_mask, H,
        gQh, gKh, gVh, gAttnh, gYconcat, gO);

    Tensor O_gpu = download_to_host(gO);

    GpuTensor gdO, gdX, gdWq, gdWk, gdWv, gdWo;
    upload(dO, gdO);
    gdX.resize(K, D);
    upload(dWq_init, gdWq); upload(dWk_init, gdWk);
    upload(dWv_init, gdWv); upload(dWo_init, gdWo);

    brogameagent::nn::gpu::mha_backward_gpu(
        gdO, gX, gQh, gKh, gVh, gAttnh, gYconcat,
        gWq, gWk, gWv, gWo, d_mask, H,
        gdX, gdWq, gdWk, gdWv, gdWo);

    Tensor dX_gpu  = download_to_host(gdX);
    Tensor dWq_gpu = download_to_host(gdWq);
    Tensor dWk_gpu = download_to_host(gdWk);
    Tensor dWv_gpu = download_to_host(gdWv);
    Tensor dWo_gpu = download_to_host(gdWo);

    if (d_mask) cudaFree(d_mask);

    // Slightly looser tolerance — multi-head attention chains many GEMMs and
    // softmaxes; FMA reordering between CPU and GPU pushes tighter tolerances
    // over (matches the documented adjustment in the single-head test).
    const float atol = 5e-5f, rtol = 5e-4f;
    compare_tensors(O_cpu,  O_gpu,  "mha.O", atol, rtol);
    compare_tensors(dX_cpu, dX_gpu, "mha.dX", atol, rtol);
    compare_tensors(dWq_cpu, dWq_gpu, "mha.dWq", atol, rtol);
    compare_tensors(dWk_cpu, dWk_gpu, "mha.dWk", atol, rtol);
    compare_tensors(dWv_cpu, dWv_gpu, "mha.dWv", atol, rtol);
    compare_tensors(dWo_cpu, dWo_gpu, "mha.dWo", atol, rtol);

    if (mask) {
        for (int i = 0; i < K; ++i) {
            if ((*mask)[i] < 0.5f) {
                for (int c = 0; c < D; ++c) BGA_CHECK(O_gpu(i, c) == 0.0f);
            }
        }
    }
}

std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;
    return m;
}

} // namespace

// Unmasked: cover h ∈ {1, 2, 4}, D ∈ {32, 64}, K ∈ {4, 16}.
BGA_PARITY_TEST(mha_h1_K4_D32)   { run_mha(4,  32, 1, 0x600ull, nullptr); }
BGA_PARITY_TEST(mha_h2_K4_D32)   { run_mha(4,  32, 2, 0x601ull, nullptr); }
BGA_PARITY_TEST(mha_h4_K4_D32)   { run_mha(4,  32, 4, 0x602ull, nullptr); }
BGA_PARITY_TEST(mha_h2_K16_D32)  { run_mha(16, 32, 2, 0x603ull, nullptr); }
BGA_PARITY_TEST(mha_h4_K16_D64)  { run_mha(16, 64, 4, 0x604ull, nullptr); }
BGA_PARITY_TEST(mha_h1_K16_D64)  { run_mha(16, 64, 1, 0x605ull, nullptr); }

// Masked.
BGA_PARITY_TEST(mha_h2_K4_D32_mask)  { auto m = partial_mask(4);  run_mha(4,  32, 2, 0x610ull, &m); }
BGA_PARITY_TEST(mha_h4_K16_D64_mask) { auto m = partial_mask(16); run_mha(16, 64, 4, 0x611ull, &m); }

int main() { return run_all("gpu mha parity"); }
