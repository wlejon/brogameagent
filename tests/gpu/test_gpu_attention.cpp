// CPU↔GPU parity tests for attention_forward / attention_backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/attention.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::ScaledDotProductAttention;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_attention(int N, int D, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor X(N, D), dO(N, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    // Build the four projection weights with the same RNG used by xavier
    // would not match host/device — we just use random weights and feed them
    // identically to both CPU and GPU paths.
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

    // CPU path: instantiate the layer, swap in our weights, run.
    ScaledDotProductAttention att;
    uint64_t s = seed ^ 0xDEADBEEFull;
    att.init(N, D, s);
    att.Wq() = Wq; att.Wk() = Wk; att.Wv() = Wv; att.Wo() = Wo;
    att.dWq() = dWq_init; att.dWk() = dWk_init;
    att.dWv() = dWv_init; att.dWo() = dWo_init;

    Tensor O_cpu(N, D), dX_cpu(N, D);
    att.forward(X, mask ? mask->data() : nullptr, O_cpu);
    att.backward(dO, dX_cpu);
    Tensor dWq_cpu = att.dWq(); Tensor dWk_cpu = att.dWk();
    Tensor dWv_cpu = att.dWv(); Tensor dWo_cpu = att.dWo();

    // GPU path.
    GpuTensor gX, gWq, gWk, gWv, gWo;
    upload(X, gX);
    upload(Wq, gWq); upload(Wk, gWk); upload(Wv, gWv); upload(Wo, gWo);

    GpuTensor gQ, gK, gV, gAttn, gYpre, gO;
    gQ.resize(N, D); gK.resize(N, D); gV.resize(N, D);
    gAttn.resize(N, N); gYpre.resize(N, D); gO.resize(N, D);

    float* d_mask = nullptr;
    if (mask) {
        cudaMalloc(&d_mask, sizeof(float) * N);
        cudaMemcpy(d_mask, mask->data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    }
    brogameagent::nn::gpu::attention_forward_gpu(
        gX, gWq, gWk, gWv, gWo, d_mask, gQ, gK, gV, gAttn, gYpre, gO);

    Tensor O_gpu = download_to_host(gO);

    GpuTensor gdO, gdX, gdWq, gdWk, gdWv, gdWo;
    upload(dO, gdO);
    gdX.resize(N, D);
    upload(dWq_init, gdWq); upload(dWk_init, gdWk);
    upload(dWv_init, gdWv); upload(dWo_init, gdWo);

    brogameagent::nn::gpu::attention_backward_gpu(
        gdO, gX, gQ, gK, gV, gAttn, gYpre,
        gWq, gWk, gWv, gWo, d_mask,
        gdX, gdWq, gdWk, gdWv, gdWo);

    Tensor dX_gpu  = download_to_host(gdX);
    Tensor dWq_gpu = download_to_host(gdWq);
    Tensor dWk_gpu = download_to_host(gdWk);
    Tensor dWv_gpu = download_to_host(gdWv);
    Tensor dWo_gpu = download_to_host(gdWo);

    if (d_mask) cudaFree(d_mask);

    // Slightly looser tolerance — attention chains many GEMMs and softmaxes.
    const float atol = 5e-5f, rtol = 5e-4f;
    compare_tensors(O_cpu,  O_gpu,  "attn.O", atol, rtol);
    compare_tensors(dX_cpu, dX_gpu, "attn.dX", atol, rtol);
    compare_tensors(dWq_cpu, dWq_gpu, "attn.dWq", atol, rtol);
    compare_tensors(dWk_cpu, dWk_gpu, "attn.dWk", atol, rtol);
    compare_tensors(dWv_cpu, dWv_gpu, "attn.dWv", atol, rtol);
    compare_tensors(dWo_cpu, dWo_gpu, "attn.dWo", atol, rtol);

    // Invalid rows in O should be zero on the GPU.
    if (mask) {
        for (int i = 0; i < N; ++i) {
            if ((*mask)[i] < 0.5f) {
                for (int c = 0; c < D; ++c) {
                    BGA_CHECK(O_gpu(i, c) == 0.0f);
                }
            }
        }
    }
}

std::vector<float> full_mask(int n) { return std::vector<float>(n, 1.0f); }
std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    // Mask out roughly the second half but keep the first valid.
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;  // also poke a hole near the front
    return m;
}

} // namespace

// Unmasked.
BGA_PARITY_TEST(attn_K2_D16)  { run_attention(2,  16, 0x300ull, nullptr); }
BGA_PARITY_TEST(attn_K8_D16)  { run_attention(8,  16, 0x301ull, nullptr); }
BGA_PARITY_TEST(attn_K8_D64)  { run_attention(8,  64, 0x302ull, nullptr); }
BGA_PARITY_TEST(attn_K16_D64) { run_attention(16, 64, 0x303ull, nullptr); }

// Full-valid mask (should match unmasked).
BGA_PARITY_TEST(attn_K8_D16_full_mask) {
    auto m = full_mask(8); run_attention(8, 16, 0x310ull, &m);
}

// Partial mask.
BGA_PARITY_TEST(attn_K8_D16_partial_mask) {
    auto m = partial_mask(8); run_attention(8, 16, 0x320ull, &m);
}
BGA_PARITY_TEST(attn_K16_D64_partial_mask) {
    auto m = partial_mask(16); run_attention(16, 64, 0x321ull, &m);
}

int main() { return run_all("gpu attention parity"); }
