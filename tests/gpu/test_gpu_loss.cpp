// CPU↔GPU parity tests for mse_vec and softmax_xent_fused.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/ops.h>

#include <cuda_runtime.h>

#include <cmath>
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

void run_mse(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor pred(n, 1), target(n, 1);
    fill_random(pred, rng);
    fill_random(target, rng);

    // CPU reference: mean of squared diffs.
    float loss_cpu = 0.0f;
    Tensor dPred_cpu(n, 1);
    for (int i = 0; i < n; ++i) {
        const float d = pred.data[i] - target.data[i];
        loss_cpu += d * d;
        dPred_cpu.data[i] = (2.0f / static_cast<float>(n)) * d;
    }
    loss_cpu /= static_cast<float>(n);

    GpuTensor gpred, gtarget, gdPred;
    upload(pred, gpred);
    upload(target, gtarget);

    const float loss_gpu = brogameagent::nn::gpu::mse_vec_forward_gpu(gpred, gtarget);
    brogameagent::nn::gpu::mse_vec_backward_gpu(gpred, gtarget, gdPred);

    Tensor dPred_gpu = download_to_host(gdPred);
    BGA_CHECK(std::fabs(loss_cpu - loss_gpu) < 1e-5f + 1e-4f * std::fabs(loss_cpu));
    compare_tensors(dPred_cpu, dPred_gpu, "mse.dPred");
}

void run_xent(int n, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor logits(n, 1), target(n, 1);
    fill_random(logits, rng);

    // Build a soft-target distribution that sums to 1 over valid entries.
    target.zero();
    float tsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && (*mask)[i] == 0.0f) continue;
        const float v = rng.next_f01() + 0.05f;
        target.data[i] = v;
        tsum += v;
    }
    if (tsum > 0.0f) {
        for (int i = 0; i < n; ++i) target.data[i] /= tsum;
    }

    Tensor probs_cpu(n, 1), dLogits_cpu(n, 1);
    const float loss_cpu = brogameagent::nn::softmax_xent_segment(
        logits.data.data(), target.data.data(),
        probs_cpu.data.data(), dLogits_cpu.data.data(),
        n, mask ? mask->data() : nullptr);

    GpuTensor glogits, gtarget, gprobs, gdLogits;
    upload(logits, glogits);
    upload(target, gtarget);

    float* d_mask = nullptr;
    if (mask) {
        LOCAL_CUDA_OK(cudaMalloc(&d_mask, sizeof(float) * n));
        LOCAL_CUDA_OK(cudaMemcpy(d_mask, mask->data(), sizeof(float) * n,
                                 cudaMemcpyHostToDevice));
    }

    const float loss_gpu = brogameagent::nn::gpu::softmax_xent_fused_gpu(
        glogits, gtarget, d_mask, gprobs, gdLogits);
    Tensor probs_gpu = download_to_host(gprobs);
    Tensor dLogits_gpu = download_to_host(gdLogits);
    if (d_mask) cudaFree(d_mask);

    BGA_CHECK(std::fabs(loss_cpu - loss_gpu) < 1e-5f + 1e-4f * std::fabs(loss_cpu));
    compare_tensors(probs_cpu, probs_gpu, "xent.probs");
    compare_tensors(dLogits_cpu, dLogits_gpu, "xent.dLogits");
}

std::vector<float> mask_all(int n)  { return std::vector<float>(n, 1.0f); }
std::vector<float> mask_half(int n) {
    std::vector<float> m(n, 0.0f);
    for (int i = 0; i < n; ++i) m[i] = (i < n / 2) ? 1.0f : 0.0f;
    if (n / 2 == 0) m[0] = 1.0f;
    return m;
}

} // namespace

BGA_PARITY_TEST(mse_n8)   { run_mse(8,   0x400ull); }
BGA_PARITY_TEST(mse_n64)  { run_mse(64,  0x401ull); }
BGA_PARITY_TEST(mse_n512) { run_mse(512, 0x402ull); }

BGA_PARITY_TEST(xent_unmasked_n8)   { run_xent(8,   0x410ull, nullptr); }
BGA_PARITY_TEST(xent_unmasked_n64)  { run_xent(64,  0x411ull, nullptr); }
BGA_PARITY_TEST(xent_unmasked_n256) { run_xent(256, 0x412ull, nullptr); }
BGA_PARITY_TEST(xent_mask_all_n32)  { auto m = mask_all(32);  run_xent(32, 0x420ull, &m); }
BGA_PARITY_TEST(xent_mask_half_n32) { auto m = mask_half(32); run_xent(32, 0x421ull, &m); }
BGA_PARITY_TEST(xent_mask_half_n128){ auto m = mask_half(128);run_xent(128,0x422ull, &m); }

int main() { return run_all("gpu loss parity"); }
