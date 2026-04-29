// CPU↔GPU parity tests for softmax_forward / softmax_backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/ops.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_softmax(int n, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor logits(n, 1), dProbs(n, 1);
    fill_random(logits, rng);
    fill_random(dProbs, rng);

    Tensor probs_cpu(n, 1);
    brogameagent::nn::softmax_forward(logits, probs_cpu, mask ? mask->data() : nullptr);
    Tensor dLogits_cpu(n, 1);
    brogameagent::nn::softmax_backward(probs_cpu, dProbs, dLogits_cpu);

    // GPU.
    GpuTensor glogits, gprobs, gdProbs, gdLogits;
    upload(logits, glogits);
    upload(dProbs, gdProbs);
    gprobs.resize(n, 1);
    gdLogits.resize(n, 1);

    float* d_mask = nullptr;
    if (mask) {
        cudaMalloc(&d_mask, sizeof(float) * n);
        cudaMemcpy(d_mask, mask->data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    }
    brogameagent::nn::gpu::softmax_forward_gpu(glogits, gprobs, d_mask);
    Tensor probs_gpu = download_to_host(gprobs);
    brogameagent::nn::gpu::softmax_backward_gpu(gprobs, gdProbs, gdLogits);
    Tensor dLogits_gpu = download_to_host(gdLogits);
    if (d_mask) cudaFree(d_mask);

    compare_tensors(probs_cpu, probs_gpu, "softmax_forward");
    compare_tensors(dLogits_cpu, dLogits_gpu, "softmax_backward");

    // For masked variant, also assert: invalid entries are exactly 0,
    // valid entries sum to 1.
    if (mask) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) {
            if ((*mask)[i] < 0.5f) {
                BGA_CHECK(probs_gpu.data[i] == 0.0f);
            } else {
                s += probs_gpu.data[i];
            }
        }
        BGA_CHECK(std::fabs(s - 1.0f) < 1e-5f);
    }
}

std::vector<float> mask_all(int n)        { return std::vector<float>(n, 1.0f); }
std::vector<float> mask_half(int n) {
    std::vector<float> m(n, 0.0f);
    for (int i = 0; i < n; ++i) m[i] = (i < n / 2) ? 1.0f : 0.0f;
    if (n / 2 == 0) m[0] = 1.0f;  // guarantee at least one valid
    return m;
}
std::vector<float> mask_one(int n) {
    std::vector<float> m(n, 0.0f);
    m[n / 2] = 1.0f;
    return m;
}

} // namespace

// Unmasked.
BGA_PARITY_TEST(softmax_unmasked_n8)   { run_softmax(8,   0x100ull, nullptr); }
BGA_PARITY_TEST(softmax_unmasked_n32)  { run_softmax(32,  0x101ull, nullptr); }
BGA_PARITY_TEST(softmax_unmasked_n128) { run_softmax(128, 0x102ull, nullptr); }

// All-valid mask should match unmasked.
BGA_PARITY_TEST(softmax_mask_all_n32) { auto m = mask_all(32); run_softmax(32, 0x110ull, &m); }
// Half-valid.
BGA_PARITY_TEST(softmax_mask_half_n8)   { auto m = mask_half(8);   run_softmax(8,   0x120ull, &m); }
BGA_PARITY_TEST(softmax_mask_half_n32)  { auto m = mask_half(32);  run_softmax(32,  0x121ull, &m); }
BGA_PARITY_TEST(softmax_mask_half_n128) { auto m = mask_half(128); run_softmax(128, 0x122ull, &m); }
// Single-valid.
BGA_PARITY_TEST(softmax_mask_one_n8)   { auto m = mask_one(8);   run_softmax(8,   0x130ull, &m); }
BGA_PARITY_TEST(softmax_mask_one_n128) { auto m = mask_one(128); run_softmax(128, 0x131ull, &m); }

int main() { return run_all("gpu softmax parity"); }
