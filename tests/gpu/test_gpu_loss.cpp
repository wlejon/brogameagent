// CPU↔GPU parity tests for mse_vec and softmax_xent_fused.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <cmath>
#include <vector>

using namespace bga_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_mse(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor pred = Tensor::vec(n), target = Tensor::vec(n);
    fill_random(pred, rng);
    fill_random(target, rng);

    // CPU reference: mean of squared diffs.
    float loss_cpu = 0.0f;
    Tensor dPred_cpu = Tensor::vec(n);
    for (int i = 0; i < n; ++i) {
        const float d = pred[i] - target[i];
        loss_cpu += d * d;
        dPred_cpu[i] = (2.0f / static_cast<float>(n)) * d;
    }
    loss_cpu /= static_cast<float>(n);

    Tensor gpred = pred.to(Device::CUDA);
    Tensor gtarget = target.to(Device::CUDA);
    Tensor gdPred = Tensor::zeros_on(Device::CUDA, n, 1);

    const float loss_gpu = brotensor::mse_vec_forward(gpred, gtarget);
    brotensor::mse_vec_backward(gpred, gtarget, gdPred);

    Tensor dPred_gpu = download_to_host(gdPred);
    BGA_CHECK(std::fabs(loss_cpu - loss_gpu) < 1e-5f + 1e-4f * std::fabs(loss_cpu));
    compare_tensors(dPred_cpu, dPred_gpu, "mse.dPred");
}

void run_xent(int n, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor logits = Tensor::vec(n), target = Tensor::vec(n);
    fill_random(logits, rng);

    // Build a soft-target distribution that sums to 1 over valid entries.
    target.zero();
    float tsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && (*mask)[i] == 0.0f) continue;
        const float v = rng.next_f01() + 0.05f;
        target[i] = v;
        tsum += v;
    }
    if (tsum > 0.0f) {
        for (int i = 0; i < n; ++i) target[i] /= tsum;
    }

    Tensor probs_cpu = Tensor::vec(n), dLogits_cpu = Tensor::vec(n);
    const float loss_cpu = brotensor::softmax_xent_segment(
        logits.ptr(), target.ptr(),
        probs_cpu.ptr(), dLogits_cpu.ptr(),
        n, mask ? mask->data() : nullptr);

    Tensor glogits = logits.to(Device::CUDA);
    Tensor gtarget = target.to(Device::CUDA);
    Tensor gprobs = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdLogits = Tensor::zeros_on(Device::CUDA, n, 1);

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    const float loss_gpu = brotensor::softmax_xent_fused(
        glogits, gtarget, d_mask, gprobs, gdLogits);
    Tensor probs_gpu = download_to_host(gprobs);
    Tensor dLogits_gpu = download_to_host(gdLogits);

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
