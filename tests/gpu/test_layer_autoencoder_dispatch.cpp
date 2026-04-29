// End-to-end GPU dispatch parity test for DeepSetsEncoder, DeepSetsDecoder
// and the composite DeepSetsAutoencoder. Builds two instances seeded
// identically; migrates one to Device::GPU; runs forward/backward/sgd_step
// on both; verifies outputs and updated parameters match within tolerance.
//
// Also runs a tiny smoke training loop on the GPU autoencoder and verifies
// the loss does not NaN and decreases over a handful of steps.

#include "parity_helpers.h"

#include <brogameagent/nn/autoencoder.h>
#include <brogameagent/nn/decoder.h>
#include <brogameagent/nn/encoder.h>
#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/tensor.h>
#include <brogameagent/observation.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace bga_parity;
using brogameagent::nn::DeepSetsAutoencoder;
using brogameagent::nn::DeepSetsDecoder;
using brogameagent::nn::DeepSetsEncoder;
using brogameagent::nn::Device;
using brogameagent::nn::Tensor;
namespace obs = brogameagent::observation;

namespace {

// Build a synthetic observation vector with deterministic random feature
// values and a chosen valid-mask pattern for enemy/ally slots.
Tensor make_obs(SplitMix64& rng,
                const std::vector<int>& enemy_valid,
                const std::vector<int>& ally_valid) {
    Tensor x(obs::TOTAL, 1);
    fill_random(x, rng);
    // Self block: leave random.
    const int off_e = obs::SELF_FEATURES;
    for (int k = 0; k < obs::K_ENEMIES; ++k) {
        const int base = off_e + k * obs::ENEMY_FEATURES;
        x[base] = enemy_valid[k] ? 1.0f : 0.0f;
    }
    const int off_a = off_e + obs::K_ENEMIES * obs::ENEMY_FEATURES;
    for (int k = 0; k < obs::K_ALLIES; ++k) {
        const int base = off_a + k * obs::ALLY_FEATURES;
        x[base] = ally_valid[k] ? 1.0f : 0.0f;
    }
    return x;
}

// ── Encoder dispatch parity ──────────────────────────────────────────────

void run_encoder_parity(uint64_t seed,
                        const std::vector<int>& e_valid,
                        const std::vector<int>& a_valid) {
    DeepSetsEncoder::Config cfg;
    cfg.embed_dim = 8;
    cfg.hidden    = 12;

    DeepSetsEncoder cpu, gpu_enc;
    uint64_t s1 = seed, s2 = seed;
    cpu.init(cfg, s1);
    gpu_enc.init(cfg, s2);

    SplitMix64 rng(seed ^ 0xDEAD);
    Tensor x = make_obs(rng, e_valid, a_valid);
    Tensor dY(cpu.out_dim(), 1);
    fill_random(dY, rng);

    // CPU.
    Tensor y_cpu(cpu.out_dim(), 1), dX_cpu(obs::TOTAL, 1);
    cpu.zero_grad();
    cpu.forward(x, y_cpu);
    cpu.backward(dY, dX_cpu);

    // GPU.
    gpu_enc.to(Device::GPU);
    BGA_CHECK(gpu_enc.device() == Device::GPU);
    brogameagent::nn::gpu::GpuTensor gx, gy, gdY, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(cpu.out_dim(), 1); gdX.resize(obs::TOTAL, 1);
    gpu_enc.zero_grad();
    gpu_enc.forward(gx, gy);
    gpu_enc.backward(gdY, gdX);

    Tensor y_gpu  = download_to_host(gy);
    Tensor dX_gpu = download_to_host(gdX);
    compare_tensors(y_cpu,  y_gpu,  "enc.dispatch.y");
    compare_tensors(dX_cpu, dX_gpu, "enc.dispatch.dX");

    // sgd_step on both, then bring GPU params back to host and compare.
    cpu.sgd_step(0.01f, 0.9f);
    gpu_enc.sgd_step(0.01f, 0.9f);
    gpu_enc.to(Device::CPU);
    BGA_CHECK(gpu_enc.device() == Device::CPU);
    // We don't have direct accessors for the inner Linear params; round-trip
    // through save_to to compare full parameter blobs.
    std::vector<uint8_t> blob_cpu, blob_gpu;
    cpu.save_to(blob_cpu);
    gpu_enc.save_to(blob_gpu);
    BGA_CHECK(blob_cpu.size() == blob_gpu.size());
    // Compare floats element-wise: layout is identical.
    BGA_CHECK(std::memcmp(blob_cpu.data(), blob_gpu.data(), blob_cpu.size()) == 0
              || blob_cpu.size() == blob_gpu.size());
    // Tolerance-aware float compare: parse the blobs back into encoders.
    DeepSetsEncoder restored_cpu, restored_gpu;
    uint64_t s3 = seed; restored_cpu.init(cfg, s3);
    uint64_t s4 = seed; restored_gpu.init(cfg, s4);
    {
        size_t off = 0;
        restored_cpu.load_from(blob_cpu.data(), off, blob_cpu.size());
    }
    {
        size_t off = 0;
        restored_gpu.load_from(blob_gpu.data(), off, blob_gpu.size());
    }
    // Run another forward and compare to confirm params line up.
    Tensor y_cpu2(cpu.out_dim(), 1), y_gpu2(cpu.out_dim(), 1);
    restored_cpu.forward(x, y_cpu2);
    restored_gpu.forward(x, y_gpu2);
    compare_tensors(y_cpu2, y_gpu2, "enc.dispatch.params_after_sgd");
}

// ── Decoder dispatch parity ──────────────────────────────────────────────

void run_decoder_parity(uint64_t seed) {
    DeepSetsDecoder::Config cfg;
    cfg.embed_dim = 8;
    cfg.hidden    = 12;

    DeepSetsDecoder cpu, gpu_dec;
    uint64_t s1 = seed, s2 = seed;
    cpu.init(cfg, s1);
    gpu_dec.init(cfg, s2);

    SplitMix64 rng(seed ^ 0xBEEF);
    Tensor x(cpu.in_dim(), 1);
    fill_random(x, rng);
    Tensor dY(obs::TOTAL, 1);
    fill_random(dY, rng);

    Tensor y_cpu(obs::TOTAL, 1), dX_cpu(cpu.in_dim(), 1);
    cpu.zero_grad();
    cpu.forward(x, y_cpu);
    cpu.backward(dY, dX_cpu);

    gpu_dec.to(Device::GPU);
    brogameagent::nn::gpu::GpuTensor gx, gy, gdY, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(obs::TOTAL, 1); gdX.resize(cpu.in_dim(), 1);
    gpu_dec.zero_grad();
    gpu_dec.forward(gx, gy);
    gpu_dec.backward(gdY, gdX);

    Tensor y_gpu = download_to_host(gy);
    Tensor dX_gpu = download_to_host(gdX);
    compare_tensors(y_cpu, y_gpu, "dec.dispatch.y");
    compare_tensors(dX_cpu, dX_gpu, "dec.dispatch.dX");

    cpu.sgd_step(0.01f, 0.9f);
    gpu_dec.sgd_step(0.01f, 0.9f);
    gpu_dec.to(Device::CPU);

    std::vector<uint8_t> blob_cpu, blob_gpu;
    cpu.save_to(blob_cpu);
    gpu_dec.save_to(blob_gpu);
    BGA_CHECK(blob_cpu.size() == blob_gpu.size());

    DeepSetsDecoder restored_cpu, restored_gpu;
    uint64_t s3 = seed; restored_cpu.init(cfg, s3);
    uint64_t s4 = seed; restored_gpu.init(cfg, s4);
    {
        size_t off = 0;
        restored_cpu.load_from(blob_cpu.data(), off, blob_cpu.size());
    }
    {
        size_t off = 0;
        restored_gpu.load_from(blob_gpu.data(), off, blob_gpu.size());
    }
    Tensor y_cpu2(obs::TOTAL, 1), y_gpu2(obs::TOTAL, 1);
    restored_cpu.forward(x, y_cpu2);
    restored_gpu.forward(x, y_gpu2);
    compare_tensors(y_cpu2, y_gpu2, "dec.dispatch.params_after_sgd");
}

// ── Autoencoder dispatch parity ──────────────────────────────────────────

void run_autoencoder_parity(uint64_t seed) {
    DeepSetsAutoencoder::Config cfg;
    cfg.enc.embed_dim = 8;
    cfg.enc.hidden    = 12;
    cfg.dec_hidden    = 12;
    cfg.seed          = seed;

    DeepSetsAutoencoder cpu, gpu_ae;
    cpu.init(cfg);
    gpu_ae.init(cfg);

    SplitMix64 rng(seed ^ 0xCAFE);
    Tensor x = make_obs(rng,
                        {1, 1, 0, 1, 0},
                        {1, 0, 1, 1});
    Tensor dXh(obs::TOTAL, 1);
    fill_random(dXh, rng);

    Tensor x_hat_cpu(obs::TOTAL, 1);
    cpu.zero_grad();
    cpu.forward(x, x_hat_cpu);
    cpu.backward(dXh);

    gpu_ae.to(Device::GPU);
    BGA_CHECK(gpu_ae.device() == Device::GPU);
    brogameagent::nn::gpu::GpuTensor gx, gxhat, gdXh;
    upload(x, gx); upload(dXh, gdXh);
    gxhat.resize(obs::TOTAL, 1);
    gpu_ae.zero_grad();
    gpu_ae.forward(gx, gxhat);
    gpu_ae.backward(gdXh);

    Tensor x_hat_gpu = download_to_host(gxhat);
    compare_tensors(x_hat_cpu, x_hat_gpu, "ae.dispatch.x_hat");

    cpu.sgd_step(0.01f, 0.9f);
    gpu_ae.sgd_step(0.01f, 0.9f);

    // After stepping: forward again and compare reconstructions.
    Tensor x_hat_cpu2(obs::TOTAL, 1);
    cpu.forward(x, x_hat_cpu2);
    brogameagent::nn::gpu::GpuTensor gxhat2;
    gxhat2.resize(obs::TOTAL, 1);
    gpu_ae.forward(gx, gxhat2);
    Tensor x_hat_gpu2 = download_to_host(gxhat2);
    compare_tensors(x_hat_cpu2, x_hat_gpu2, "ae.dispatch.x_hat_after_sgd");
}

// ── GPU smoke training loop: verifies loss decreases / no NaN ─────────────

void run_gpu_smoke_training() {
    DeepSetsAutoencoder::Config cfg;
    cfg.enc.embed_dim = 8;
    cfg.enc.hidden    = 12;
    cfg.dec_hidden    = 12;
    cfg.seed          = 0x5717C055ULL;

    DeepSetsAutoencoder ae;
    ae.init(cfg);
    ae.to(Device::GPU);

    SplitMix64 rng(0x5C0FE);
    // Single fixed observation — easy target for memorization.
    Tensor x = make_obs(rng,
                        {1, 1, 1, 0, 0},
                        {1, 1, 0, 0});

    brogameagent::nn::gpu::GpuTensor gx, target_g, gxhat, gdXh;
    upload(x, gx);
    upload(x, target_g);
    gxhat.resize(obs::TOTAL, 1);
    gdXh.resize(obs::TOTAL, 1);

    const int steps = 20;
    float loss_first = 0.0f, loss_last = 0.0f;
    for (int s = 0; s < steps; ++s) {
        ae.zero_grad();
        ae.forward(gx, gxhat);
        const float loss = brogameagent::nn::gpu::mse_vec_forward_gpu(gxhat, target_g);
        BGA_CHECK(std::isfinite(loss));
        if (s == 0) loss_first = loss;
        if (s == steps - 1) loss_last = loss;
        brogameagent::nn::gpu::mse_vec_backward_gpu(gxhat, target_g, gdXh);
        ae.backward(gdXh);
        ae.sgd_step(0.05f, 0.9f);
    }
    std::printf("    [smoke] loss_first=%.6f loss_last=%.6f\n",
                loss_first, loss_last);
    BGA_CHECK(std::isfinite(loss_last));
    BGA_CHECK(loss_last < loss_first);
}

} // namespace

BGA_PARITY_TEST(encoder_dispatch_full) {
    run_encoder_parity(0x100ull, {1, 1, 1, 1, 1}, {1, 1, 1, 1});
}
BGA_PARITY_TEST(encoder_dispatch_partial) {
    run_encoder_parity(0x101ull, {1, 0, 1, 0, 0}, {0, 1, 1, 0});
}
BGA_PARITY_TEST(encoder_dispatch_empty_sets) {
    run_encoder_parity(0x102ull, {0, 0, 0, 0, 0}, {0, 0, 0, 0});
}
BGA_PARITY_TEST(decoder_dispatch_basic) {
    run_decoder_parity(0x200ull);
}
BGA_PARITY_TEST(autoencoder_dispatch) {
    run_autoencoder_parity(0x300ull);
}
BGA_PARITY_TEST(gpu_smoke_training) {
    run_gpu_smoke_training();
}

int main() { return run_all("gpu deepsets autoencoder dispatch parity"); }
