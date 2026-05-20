// Parity: SingleHeroNetTX single-sample GPU forwards match the CPU reference
// across a batch of inputs.
//
// SingleHeroNetTX no longer exposes GPU-only batched forward kernels (they
// were removed in the unified-brotensor migration). This test now drives the
// net through its single-sample forward over a batch of B inputs and checks
// CPU↔CUDA dispatch parity on every row.

#include "parity_helpers.h"

#include <brogameagent/nn/net_tx.h>
#include <brotensor/runtime.h>
#include <brogameagent/observation.h>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace bga_parity;
using brotensor::Tensor;
using brotensor::Device;
using brogameagent::nn::SingleHeroNetTX;

namespace obs = brogameagent::observation;

namespace {

// Build a (B, TOTAL) input tensor with random per-row entries, including
// flipping the per-slot validity flag (first feature of each slot block) so
// the masked-mean-pool path is exercised.
Tensor make_batch_inputs(int B, SplitMix64& rng) {
    Tensor X = Tensor::mat(B, obs::TOTAL);
    for (int b = 0; b < B; ++b) {
        const size_t row = static_cast<size_t>(b) * obs::TOTAL;
        for (int j = 0; j < obs::TOTAL; ++j)
            X[row + j] = rng.next_unit() * 0.5f;
        // Self block: first feature is just data, leave random.
        // Enemy slots: first feature in each slot is the validity flag.
        const int off_e = obs::SELF_FEATURES;
        for (int k = 0; k < obs::K_ENEMIES; ++k) {
            const int base = off_e + k * obs::ENEMY_FEATURES;
            X[row + base] = (rng.next_f01() > 0.3f) ? 1.0f : 0.0f;
        }
        const int off_a = off_e + obs::K_ENEMIES * obs::ENEMY_FEATURES;
        for (int k = 0; k < obs::K_ALLIES; ++k) {
            const int base = off_a + k * obs::ALLY_FEATURES;
            X[row + base] = (rng.next_f01() > 0.4f) ? 1.0f : 0.0f;
        }
    }
    return X;
}

// Extract row b of a (B, TOTAL) host tensor into a (TOTAL, 1) host vector.
Tensor row_vec(const Tensor& X_BD, int b) {
    Tensor xb = Tensor::vec(obs::TOTAL);
    for (int j = 0; j < obs::TOTAL; ++j)
        xb[j] = X_BD[static_cast<size_t>(b) * obs::TOTAL + j];
    return xb;
}

void run_tx_batched(int B, uint64_t seed) {
    SingleHeroNetTX::Config cfg;
    cfg.d_model      = 16;
    cfg.d_ff         = 32;
    cfg.num_heads    = 2;
    cfg.num_blocks   = 1;
    cfg.self_hidden  = 16;
    cfg.trunk_hidden = 32;
    cfg.value_hidden = 16;
    cfg.seed         = seed;

    // CPU reference net + GPU net seeded identically.
    SingleHeroNetTX cpu_net, gpu_net;
    cpu_net.init(cfg);
    gpu_net.init(cfg);
    gpu_net.to(Device::CUDA);

    SplitMix64 rng(seed ^ 0xBADBA77Dull);
    Tensor X_BD = make_batch_inputs(B, rng);

    const int L = gpu_net.logits_dim();

    // Reference: B single-sample forwards on CPU.
    Tensor logits_ref = Tensor::mat(B, L);
    Tensor values_ref = Tensor::mat(B, 1);
    for (int b = 0; b < B; ++b) {
        Tensor xb = row_vec(X_BD, b);
        float v = 0.0f;
        Tensor l = Tensor::vec(L);
        cpu_net.forward(xb, v, l);
        for (int j = 0; j < L; ++j)
            logits_ref[static_cast<size_t>(b) * L + j] = l[j];
        values_ref[b] = v;
    }

    // GPU: B single-sample forwards on CUDA.
    Tensor logits_gpu = Tensor::mat(B, L);
    Tensor values_gpu = Tensor::mat(B, 1);
    for (int b = 0; b < B; ++b) {
        Tensor xb = row_vec(X_BD, b);
        float v = 0.0f;
        Tensor l = Tensor::vec(L);
        gpu_net.forward(xb, v, l);
        Tensor l_h = l.to(Device::CPU);
        for (int j = 0; j < L; ++j)
            logits_gpu[static_cast<size_t>(b) * L + j] = l_h[j];
        values_gpu[b] = v;
    }

    BGA_CHECK(logits_gpu.rows == B);
    BGA_CHECK(logits_gpu.cols == L);
    BGA_CHECK(values_gpu.rows == B);
    BGA_CHECK(values_gpu.cols == 1);

    compare_tensors(logits_ref, logits_gpu, "tx.batched.logits", 1e-4f, 5e-3f);
    compare_tensors(values_ref, values_gpu, "tx.batched.values", 1e-4f, 5e-3f);
}

} // namespace

BGA_PARITY_TEST(tx_batched_B1)  { run_tx_batched(1,  0x20001ull); }
BGA_PARITY_TEST(tx_batched_B4)  { run_tx_batched(4,  0x20002ull); }
BGA_PARITY_TEST(tx_batched_B16) { run_tx_batched(16, 0x20003ull); }

// Smoke bench: B single-sample GPU forwards vs B single-sample CPU forwards.
// Prints both wall-times. Doesn't assert a specific ratio — environments
// vary — but the GPU path should be measurably faster for a larger net.
BGA_PARITY_TEST(tx_batched_speedup_smoke) {
    const int B = 64;
    const int N_iters = 32;

    SingleHeroNetTX::Config cfg;
    cfg.d_model = 32; cfg.d_ff = 64; cfg.num_heads = 4; cfg.num_blocks = 2;
    cfg.self_hidden = 32; cfg.trunk_hidden = 64; cfg.value_hidden = 32;
    cfg.seed = 0xBE7C7Bull;

    SingleHeroNetTX cpu_net, gpu_net;
    cpu_net.init(cfg);
    gpu_net.init(cfg);
    gpu_net.to(Device::CUDA);

    SplitMix64 rng(0xBE7C7Bull ^ 0xFEEDull);
    Tensor X_BD = make_batch_inputs(B, rng);
    const int L = gpu_net.logits_dim();

    // Warmup.
    for (int i = 0; i < 4; ++i) {
        for (int b = 0; b < B; ++b) {
            Tensor xb = row_vec(X_BD, b);
            float v = 0.0f;
            Tensor l = Tensor::vec(L);
            gpu_net.forward(xb, v, l);
        }
    }
    brotensor::sync_all();

    // Time GPU single-sample forwards.
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_iters; ++i) {
        for (int b = 0; b < B; ++b) {
            Tensor xb = row_vec(X_BD, b);
            float v = 0.0f;
            Tensor l = Tensor::vec(L);
            gpu_net.forward(xb, v, l);
        }
    }
    brotensor::sync_all();
    auto t1 = std::chrono::steady_clock::now();
    const double gpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Time CPU single-sample forwards.
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_iters; ++i) {
        for (int b = 0; b < B; ++b) {
            Tensor xb = row_vec(X_BD, b);
            float v = 0.0f;
            Tensor l = Tensor::vec(L);
            cpu_net.forward(xb, v, l);
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    const double speedup = cpu_ms / std::max(gpu_ms, 1e-9);
    std::printf("\n[bench] B=%d iters=%d  cpu=%.2f ms  gpu=%.2f ms  "
                "speedup=%.2fx\n", B, N_iters, cpu_ms, gpu_ms, speedup);
}

int main() { return run_all("SingleHeroNetTX single-sample dispatch parity"); }
