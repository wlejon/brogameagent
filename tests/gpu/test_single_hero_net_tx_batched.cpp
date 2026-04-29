// Parity: SingleHeroNetTX::forward_batched matches B sequential single-sample
// GPU forwards on the same net.

#include "parity_helpers.h"

#include <brogameagent/nn/net_tx.h>
#include <brogameagent/observation.h>

#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::Device;
using brogameagent::nn::SingleHeroNetTX;
using brogameagent::nn::gpu::GpuTensor;

namespace obs = brogameagent::observation;

namespace {

// Build a (B, TOTAL) input tensor with random per-row entries, including
// flipping the per-slot validity flag (first feature of each slot block) so
// the masked-mean-pool path is exercised.
Tensor make_batch_inputs(int B, SplitMix64& rng) {
    Tensor X(B, obs::TOTAL);
    for (int b = 0; b < B; ++b) {
        const size_t row = static_cast<size_t>(b) * obs::TOTAL;
        for (int j = 0; j < obs::TOTAL; ++j)
            X.data[row + j] = rng.next_unit() * 0.5f;
        // Self block: first feature is just data, leave random.
        // Enemy slots: first feature in each slot is the validity flag.
        const int off_e = obs::SELF_FEATURES;
        for (int k = 0; k < obs::K_ENEMIES; ++k) {
            const int base = off_e + k * obs::ENEMY_FEATURES;
            X.data[row + base] = (rng.next_f01() > 0.3f) ? 1.0f : 0.0f;
        }
        const int off_a = off_e + obs::K_ENEMIES * obs::ENEMY_FEATURES;
        for (int k = 0; k < obs::K_ALLIES; ++k) {
            const int base = off_a + k * obs::ALLY_FEATURES;
            X.data[row + base] = (rng.next_f01() > 0.4f) ? 1.0f : 0.0f;
        }
    }
    return X;
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

    SingleHeroNetTX net;
    net.init(cfg);
    net.to(Device::GPU);

    SplitMix64 rng(seed ^ 0xBADBA77Dull);
    Tensor X_BD = make_batch_inputs(B, rng);

    const int L = net.logits_dim();

    // Reference: B sequential single-sample forwards on GPU.
    Tensor logits_ref(B, L);
    Tensor values_ref(B, 1);
    for (int b = 0; b < B; ++b) {
        Tensor xb(obs::TOTAL, 1);
        for (int j = 0; j < obs::TOTAL; ++j)
            xb.data[j] = X_BD.data[static_cast<size_t>(b) * obs::TOTAL + j];
        GpuTensor gxb;
        upload(xb, gxb);
        GpuTensor glogits;
        net.forward(gxb, glogits);
        Tensor h_logits = download_to_host(glogits);
        Tensor h_value  = download_to_host(net.value_gpu());
        for (int j = 0; j < L; ++j)
            logits_ref.data[static_cast<size_t>(b) * L + j] = h_logits.data[j];
        values_ref.data[b] = h_value.data[0];
    }

    // Batched.
    GpuTensor gX_BD, glogits_BD, gvalues_B1;
    upload(X_BD, gX_BD);
    net.forward_batched(gX_BD, glogits_BD, gvalues_B1);
    Tensor logits_batched = download_to_host(glogits_BD);
    Tensor values_batched = download_to_host(gvalues_B1);

    BGA_CHECK(logits_batched.rows == B);
    BGA_CHECK(logits_batched.cols == L);
    BGA_CHECK(values_batched.rows == B);
    BGA_CHECK(values_batched.cols == 1);

    compare_tensors(logits_ref, logits_batched, "tx.forward_batched.logits");
    compare_tensors(values_ref, values_batched, "tx.forward_batched.values");
}

} // namespace

BGA_PARITY_TEST(tx_batched_B1)  { run_tx_batched(1,  0x20001ull); }
BGA_PARITY_TEST(tx_batched_B4)  { run_tx_batched(4,  0x20002ull); }
BGA_PARITY_TEST(tx_batched_B16) { run_tx_batched(16, 0x20003ull); }

int main() { return run_all("SingleHeroNetTX forward_batched parity"); }
