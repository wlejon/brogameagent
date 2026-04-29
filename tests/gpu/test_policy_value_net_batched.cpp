// Parity: PolicyValueNet::forward_batched matches B sequential single-sample
// GPU forwards on the same net.

#include "parity_helpers.h"

#include <brogameagent/nn/policy_value_net.h>

#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::Device;
using brogameagent::nn::PolicyValueNet;
using brogameagent::nn::gpu::GpuTensor;

static void run_pvn_batched(int B, uint64_t seed) {
    PolicyValueNet::Config cfg;
    cfg.in_dim       = 24;
    cfg.hidden       = {32, 16};
    cfg.value_hidden = 12;
    cfg.num_actions  = 9;
    cfg.seed         = seed;

    PolicyValueNet net;
    net.init(cfg);
    net.to(Device::GPU);

    SplitMix64 rng(seed ^ 0xBADCAFEull);

    // (B, in_dim) random observations.
    Tensor X_BD(B, cfg.in_dim);
    fill_random(X_BD, rng);

    // ── Reference: B sequential single-sample forwards on GPU. ─────────────
    Tensor logits_ref(B, cfg.num_actions);
    Tensor values_ref(B, 1);
    for (int b = 0; b < B; ++b) {
        Tensor xb(cfg.in_dim, 1);
        for (int j = 0; j < cfg.in_dim; ++j)
            xb.data[j] = X_BD.data[static_cast<size_t>(b) * cfg.in_dim + j];
        GpuTensor gxb;
        upload(xb, gxb);
        GpuTensor glogits;
        net.forward(gxb, glogits);
        Tensor h_logits = download_to_host(glogits);
        Tensor h_value  = download_to_host(net.value_gpu());
        for (int j = 0; j < cfg.num_actions; ++j)
            logits_ref.data[static_cast<size_t>(b) * cfg.num_actions + j] =
                h_logits.data[j];
        values_ref.data[b] = h_value.data[0];
    }

    // ── Batched. ───────────────────────────────────────────────────────────
    GpuTensor gX_BD, glogits_BD, gvalues_B1;
    upload(X_BD, gX_BD);
    net.forward_batched(gX_BD, glogits_BD, gvalues_B1);
    Tensor logits_batched = download_to_host(glogits_BD);
    Tensor values_batched = download_to_host(gvalues_B1);

    BGA_CHECK(logits_batched.rows == B);
    BGA_CHECK(logits_batched.cols == cfg.num_actions);
    BGA_CHECK(values_batched.rows == B);
    BGA_CHECK(values_batched.cols == 1);

    compare_tensors(logits_ref, logits_batched, "pvn.forward_batched.logits");
    compare_tensors(values_ref, values_batched, "pvn.forward_batched.values");
}

BGA_PARITY_TEST(pvn_batched_B1)  { run_pvn_batched(1,  0x10001ull); }
BGA_PARITY_TEST(pvn_batched_B4)  { run_pvn_batched(4,  0x10002ull); }
BGA_PARITY_TEST(pvn_batched_B64) { run_pvn_batched(64, 0x10003ull); }

int main() { return run_all("PolicyValueNet forward_batched parity"); }
