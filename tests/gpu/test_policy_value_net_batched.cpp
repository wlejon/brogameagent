// Parity: PolicyValueNet::forward_batched matches B sequential single-sample
// GPU forwards on the same net.

#include "parity_helpers.h"

#include <brogameagent/nn/policy_value_net.h>

#include <vector>

using namespace bga_parity;
using brotensor::Tensor;
using brotensor::Device;
using brogameagent::nn::PolicyValueNet;

static void run_pvn_batched(int B, uint64_t seed) {
    PolicyValueNet::Config cfg;
    cfg.in_dim       = 24;
    cfg.hidden       = {32, 16};
    cfg.value_hidden = 12;
    cfg.num_actions  = 9;
    cfg.seed         = seed;

    PolicyValueNet net;
    net.init(cfg);
    net.to(Device::CUDA);

    SplitMix64 rng(seed ^ 0xBADCAFEull);

    // (B, in_dim) random observations.
    Tensor X_BD = Tensor::mat(B, cfg.in_dim);
    fill_random(X_BD, rng);

    // ── Reference: B sequential single-sample forwards on GPU. ─────────────
    Tensor logits_ref = Tensor::mat(B, cfg.num_actions);
    Tensor values_ref = Tensor::mat(B, 1);
    for (int b = 0; b < B; ++b) {
        Tensor xb = Tensor::vec(cfg.in_dim);
        for (int j = 0; j < cfg.in_dim; ++j)
            xb[j] = X_BD[static_cast<size_t>(b) * cfg.in_dim + j];
        Tensor gxb = xb.to(Device::CUDA);
        Tensor glogits = Tensor::zeros_on(Device::CUDA, cfg.num_actions, 1);
        float gv = 0.0f;
        net.forward(gxb, gv, glogits);
        Tensor h_logits = download_to_host(glogits);
        for (int j = 0; j < cfg.num_actions; ++j)
            logits_ref[static_cast<size_t>(b) * cfg.num_actions + j] =
                h_logits[j];
        values_ref[b] = gv;
    }

    // ── Batched. ───────────────────────────────────────────────────────────
    Tensor gX_BD = X_BD.to(Device::CUDA);
    Tensor glogits_BD = Tensor::zeros_on(Device::CUDA, B, cfg.num_actions);
    Tensor gvalues_B1 = Tensor::zeros_on(Device::CUDA, B, 1);
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
