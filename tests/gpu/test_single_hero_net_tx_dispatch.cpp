// End-to-end GPU dispatch parity test for SingleHeroNetTX. Two identical-seed
// instances; one kept on CPU, one migrated to GPU. The two TransformerEncoder
// stacks dispatch to GPU; the per-slot projections, self MLP, trunk, and
// heads stay on CPU in both cases. Outputs and gradients should match within
// the same tolerances used by other transformer dispatch tests.

#include "parity_helpers.h"

#include <brogameagent/nn/net_tx.h>
#include <brogameagent/observation.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::SingleHeroNetTX;
using brogameagent::nn::NormPlacement;
using brogameagent::nn::Tensor;
namespace obs = brogameagent::observation;

namespace {

SingleHeroNetTX::Config tiny_cfg(uint64_t seed) {
    SingleHeroNetTX::Config c;
    c.self_hidden = 8;
    c.slot_proj   = 8;
    c.d_model     = 8;
    c.d_ff        = 16;
    c.num_heads   = 2;
    c.num_blocks  = 2;
    c.trunk_hidden = 12;
    c.value_hidden = 6;
    c.norm = NormPlacement::PreNorm;
    c.seed = seed;
    return c;
}

void make_obs(Tensor& x, SplitMix64& rng, int n_enemies, int n_allies) {
    x = Tensor::vec(obs::TOTAL);
    x.zero();
    for (int i = 0; i < obs::SELF_FEATURES; ++i) x[i] = rng.next_unit();
    int off = obs::SELF_FEATURES;
    for (int k = 0; k < obs::K_ENEMIES; ++k) {
        const int base = off + k * obs::ENEMY_FEATURES;
        if (k < n_enemies) {
            x[base] = 1.0f;
            for (int j = 1; j < obs::ENEMY_FEATURES; ++j) x[base + j] = rng.next_unit();
        }
    }
    off = obs::SELF_FEATURES + obs::K_ENEMIES * obs::ENEMY_FEATURES;
    for (int k = 0; k < obs::K_ALLIES; ++k) {
        const int base = off + k * obs::ALLY_FEATURES;
        if (k < n_allies) {
            x[base] = 1.0f;
            for (int j = 1; j < obs::ALLY_FEATURES; ++j) x[base + j] = rng.next_unit();
        }
    }
}

void run_dispatch(uint64_t seed) {
    SingleHeroNetTX cpu, gnet;
    cpu.init(tiny_cfg(seed));
    gnet.init(tiny_cfg(seed));

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor x; make_obs(x, rng, 3, 2);
    float dValue = 0.6f;
    Tensor dLogits = Tensor::vec(cpu.policy_logits());
    for (int i = 0; i < dLogits.size(); ++i) dLogits[i] = rng.next_unit() * 0.4f;

    // CPU forward + backward.
    float v_cpu; Tensor l_cpu = Tensor::vec(cpu.policy_logits());
    cpu.zero_grad();
    cpu.forward(x, v_cpu, l_cpu);
    cpu.backward(dValue, dLogits);

    // GPU dispatch.
    gnet.to(Device::GPU);
    BGA_CHECK(gnet.device() == Device::GPU);
    float v_gpu; Tensor l_gpu = Tensor::vec(gnet.policy_logits());
    gnet.zero_grad();
    gnet.forward(x, v_gpu, l_gpu);
    gnet.backward(dValue, dLogits);

    // Outputs.
    BGA_CHECK(std::fabs(v_cpu - v_gpu) < 5e-3f);
    compare_tensors(l_cpu, l_gpu, "tx.dispatch.logits", 1e-4f, 5e-3f);

    // Step both, then migrate gnet back and compare resultant weights.
    cpu.sgd_step(0.01f, 0.9f);
    gnet.sgd_step(0.01f, 0.9f);
    gnet.to(Device::CPU);
    BGA_CHECK(gnet.device() == Device::CPU);
    compare_tensors(cpu.enemy_enc().block(0).mha().Wq(),
                    gnet.enemy_enc().block(0).mha().Wq(),
                    "tx.dispatch.enemy_Wq_after_sgd", 1e-4f, 2e-3f);
    compare_tensors(cpu.trunk().W(), gnet.trunk().W(),
                    "tx.dispatch.trunk_W_after_sgd", 1e-5f, 1e-4f);

    // save/load round-trip after migration. Migrate again to GPU first.
    gnet.to(Device::GPU);
    auto blob = gnet.save();
    SingleHeroNetTX restored;
    restored.init(tiny_cfg(seed ^ 0x12345));   // different init seed
    restored.load(blob);
    compare_tensors(cpu.trunk().W(), restored.trunk().W(),
                    "tx.dispatch.save_load_trunk_W", 1e-5f, 1e-4f);
    compare_tensors(cpu.enemy_enc().block(0).mha().Wq(),
                    restored.enemy_enc().block(0).mha().Wq(),
                    "tx.dispatch.save_load_enemy_Wq", 1e-4f, 2e-3f);
}

void run_smoke_training_gpu(uint64_t seed) {
    SingleHeroNetTX net;
    net.init(tiny_cfg(seed));
    net.to(Device::GPU);

    SplitMix64 rng(seed ^ 0xCAFEull);
    Tensor x; make_obs(x, rng, 2, 2);
    float dValue = -0.4f;
    Tensor dLogits = Tensor::vec(net.policy_logits());
    for (int i = 0; i < dLogits.size(); ++i) dLogits[i] = rng.next_unit() * 0.2f;

    auto compute_loss = [&]() {
        float v; Tensor l = Tensor::vec(net.policy_logits());
        net.forward(x, v, l);
        float L = dValue * v;
        for (int i = 0; i < l.size(); ++i) L += dLogits[i] * l[i];
        return L;
    };

    float L0 = compute_loss();
    BGA_CHECK(std::isfinite(L0));

    for (int s = 1; s <= 20; ++s) {
        net.zero_grad();
        float v; Tensor l = Tensor::vec(net.policy_logits());
        net.forward(x, v, l);
        net.backward(dValue, dLogits);
        net.adam_step(1e-2f, 0.9f, 0.999f, 1e-8f, s);
    }
    float L1 = compute_loss();
    BGA_CHECK(std::isfinite(L1));
    BGA_CHECK(L1 < L0);
}

} // namespace

BGA_PARITY_TEST(tx_dispatch_basic) { run_dispatch(0xA001ull); }
BGA_PARITY_TEST(tx_dispatch_smoke_training_gpu) { run_smoke_training_gpu(0xA002ull); }

int main() { return run_all("gpu SingleHeroNetTX dispatch parity"); }
