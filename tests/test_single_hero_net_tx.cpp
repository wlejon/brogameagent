// CPU correctness for SingleHeroNetTX:
//   - forward smoke: shapes, value range
//   - finite-difference gradient check (small config)
//   - param count sanity
//   - save/load round-trip

#include <brogameagent/nn/net_tx.h>
#include <brogameagent/observation.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using brogameagent::nn::SingleHeroNetTX;
using brogameagent::nn::Tensor;
using brogameagent::nn::NormPlacement;
namespace obs = brogameagent::observation;

namespace {

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_f01() { return static_cast<float>(next_u64() >> 40) / 16777216.0f; }
    float next_unit() { return next_f01() * 2.0f - 1.0f; }
};

int g_failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){ std::printf("    FAIL %s: %s\n", __func__, msg); ++g_failures; } } while(0)

SingleHeroNetTX::Config small_cfg(uint64_t seed = 42) {
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
    // self block
    for (int i = 0; i < obs::SELF_FEATURES; ++i) x[i] = rng.next_unit();
    // enemies
    int off = obs::SELF_FEATURES;
    for (int k = 0; k < obs::K_ENEMIES; ++k) {
        const int base = off + k * obs::ENEMY_FEATURES;
        if (k < n_enemies) {
            x[base] = 1.0f; // valid flag
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

void test_forward_shapes_smoke() {
    SingleHeroNetTX net;
    net.init(small_cfg(0xA1));
    SplitMix64 rng(0xB1);
    Tensor x;
    make_obs(x, rng, 3, 2);

    float v = 0;
    Tensor logits = Tensor::vec(net.policy_logits());
    net.forward(x, v, logits);
    CHECK(std::isfinite(v), "value finite");
    CHECK(v >= -1.0f && v <= 1.0f, "value in [-1,1]");
    CHECK(logits.size() == net.policy_logits(), "logits size");
    for (int i = 0; i < logits.size(); ++i) CHECK(std::isfinite(logits[i]), "logits finite");
    CHECK(net.num_params() > 0, "params > 0");
}

void test_param_count_matches_children() {
    SingleHeroNetTX net;
    net.init(small_cfg(0xA2));
    int sum = 0;
    sum += net.self_fc1().num_params() + net.self_fc2().num_params();
    sum += net.enemy_proj().num_params() + net.enemy_enc().num_params();
    sum += net.ally_proj().num_params()  + net.ally_enc().num_params();
    sum += net.trunk().num_params();
    sum += net.value_head().num_params() + net.policy_head().num_params();
    CHECK(sum == net.num_params(), "child sum == net.num_params()");
}

float run_loss(SingleHeroNetTX& net, const Tensor& x,
               const Tensor& dLogits, float dValue,
               float& v, Tensor& logits) {
    v = 0;
    logits.zero();
    net.forward(x, v, logits);
    // synthetic scalar loss = dValue*v + sum dLogits[i]*logits[i].
    float loss = dValue * v;
    for (int i = 0; i < logits.size(); ++i) loss += dLogits[i] * logits[i];
    return loss;
}

void test_finite_difference_gradient() {
    // We perturb a few weights in small layers and confirm dL/dw matches.
    SingleHeroNetTX net;
    auto cfg = small_cfg(0xA3);
    net.init(cfg);

    SplitMix64 rng(0xC3);
    Tensor x;
    make_obs(x, rng, 2, 2);

    // Pick a fixed upstream gradient.
    float dValue = 0.7f;
    Tensor dLogits = Tensor::vec(net.policy_logits());
    for (int i = 0; i < dLogits.size(); ++i) dLogits[i] = rng.next_unit() * 0.5f;

    // Forward + analytic backward.
    float v_ref = 0;
    Tensor logits = Tensor::vec(net.policy_logits());
    net.zero_grad();
    net.forward(x, v_ref, logits);
    net.backward(dValue, dLogits);

    // Pick a target: trunk_.W() — touches all upstream paths.
    auto& W = net.trunk().W();
    auto& dW = net.trunk().dW();
    const int n = W.size();
    const int probe_count = 5;
    const float h = 1e-3f;
    int max_idx = -1;
    float max_rel = 0.0f, max_abs_anal = 0.0f, max_abs_num = 0.0f;
    for (int p = 0; p < probe_count; ++p) {
        const int idx = (int)(rng.next_u64() % (uint64_t)n);
        const float orig = W.data[idx];
        W.data[idx] = orig + h;
        float v1; Tensor lo1 = Tensor::vec(net.policy_logits());
        net.forward(x, v1, lo1);
        float lp = dValue * v1;
        for (int i = 0; i < lo1.size(); ++i) lp += dLogits[i] * lo1[i];

        W.data[idx] = orig - h;
        float v2; Tensor lo2 = Tensor::vec(net.policy_logits());
        net.forward(x, v2, lo2);
        float lm = dValue * v2;
        for (int i = 0; i < lo2.size(); ++i) lm += dLogits[i] * lo2[i];

        W.data[idx] = orig;

        const float num = (lp - lm) / (2.0f * h);
        const float anal = dW.data[idx];
        const float diff = std::fabs(num - anal);
        const float denom = std::fabs(num) + std::fabs(anal) + 1e-6f;
        const float rel = diff / denom;
        if (rel > max_rel) {
            max_rel = rel; max_idx = idx;
            max_abs_anal = anal; max_abs_num = num;
        }
    }
    if (!(max_rel < 1e-2f)) {
        std::printf("    FD probe failed: idx=%d  num=%.7g anal=%.7g  rel=%.3g\n",
                    max_idx, max_abs_num, max_abs_anal, max_rel);
    }
    CHECK(max_rel < 1e-2f, "trunk W finite-diff gradient matches analytic");
}

void test_save_load_roundtrip() {
    SingleHeroNetTX a, b;
    a.init(small_cfg(0xD1));
    b.init(small_cfg(0xD2));   // different seed → different weights
    SplitMix64 rng(0xE1);
    Tensor x; make_obs(x, rng, 3, 1);
    float va = 0, vb = 0;
    Tensor la = Tensor::vec(a.policy_logits());
    Tensor lb = Tensor::vec(b.policy_logits());
    a.forward(x, va, la);
    b.forward(x, vb, lb);

    auto blob = a.save();
    SingleHeroNetTX c;
    c.init(small_cfg(0xD2));
    c.load(blob);

    float vc = 0;
    Tensor lc = Tensor::vec(c.policy_logits());
    c.forward(x, vc, lc);
    CHECK(std::fabs(vc - va) < 1e-5f, "value matches after save/load");
    bool logits_match = true;
    for (int i = 0; i < la.size(); ++i) {
        if (std::fabs(la[i] - lc[i]) > 1e-5f) { logits_match = false; break; }
    }
    CHECK(logits_match, "logits match after save/load");
}

void test_smoke_training() {
    // Tiny CPU SGD run; verify loss decreases.
    SingleHeroNetTX net;
    net.init(small_cfg(0xF1));
    SplitMix64 rng(0xF2);
    Tensor x; make_obs(x, rng, 2, 2);
    float dValue = -0.5f;
    Tensor dLogits = Tensor::vec(net.policy_logits());
    for (int i = 0; i < dLogits.size(); ++i) dLogits[i] = rng.next_unit() * 0.2f;

    auto loss_fn = [&](float& v_out, Tensor& l_out) {
        net.forward(x, v_out, l_out);
        float L = dValue * v_out;
        for (int i = 0; i < l_out.size(); ++i) L += dLogits[i] * l_out[i];
        return L;
    };

    float v0; Tensor l0 = Tensor::vec(net.policy_logits());
    float L_before = loss_fn(v0, l0);

    for (int s = 0; s < 20; ++s) {
        net.zero_grad();
        float v; Tensor l = Tensor::vec(net.policy_logits());
        net.forward(x, v, l);
        net.backward(dValue, dLogits);
        net.sgd_step(0.01f, 0.9f);
    }
    float v1; Tensor l1 = Tensor::vec(net.policy_logits());
    float L_after = loss_fn(v1, l1);
    CHECK(std::isfinite(L_after), "loss finite after training");
    CHECK(L_after < L_before, "loss decreased after 20 SGD steps");
}

} // namespace

int main() {
    std::printf("SingleHeroNetTX CPU correctness\n");
    test_forward_shapes_smoke();
    test_param_count_matches_children();
    test_finite_difference_gradient();
    test_save_load_roundtrip();
    test_smoke_training();
    std::printf("%s\n", g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
