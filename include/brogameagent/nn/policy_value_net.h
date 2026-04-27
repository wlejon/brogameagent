#pragma once

#include "circuits.h"
#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── PolicyValueNet ───────────────────────────────────────────────────────
//
// A small, hand-coded MLP with a value head and a single (flat) policy head,
// decoupled from the MOBA-shaped observation/action layout that SingleHeroNet
// assumes. This is the "bring your own observation, bring your own action
// space" net: useful for any small-state-space discrete-action problem
// (platformers, puzzle games, gridworlds, etc.).
//
// Architecture:
//   trunk   : in_dim → hidden[0] → ReLU → hidden[1] → ReLU → ... → hidden[n-1]
//   value   : hidden[n-1] → value_hidden → ReLU → 1 → tanh        (in [-1, 1])
//   policy  : hidden[n-1] → num_actions                            (raw logits)
//
// Forward returns (value, logits). Backward expects (dValue, dLogits) where
// dLogits is gradient on the raw logits — typically (probs - target) from a
// masked softmax-xent (use nn::softmax_xent with the legal-action mask).
//
// Wire format: distinct magic from SingleHeroNet so we can't mix them up.

class PolicyValueNet {
public:
    struct Config {
        int in_dim = 0;                          // observation length
        std::vector<int> hidden = {64, 64};      // trunk hidden widths
        int value_hidden = 32;
        int num_actions = 0;                     // policy head output width
        uint64_t seed = 0xC0DE1234ULL;
    };

    PolicyValueNet() = default;
    void init(const Config& cfg);
    const Config& config() const { return cfg_; }

    // Forward + backward. logits.size() must equal num_actions.
    void forward(const Tensor& x, float& value, Tensor& logits);
    void backward(float dValue, const Tensor& dLogits);

    void zero_grad();
    void sgd_step(float lr, float momentum);

    int in_dim()       const { return cfg_.in_dim; }
    int num_actions()  const { return cfg_.num_actions; }
    int trunk_dim()    const { return cfg_.hidden.empty() ? cfg_.in_dim : cfg_.hidden.back(); }
    int num_params()   const;

    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

private:
    Config cfg_{};

    // Trunk: alternating Linear + Relu, last activation included.
    std::vector<Linear> trunk_;
    std::vector<Relu>   trunk_acts_;

    // Value head.
    Linear v_fc1_;
    Relu   v_act_;
    Linear v_fc2_;          // out = 1
    Tanh   v_tanh_;

    // Policy head.
    Linear p_fc_;           // out = num_actions

    // Activation caches sized once at init() so forward/backward reuse them.
    std::vector<Tensor> trunk_raw_;     // pre-activation per layer
    std::vector<Tensor> trunk_act_;     // post-activation per layer
    Tensor v_h_raw_, v_h_act_;          // value-head hidden pre/post ReLU
    Tensor v_pre_tanh_, v_post_tanh_;   // size 1
};

} // namespace brogameagent::nn
