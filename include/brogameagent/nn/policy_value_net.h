#pragma once

#include "circuits.h"
#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── PolicyValueNet ───────────────────────────────────────────────────────
//
// A small, hand-coded MLP with a value head and one or more policy heads,
// decoupled from the MOBA-shaped observation/action layout that SingleHeroNet
// assumes. This is the "bring your own observation, bring your own action
// space" net: useful for any small-state-space discrete-action problem
// (platformers, puzzle games, gridworlds, etc.).
//
// Architecture:
//   trunk   : in_dim → hidden[0] → ReLU → hidden[1] → ReLU → ... → hidden[n-1]
//   value   : hidden[n-1] → value_hidden → ReLU → 1 → tanh        (in [-1, 1])
//   policy  : hidden[n-1] → sum(head_sizes)                       (raw logits)
//
// Forward returns (value, logits). The policy output is the concatenation of
// per-head logit segments. With a single head (the default), this is just the
// flat logit vector callers had before — same shape, same gradient. With
// multiple heads, callers (the trainer, MCTS prior helpers) split `logits`
// at head_offsets() and apply a softmax per segment. The net itself does not
// know about heads beyond their total width — backward sums whatever
// gradient signal is concatenated in dLogits.
//
// Backward expects (dValue, dLogits) where dLogits is gradient on the raw
// logits — typically (probs - target) from a masked softmax-xent applied per
// head segment (use nn::softmax_xent_raw with each head's offset, or the
// trainer's per-head loop).
//
// Wire format: distinct magic from SingleHeroNet so we can't mix them up.

class PolicyValueNet {
public:
    struct Config {
        int in_dim = 0;                          // observation length
        std::vector<int> hidden = {64, 64};      // trunk hidden widths
        int value_hidden = 32;

        // Policy head shape. Two ways to configure:
        //   single-head: set num_actions, leave head_sizes empty.
        //                head_sizes is treated as {num_actions} internally.
        //   factored:    set head_sizes = {h0, h1, ...} (each > 0).
        //                num_actions is auto-set to sum(head_sizes) if 0,
        //                otherwise must equal sum(head_sizes).
        // The policy output width is always sum of head sizes.
        int num_actions = 0;
        std::vector<int> head_sizes;             // empty == single flat head

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
    void adam_step(float lr, float beta1, float beta2, float eps, int step);

    int in_dim()       const { return cfg_.in_dim; }
    int num_actions()  const { return cfg_.num_actions; }
    int trunk_dim()    const { return cfg_.hidden.empty() ? cfg_.in_dim : cfg_.hidden.back(); }
    int num_params()   const;

    // Per-head accessors. head_sizes() always has at least one entry; for
    // single-head nets it's {num_actions}. head_offsets() has one extra
    // sentinel entry at the end equal to num_actions, so the i'th head
    // spans [offsets[i], offsets[i+1]).
    const std::vector<int>& head_sizes()   const { return head_sizes_; }
    const std::vector<int>& head_offsets() const { return head_offsets_; }
    int num_heads() const { return static_cast<int>(head_sizes_.size()); }

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

    // Resolved head shape. Always populated by init():
    //   head_sizes_   = cfg_.head_sizes if non-empty, else {cfg_.num_actions}.
    //   head_offsets_ = exclusive prefix sums of head_sizes_, with a trailing
    //                   entry equal to sum (== num_actions). One past the last
    //                   head, used as a half-open end.
    std::vector<int> head_sizes_;
    std::vector<int> head_offsets_;
};

} // namespace brogameagent::nn
