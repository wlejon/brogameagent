#pragma once

#include "circuits.h"
#include "heads.h"
#include "tensor.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── ForwardModelHead (SPR-lite) ──────────────────────────────────────────
//
// Given current embedding and a concatenated action one-hot
// (move(9) || attack(N_ATTACK) || ability(N_ABILITY)), predicts the next
// step's embedding (same dim as input embed). Loss is MSE against the
// encoded next state. Backward splits dInput into dEmbed (propagated) and
// dAction (discarded — actions are one-hot, no trainable upstream).

class ForwardModelHead : public ICircuit {
public:
    static constexpr int N_MOVE    = FactoredPolicyHead::N_MOVE;
    static constexpr int N_ATTACK  = FactoredPolicyHead::N_ATTACK;
    static constexpr int N_ABILITY = FactoredPolicyHead::N_ABILITY;
    static constexpr int ACTION_DIM = N_MOVE + N_ATTACK + N_ABILITY;

    ForwardModelHead() = default;

    void init(int embed_dim, int hidden, uint64_t& rng_state);

    int embed_dim() const { return embed_dim_; }

    // embed: size embed_dim. action: size ACTION_DIM (three one-hots
    // concatenated). pred_next: size embed_dim.
    void forward(const Tensor& embed, const Tensor& action, Tensor& pred_next);
    // Backward through the MLP. dPred is the gradient on the predicted
    // embedding (use spr_loss to build it from MSE); dEmbed receives the
    // input gradient, action gradient is discarded.
    void backward(const Tensor& dPred, Tensor& dEmbed);

    const char* name() const override { return "ForwardModelHead"; }
    int  num_params() const override { return fc1_.num_params() + fc2_.num_params(); }
    void zero_grad() override { fc1_.zero_grad(); fc2_.zero_grad(); }
    void sgd_step(float lr, float m) override { fc1_.sgd_step(lr, m); fc2_.sgd_step(lr, m); }
    void adam_step(float lr, float b1, float b2, float eps, int step) {
        fc1_.adam_step(lr, b1, b2, eps, step); fc2_.adam_step(lr, b1, b2, eps, step);
    }
    void save_to(std::vector<uint8_t>& out) const override { fc1_.save_to(out); fc2_.save_to(out); }
    void load_from(const uint8_t* d, size_t& o, size_t s) override {
        fc1_.load_from(d, o, s); fc2_.load_from(d, o, s);
    }

private:
    int embed_dim_ = 0;
    Linear fc1_, fc2_;
    Tensor input_cat_;   // [embed || action]
    Tensor h_raw_, h_act_;
};

// Build action one-hot into `out` (size ACTION_DIM) from three class indices.
void build_action_onehot(int move_idx, int attack_idx, int ability_idx, Tensor& out);

// 0.5 * ||pred - target||^2 (summed), dPred = pred - target. Returns loss.
float spr_loss(const Tensor& pred, const Tensor& target, Tensor& dPred);

} // namespace brogameagent::nn
