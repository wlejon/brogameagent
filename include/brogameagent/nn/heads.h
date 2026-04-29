#pragma once

#include "circuits.h"
#include "tensor.h"
#include "brogameagent/action_mask.h"
#include "brogameagent/observation.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── ValueHead ─────────────────────────────────────────────────────────────
//
// embed -> hidden -> 1 -> tanh. Output scalar in [-1, 1]. Trained with MSE
// against discounted return (also clipped to [-1, 1]).

class ValueHead : public ICircuit {
public:
    ValueHead() = default;

    void init(int embed_dim, int hidden, uint64_t& rng_state);

    void forward(const Tensor& embed, float& value);
    void backward(float dValue, Tensor& dEmbed);

    const char* name() const override { return "ValueHead"; }
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
    Linear fc1_, fc2_;
    Tensor h_raw_, h_act_;       // hidden before/after ReLU
    Tensor out_raw_;              // pre-tanh scalar (1-vec)
    float  y_cache_ = 0.0f;       // post-tanh scalar
};

// ─── FactoredPolicyHead ────────────────────────────────────────────────────
//
// Three independent soft-max heads over:
//   MoveDir:     9 classes (Hold, N, NE, E, SE, S, SW, W, NW) — always legal.
//   AttackSlot:  action_mask::N_ENEMY_SLOTS + 1 classes — last class = "no attack".
//   AbilitySlot: action_mask::N_ABILITY_SLOTS + 1 classes — last class = "no cast".
//
// Pathfinding-aware move kinds (PathToTarget / PathAway, kinds 9/10) are not
// represented in the policy head for v1; they can be recovered via the
// opponent/rollout policy. Keeping the move head at 9 matches observation
// and keeps the output small.
//
// The attack and ability heads receive the legal-action mask from
// action_mask::build, with the "no-op" trailing class always legal. Masked
// softmax + cross-entropy handles the rest.

class FactoredPolicyHead : public ICircuit {
public:
    static constexpr int N_MOVE    = 9;
    static constexpr int N_ATTACK  = action_mask::N_ENEMY_SLOTS + 1;
    static constexpr int N_ABILITY = action_mask::N_ABILITY_SLOTS + 1;

    void init(int embed_dim, uint64_t& rng_state);

    // logits_out has 3 pieces, concatenated: [move(9), attack(N_ATTACK), ability(N_ABILITY)].
    int total_logits() const { return N_MOVE + N_ATTACK + N_ABILITY; }

    // Forward: produces logits. Softmax happens inside the xent loss or at
    // inference time via apply_softmax() helper.
    void forward(const Tensor& embed, Tensor& logits);
    // Backward: dLogits is size total_logits(), dEmbed is size embed_dim.
    void backward(const Tensor& dLogits, Tensor& dEmbed);

    const char* name() const override { return "FactoredPolicyHead"; }
    int  num_params() const override { return move_.num_params() + atk_.num_params() + abil_.num_params(); }
    void zero_grad() override { move_.zero_grad(); atk_.zero_grad(); abil_.zero_grad(); }
    void sgd_step(float lr, float m) override {
        move_.sgd_step(lr, m); atk_.sgd_step(lr, m); abil_.sgd_step(lr, m);
    }
    void adam_step(float lr, float b1, float b2, float eps, int step) {
        move_.adam_step(lr, b1, b2, eps, step);
        atk_.adam_step(lr, b1, b2, eps, step);
        abil_.adam_step(lr, b1, b2, eps, step);
    }
    void save_to(std::vector<uint8_t>& out) const override {
        move_.save_to(out); atk_.save_to(out); abil_.save_to(out);
    }
    void load_from(const uint8_t* d, size_t& o, size_t s) override {
        move_.load_from(d, o, s); atk_.load_from(d, o, s); abil_.load_from(d, o, s);
    }

private:
    Linear move_, atk_, abil_;
};

// ─── OpponentPolicyHead ───────────────────────────────────────────────────
//
// Auxiliary head predicting the observed opponent's action. Structurally
// identical to FactoredPolicyHead (same 9 / N_ATTACK / N_ABILITY factoring),
// reused via alias. Trained with factored_xent against the opponent target.
using OpponentPolicyHead = FactoredPolicyHead;

// Helper: in-place softmax on each of the three factored regions of a logits
// vector. Optional attack_mask / ability_mask: size N_ATTACK-1 and
// N_ABILITY-1 respectively, reflecting action_mask::build output. The
// trailing "no-op" class is always legal.
void factored_softmax(const Tensor& logits, Tensor& probs,
                      const float* attack_mask = nullptr,
                      const float* ability_mask = nullptr);

// Combined backward: cross-entropy loss against (move_target, attack_target,
// ability_target) each expressed as a soft distribution (visit fractions from
// MCTS). Returns total loss summed across the three heads. dLogits has
// gradient contributions from all three.
float factored_xent(const Tensor& logits,
                    const Tensor& move_target,
                    const Tensor& attack_target,
                    const Tensor& ability_target,
                    Tensor& probs, Tensor& dLogits,
                    const float* attack_mask = nullptr,
                    const float* ability_mask = nullptr);

} // namespace brogameagent::nn
