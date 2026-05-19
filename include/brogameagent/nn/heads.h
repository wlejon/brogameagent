#pragma once

#include "circuits.h"
#include <brotensor/device.h>
#include <brotensor/tensor.h>
#include "brogameagent/action_mask.h"
#include "brogameagent/observation.h"

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/tensor.h>
#endif

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

    void forward(const brotensor::Tensor& embed, float& value);
    void backward(float dValue, brotensor::Tensor& dEmbed);

#ifdef BROTENSOR_HAS_GPU
    // GPU code path. Internal Linears must be on GPU (call to(GPU)).
    //   embed: (embed_dim, 1). Post-tanh value is cached in value_gpu().
    void forward(const brotensor::GpuTensor& embed);
    // dValue must point to a (1,1) device tensor with d(loss)/d(value).
    // Caller writes via dValue_gpu().
    void backward(brotensor::GpuTensor& dEmbed);

    const brotensor::GpuTensor& value_gpu() const { return post_tanh_g_; }
    brotensor::GpuTensor&       dValue_gpu()      { return dValue_g_; }
#endif

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

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
    brotensor::Tensor h_raw_, h_act_;       // hidden before/after ReLU
    brotensor::Tensor out_raw_;              // pre-tanh scalar (1-vec)
    float  y_cache_ = 0.0f;       // post-tanh scalar

    brotensor::Device device_ = brotensor::Device::CPU;
#ifdef BROTENSOR_HAS_GPU
    // GPU forward caches (sized at to(GPU)).
    brotensor::GpuTensor h_raw_g_, h_act_g_;
    brotensor::GpuTensor pre_tanh_g_;        // (1,1) pre-tanh scalar
    brotensor::GpuTensor post_tanh_g_;       // (1,1) cached post-tanh, used in backward
    brotensor::GpuTensor dValue_g_;          // (1,1) caller-written grad slot
    brotensor::GpuTensor dPre_g_;            // (1,1) backward scratch
    brotensor::GpuTensor dHact_g_, dHraw_g_; // (hidden,1) backward scratch
#endif
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
    void forward(const brotensor::Tensor& embed, brotensor::Tensor& logits);
    // Backward: dLogits is size total_logits(), dEmbed is size embed_dim.
    void backward(const brotensor::Tensor& dLogits, brotensor::Tensor& dEmbed);

#ifdef BROTENSOR_HAS_GPU
    // GPU code path. Internal Linears must be on GPU.
    //   embed:  (embed_dim, 1)
    //   logits: (total_logits, 1) — concatenated [move | atk | abil].
    void forward(const brotensor::GpuTensor& embed, brotensor::GpuTensor& logits);
    void backward(const brotensor::GpuTensor& dLogits, brotensor::GpuTensor& dEmbed);
#endif

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

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
    brotensor::Device device_ = brotensor::Device::CPU;
#ifdef BROTENSOR_HAS_GPU
    // GPU per-segment buffers (allocated at to(GPU); reused).
    brotensor::GpuTensor lm_g_, la_g_, lb_g_;     // forward outputs per segment
    brotensor::GpuTensor dLm_g_, dLa_g_, dLb_g_;  // sliced gradients per segment
    brotensor::GpuTensor dEmbedTmp_g_;            // (embed_dim,1) scratch
#endif
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
void factored_softmax(const brotensor::Tensor& logits, brotensor::Tensor& probs,
                      const float* attack_mask = nullptr,
                      const float* ability_mask = nullptr);

// Combined backward: cross-entropy loss against (move_target, attack_target,
// ability_target) each expressed as a soft distribution (visit fractions from
// MCTS). Returns total loss summed across the three heads. dLogits has
// gradient contributions from all three.
float factored_xent(const brotensor::Tensor& logits,
                    const brotensor::Tensor& move_target,
                    const brotensor::Tensor& attack_target,
                    const brotensor::Tensor& ability_target,
                    brotensor::Tensor& probs, brotensor::Tensor& dLogits,
                    const float* attack_mask = nullptr,
                    const float* ability_mask = nullptr);

} // namespace brogameagent::nn
