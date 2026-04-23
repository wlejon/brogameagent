#pragma once

#include "circuits.h"
#include "tensor.h"
#include "brogameagent/observation.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── DeepSetsEncoder ───────────────────────────────────────────────────────
//
// Egocentric set encoder tailored to observation::build.
//
// Input: a TOTAL-dim vector in the layout that observation::build emits —
//   self[SELF_FEATURES], enemies[K_ENEMIES*ENEMY_FEATURES], allies[K_ALLIES*ALLY_FEATURES].
//
// Processing:
//   - Self block goes through its own small MLP: phi_self.
//   - Each enemy slot (K_ENEMIES of them) goes through a shared phi_enemy
//     MLP, output mean-pooled across valid slots. A slot is "valid" iff its
//     first feature (the valid flag) is > 0.5; invalid slots are masked out
//     of the pool (do not contribute, divide by count of valid only).
//   - Same for allies via phi_ally.
//   - Outputs concatenated into a single embedding vector of dim out_dim().
//
// Permutation invariance is structural: the pool is order-invariant and the
// slots are already sorted nearest-first by observation::build, so slot
// identity is stable. Backward is hand-authored to match — each slot's dE
// is distributed from the pool, scaled by 1/n_valid, and invalid slots
// receive zero gradient.

class DeepSetsEncoder : public ICircuit {
public:
    DeepSetsEncoder() = default;

    struct Config {
        int hidden    = 32;   // per-stream hidden width (self/enemy/ally MLPs)
        int embed_dim = 32;   // per-stream output embedding width
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int out_dim() const { return 3 * cfg_.embed_dim; }

    // x: size observation::TOTAL. y: size out_dim().
    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

    const char* name() const override { return "DeepSetsEncoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

private:
    Config cfg_{};

    // Self stream: SELF_FEATURES -> hidden -> embed
    Linear self_fc1_, self_fc2_;
    Relu   self_act_;
    Tensor self_h_, self_z_;     // hidden after fc1+relu, embed after fc2

    // Enemy stream: ENEMY_FEATURES -> hidden -> embed (applied per slot)
    Linear enemy_fc1_, enemy_fc2_;
    // Ally stream
    Linear ally_fc1_, ally_fc2_;

    // Per-slot caches. Sized by init().
    std::vector<Tensor> e_h_, e_z_;
    std::vector<Tensor> a_h_, a_z_;
    std::vector<uint8_t> e_valid_, a_valid_;
    int e_n_valid_ = 0, a_n_valid_ = 0;

    Tensor x_cache_;  // copy of input for backward shape

    // scratch for per-slot backward (input grad for an entire slot row)
    Tensor slot_grad_in_;
};

} // namespace brogameagent::nn
