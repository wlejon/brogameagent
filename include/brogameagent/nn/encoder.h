#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>
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
//
// Device: device_ tracks where parameters live. `to(brotensor::Device)`
// migrates every owned tensor (and sub-Linear); brotensor ops dispatch on
// operand device at runtime, so a single forward/backward path suffices.

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
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "DeepSetsEncoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    const Config& config() const { return cfg_; }

private:
    Config cfg_{};

    // Self stream: SELF_FEATURES -> hidden -> embed
    Linear self_fc1_, self_fc2_;
    Relu   self_act_;
    brotensor::Tensor self_h_, self_z_;     // hidden after fc1+relu, embed after fc2
    brotensor::Tensor self_in_;             // sliced self input (device-resident)
    brotensor::Tensor self_h_pre_;          // pre-relu hidden cache for backward

    // Enemy stream: ENEMY_FEATURES -> hidden -> embed (applied per slot)
    Linear enemy_fc1_, enemy_fc2_;
    // Ally stream
    Linear ally_fc1_, ally_fc2_;

    // Batched per-slot caches (device-resident). Shapes:
    //   *_in_   (K, FEATURES)   sliced slot inputs
    //   *_hpre_ (K, hidden)     pre-relu hidden
    //   *_h_    (K, hidden)     post-relu hidden
    //   *_z_    (K, embed)      per-slot embeddings
    //   *_mask_ (K, 1)          slot-validity mask
    brotensor::Tensor e_in_, e_hpre_, e_h_, e_z_, e_mask_;
    brotensor::Tensor a_in_, a_hpre_, a_h_, a_z_, a_mask_;
    int e_n_valid_ = 0, a_n_valid_ = 0;

    brotensor::Tensor x_cache_;  // copy of input for backward shape

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
