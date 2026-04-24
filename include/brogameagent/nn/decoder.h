#pragma once

#include "circuits.h"
#include "tensor.h"
#include "brogameagent/observation.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── DeepSetsDecoder ───────────────────────────────────────────────────────
//
// Mirror of DeepSetsEncoder for unsupervised autoencoding pretraining.
//
// Input: embedding of size 3 * embed_dim — [self_z, pooled_e, pooled_a].
// Output: reconstructed observation vector of size observation::TOTAL.
//
// Architecture:
//   - Self stream:  embed_dim -> hidden -> SELF_FEATURES  via Linear+ReLU+Linear.
//   - Enemy stream: the pooled enemy embedding is broadcast to each of the
//       K_ENEMIES slots (same pooled vector per slot), and each slot goes
//       through a shared embed_dim -> hidden -> ENEMY_FEATURES MLP. Because
//       we only have a single pooled vector to reconstruct K_ENEMIES slots
//       from, every slot's reconstruction is identical. This is an expected
//       property of pooling-based autoencoders — the pool is a lossy summary
//       and cannot recover per-slot identity. The reconstruction still
//       carries useful signal (mean per-feature values, valid-flag rate,
//       mean distance/hp, etc.) which is enough for pretraining.
//   - Ally stream: same pattern for K_ALLIES with ALLY_FEATURES.
//
// No bottleneck beyond what the encoder already produces. Hand-authored
// forward/backward — see decoder.cpp.

class DeepSetsDecoder : public ICircuit {
public:
    DeepSetsDecoder() = default;

    struct Config {
        int embed_dim = 32;   // per-stream input embed width (matches encoder)
        int hidden    = 32;   // per-stream hidden width
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int in_dim() const { return 3 * cfg_.embed_dim; }
    int out_dim() const { return observation::TOTAL; }

    // x: size 3*embed_dim. y: size observation::TOTAL.
    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

    const char* name() const override { return "DeepSetsDecoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

private:
    Config cfg_{};

    // Self stream.
    Linear self_fc1_, self_fc2_;
    Tensor self_h_raw_, self_h_;   // post-fc1, post-relu

    // Enemy stream (shared across slots; per-slot caches).
    Linear enemy_fc1_, enemy_fc2_;
    std::vector<Tensor> e_h_raw_, e_h_;  // per-slot hidden caches

    // Ally stream.
    Linear ally_fc1_, ally_fc2_;
    std::vector<Tensor> a_h_raw_, a_h_;

    // Cached split of input for backward.
    Tensor self_in_, pooled_e_, pooled_a_;
};

} // namespace brogameagent::nn
