#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>
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
//
// Device: device_ field tracks where parameters live; to(brotensor::Device)
// migrates every owned tensor (and sub-Linear). brotensor ops dispatch on
// operand device at runtime, so a single forward/backward path suffices.

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
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "DeepSetsDecoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    const Config& config() const { return cfg_; }

private:
    Config cfg_{};

    // Self stream.
    Linear self_fc1_, self_fc2_;
    brotensor::Tensor self_h_raw_, self_h_;   // post-fc1, post-relu
    brotensor::Tensor self_out_;              // reconstructed self block

    // Enemy stream (shared MLP applied per slot; batched (K, *) caches).
    Linear enemy_fc1_, enemy_fc2_;
    brotensor::Tensor e_in_;       // (K_ENEMIES, embed_dim) broadcast input
    brotensor::Tensor e_h_raw_;    // (K_ENEMIES, hidden) pre-relu
    brotensor::Tensor e_h_;        // (K_ENEMIES, hidden) post-relu
    brotensor::Tensor e_out_;      // (K_ENEMIES, ENEMY_FEATURES)

    // Ally stream.
    Linear ally_fc1_, ally_fc2_;
    brotensor::Tensor a_in_;       // (K_ALLIES, embed_dim)
    brotensor::Tensor a_h_raw_;    // (K_ALLIES, hidden)
    brotensor::Tensor a_h_;        // (K_ALLIES, hidden)
    brotensor::Tensor a_out_;      // (K_ALLIES, ALLY_FEATURES)

    // Cached split of input for backward.
    brotensor::Tensor self_in_, pooled_e_, pooled_a_;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
