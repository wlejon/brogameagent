#pragma once

#include "decoder.h"
#include "encoder.h"
#include "net.h"
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── DeepSetsAutoencoder ───────────────────────────────────────────────────
//
// Unsupervised pretraining composite: DeepSetsEncoder -> DeepSetsDecoder.
// Observation in, observation reconstruction out. Loss is masked MSE
// (see reconstruction_loss in autoencoder.cpp).
//
// Device: to(brotensor::Device) recurses into encoder/decoder and migrates the
// `embed`/`dEmbed` caches. forward/backward run on whatever device the
// tensors live on — brotensor ops dispatch at runtime.

class DeepSetsAutoencoder {
public:
    struct Config {
        DeepSetsEncoder::Config enc{};
        int dec_hidden = 32;
        uint64_t seed  = 0xA11CEBEEFULL;
    };

    DeepSetsAutoencoder() = default;
    void init(const Config& cfg);
    const Config& config() const { return cfg_; }

    // Single-sample forward/backward.
    void forward(const brotensor::Tensor& x, brotensor::Tensor& x_hat);
    void backward(const brotensor::Tensor& dX_hat);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    void zero_grad();
    void sgd_step(float lr, float momentum);
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    int num_params() const;

    // Accessors.
    DeepSetsEncoder&       encoder()       { return enc_; }
    const DeepSetsEncoder& encoder() const { return enc_; }
    DeepSetsDecoder&       decoder()       { return dec_; }
    const DeepSetsDecoder& decoder() const { return dec_; }

    // Full-AE serialization (binary, magic-tagged).
    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

    // Just the encoder sub-blob — binary-compatible with
    // DeepSetsEncoder::save_to / load_from (no header).
    std::vector<uint8_t> save_encoder() const;

private:
    Config cfg_{};
    DeepSetsEncoder enc_;
    DeepSetsDecoder dec_;
    brotensor::Tensor embed_;       // cached forward output of encoder
    brotensor::Tensor dEmbed_;      // scratch for backward

    brotensor::Device device_ = brotensor::Device::CPU;
};

// Compute masked reconstruction loss and gradient wrt the reconstruction.
// Semantics:
//  - Self block: plain MSE over all SELF_FEATURES.
//  - Each enemy/ally slot: if the slot's valid flag (feature 0) in `x` is
//    <=0.5, the slot is padding — its loss and gradient are zero across all
//    ENEMY_FEATURES/ALLY_FEATURES. Otherwise MSE over all features (including
//    the valid flag, so the decoder learns to emit valid=1 for populated
//    slots).
// Returns mean per-element loss over the count of features that actually
// contribute (self features + sum of valid slot features).
float reconstruction_loss(const brotensor::Tensor& x, const brotensor::Tensor& x_hat, brotensor::Tensor& dX_hat);

// Copy encoder weights from an autoencoder into a SingleHeroNet by
// round-tripping through the encoder's save/load byte layout.
void copy_encoder_weights(const DeepSetsAutoencoder& src, SingleHeroNet& dst);

} // namespace brogameagent::nn
