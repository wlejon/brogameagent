#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>
#include "transformer_block.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace brogameagent::nn {

// ─── TransformerEncoder ────────────────────────────────────────────────────
//
// Stack of N TransformerBlocks sharing d_model, num_heads, d_ff, mask
// semantics, and norm placement. For pre-norm stacks a final LayerNorm is
// applied unconditionally (standard practice — keeps output activation
// scale stable). For post-norm the final LayerNorm is omitted (the last
// block already produces a normalized output).
//
// Serialization: writes block count (int32) + each block's serialization,
// then the optional final-LN serialization (only for pre-norm).
//
// Device: composite layer. `to(brotensor::Device)` recurses to all blocks
// and the final RowLN; brotensor ops dispatch on operand device at runtime.

class TransformerEncoder : public ICircuit {
public:
    TransformerEncoder() = default;

    struct Config {
        int n_layers  = 2;
        int dim       = 32;
        int num_heads = 4;
        int d_ff      = 64;
        int n_slots   = 0;
        float ln_eps  = 1e-5f;
        NormPlacement norm = NormPlacement::PreNorm;
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int dim()       const { return cfg_.dim; }
    int num_heads() const { return cfg_.num_heads; }
    int n_layers()  const { return cfg_.n_layers; }

    // X: (K, D); Y: (K, D); resized if mis-shaped.
    void forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& Y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "TransformerEncoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    TransformerBlock& block(int i) { return *blocks_[i]; }

private:
    Config cfg_{};
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    // Activations between blocks; activations_[0] = X copy,
    // activations_[i+1] = output of block i. Used for backward chaining.
    std::vector<brotensor::Tensor> activations_;

    // Optional final LN (pre-norm only).
    TransformerBlock::RowLN final_ln_;
    bool has_final_ln_ = false;
    // Cached pre-final-LN value for backward when has_final_ln_ is true.
    brotensor::Tensor pre_final_ln_;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
