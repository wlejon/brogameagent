#pragma once

#include "circuits.h"
#include "tensor.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── DistributionalValueHead (C51-style) ──────────────────────────────────
//
// embed -> Linear(hidden) -> ReLU -> Linear(K) -> softmax over K bins
// covering [-1, +1] with support v_i = -1 + 2*i/(K-1). Forward writes both
// the probability vector (size K) and the scalar expected value. Backward
// takes a target distribution p_target (size K, sums to 1) and uses
// dLogits = probs - p_target.

class DistributionalValueHead : public ICircuit {
public:
    DistributionalValueHead() = default;

    void init(int embed_dim, int hidden, int K, uint64_t& rng_state);

    int num_bins() const { return K_; }
    float support(int i) const { return -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(K_ - 1); }

    // Forward: writes probs (size K) and expected scalar value.
    void forward(const Tensor& embed, Tensor& probs, float& value);

    // Compute cross-entropy loss vs p_target, populate dLogits = probs - p_target,
    // run backward through the two linears producing dEmbed.
    float xent_backward(const Tensor& probs, const Tensor& p_target, Tensor& dEmbed);

    // Two-hot projection of scalar z in [-1,1] onto the K-bin support.
    void project_target(float z, Tensor& out) const;

    const char* name() const override { return "DistributionalValueHead"; }
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
    int K_ = 0;
    Linear fc1_, fc2_;
    Tensor h_raw_, h_act_;
    Tensor logits_;
};

} // namespace brogameagent::nn
