#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── FeedForward ───────────────────────────────────────────────────────────
//
// Position-wise 2-layer MLP: x → Linear(D, d_ff) → ReLU → Linear(d_ff, D).
// Operates on (K, D) input matrices: same parameters applied independently
// to each of the K rows. Parameters:
//   W1 (d_ff, D), b1 (d_ff)
//   W2 (D, d_ff), b2 (D)
// Activation is ReLU (deliberately simple; GELU can come later without
// touching call sites).
//
// Device: brotensor::Tensor carries its own Device tag and the brotensor
// batched linear / relu ops dispatch on it at runtime, so there is a single
// forward/backward that runs on whatever device the parameters live on.
// `to(brotensor::Device)` migrates every owned tensor; `device()` reports
// where they currently are.

class FeedForward : public ICircuit {
public:
    FeedForward() = default;

    void init(int dim, int d_ff, uint64_t& rng_state);

    int dim()  const { return d_; }
    int d_ff() const { return df_; }

    // X: (K, D); Y: (K, D); resized if mis-shaped.
    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "FeedForward"; }
    int  num_params() const override {
        return W1_.size() + b1_.size() + W2_.size() + b2_.size();
    }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor& W1() { return W1_; } brotensor::Tensor& b1() { return b1_; }
    brotensor::Tensor& W2() { return W2_; } brotensor::Tensor& b2() { return b2_; }
    brotensor::Tensor& dW1() { return dW1_; } brotensor::Tensor& dB1() { return dB1_; }
    brotensor::Tensor& dW2() { return dW2_; } brotensor::Tensor& dB2() { return dB2_; }

private:
    int d_  = 0;
    int df_ = 0;

    brotensor::Tensor W1_, b1_, W2_, b2_;
    brotensor::Tensor dW1_, dB1_, dW2_, dB2_;
    brotensor::Tensor vW1_, vB1_, vW2_, vB2_;
    brotensor::Tensor mW1_, mB1_, mW2_, mB2_;
    brotensor::Tensor vAW1_, vAB1_, vAW2_, vAB2_;

    // Caches.
    brotensor::Tensor X_cache_;   // (K, D)
    brotensor::Tensor H_pre_;     // (K, d_ff) pre-activation
    brotensor::Tensor H_post_;    // (K, d_ff) post-ReLU

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
