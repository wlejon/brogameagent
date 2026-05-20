#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>

#include <cstdint>

namespace brogameagent::nn {

// ─── LayerNorm ─────────────────────────────────────────────────────────────
//
// Per-vector normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
// Learnable gamma, beta of size N. Single-sample only (vector in, vector out).
//
// forward/backward call the device-dispatched brotensor layernorm ops, so the
// layer runs on whatever device its parameters live on. `to(Device)` migrates
// every owned tensor.

class LayerNorm : public ICircuit {
public:
    LayerNorm() = default;
    LayerNorm(int n, float eps = 1e-5f) { init(n, eps); }

    void init(int n, float eps = 1e-5f);

    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    int dim() const { return gamma_.size(); }

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "LayerNorm"; }
    int  num_params() const override { return gamma_.size() + beta_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor&       gamma()       { return gamma_; }
    const brotensor::Tensor& gamma() const { return gamma_; }
    brotensor::Tensor&       beta()        { return beta_; }
    const brotensor::Tensor& beta()  const { return beta_; }
    brotensor::Tensor&       dGamma()       { return dGamma_; }
    brotensor::Tensor&       dBeta()        { return dBeta_; }

private:
    brotensor::Tensor gamma_, beta_;
    brotensor::Tensor dGamma_, dBeta_;
    brotensor::Tensor vGamma_, vBeta_;
    // Adam moment buffers.
    brotensor::Tensor mGamma_, mBeta_;
    brotensor::Tensor vAGamma_, vABeta_;
    float eps_ = 1e-5f;
    // Caches for backward.
    brotensor::Tensor xhat_;     // normalized x
    float mean_ = 0.0f;
    float rstd_ = 0.0f;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
