#pragma once

#include "circuits.h"
#include "tensor.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── LayerNorm ─────────────────────────────────────────────────────────────
//
// Per-vector normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
// Learnable gamma, beta of size N. Single-sample only (vector in, vector out).

class LayerNorm : public ICircuit {
public:
    LayerNorm() = default;
    LayerNorm(int n, float eps = 1e-5f) { init(n, eps); }

    void init(int n, float eps = 1e-5f);

    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

    int dim() const { return gamma_.size(); }

    const char* name() const override { return "LayerNorm"; }
    int  num_params() const override { return gamma_.size() + beta_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor&       gamma()       { return gamma_; }
    const Tensor& gamma() const { return gamma_; }
    Tensor&       beta()        { return beta_; }
    const Tensor& beta()  const { return beta_; }
    Tensor&       dGamma()       { return dGamma_; }
    Tensor&       dBeta()        { return dBeta_; }

private:
    Tensor gamma_, beta_;
    Tensor dGamma_, dBeta_;
    Tensor vGamma_, vBeta_;
    float eps_ = 1e-5f;
    // Caches for backward.
    Tensor xhat_;     // normalized x
    float mean_ = 0.0f;
    float rstd_ = 0.0f;
};

} // namespace brogameagent::nn
