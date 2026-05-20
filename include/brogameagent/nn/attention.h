#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── ScaledDotProductAttention ─────────────────────────────────────────────
//
// Single-head self-attention over N slots of dim D. Four learnable (D, D)
// projections Wq/Wk/Wv/Wo; no biases (simpler, works fine as a layer norm
// follows in typical stacks). Invalid rows (mask[k] == 0) are excluded from
// the softmax denominator (additive -inf on scores pre-softmax) and produce
// zero output rows — this matches the convention of softmax_forward's mask.
//
// Device: brotensor::Tensor carries its own Device tag and the brotensor
// attention ops dispatch on it at runtime, so there is a single
// forward/backward that runs on whatever device the parameters live on.
// `to(brotensor::Device)` migrates every owned tensor; `device()` reports
// where they currently are.

class ScaledDotProductAttention : public ICircuit {
public:
    ScaledDotProductAttention() = default;

    void init(int n_slots, int dim, uint64_t& rng_state, int num_heads = 1);

    int n_slots() const { return n_; }
    int dim()     const { return d_; }

    // X: (N, D). mask: length N (1 valid, 0 invalid), may be nullptr.
    // O: (N, D).
    void forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& O);
    void backward(const brotensor::Tensor& dO, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "ScaledDotProductAttention"; }
    int  num_params() const override { return Wq_.size() + Wk_.size() + Wv_.size() + Wo_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor&       Wq()       { return Wq_; }
    brotensor::Tensor&       Wk()       { return Wk_; }
    brotensor::Tensor&       Wv()       { return Wv_; }
    brotensor::Tensor&       Wo()       { return Wo_; }
    brotensor::Tensor&       dWq()      { return dWq_; }
    brotensor::Tensor&       dWk()      { return dWk_; }
    brotensor::Tensor&       dWv()      { return dWv_; }
    brotensor::Tensor&       dWo()      { return dWo_; }

private:
    int n_ = 0, d_ = 0;
    brotensor::Tensor Wq_, Wk_, Wv_, Wo_;
    brotensor::Tensor dWq_, dWk_, dWv_, dWo_;
    brotensor::Tensor vWq_, vWk_, vWv_, vWo_;
    // Adam moment buffers.
    brotensor::Tensor mWq_, mWk_, mWv_, mWo_;
    brotensor::Tensor vAWq_, vAWk_, vAWv_, vAWo_;

    // Caches for backward.
    brotensor::Tensor X_cache_;       // (N, D)
    brotensor::Tensor Q_, K_, V_;     // (N, D)
    brotensor::Tensor Attn_;          // (N, N)
    brotensor::Tensor Y_;             // (N, D) = Attn @ V (pre-Wo)
    // Validity-mask pointer used by the most recent forward; reused verbatim
    // by backward. The caller owns the buffer and must keep it alive (and on
    // the same device as X) between the forward and the matching backward.
    const float* mask_ptr_ = nullptr;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
