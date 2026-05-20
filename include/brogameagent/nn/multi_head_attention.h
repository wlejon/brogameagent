#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── MultiHeadAttention ────────────────────────────────────────────────────
//
// Multi-head self-attention over K tokens with d_model = D. The four
// projections Wq/Wk/Wv/Wo are each (D, D) — internally split into h
// (head_dim, D) per-head sub-projections, with head_dim = D / h. No biases
// (matches ScaledDotProductAttention convention; a downstream LayerNorm
// usually follows).
//
// Mask semantics mirror ScaledDotProductAttention:
//   - mask is length K, 1 valid / 0 invalid (float*).
//   - Invalid keys are excluded from the softmax denominator.
//   - Invalid query rows produce zero output rows.
//
// Backward accumulates into dWq/dWk/dWv/dWo (does NOT overwrite) — caller
// is responsible for zero_grad between batches.
//
// Device: brotensor::Tensor carries its own Device tag and the brotensor mha
// ops dispatch on it at runtime, so there is a single forward/backward that
// runs on whatever device the parameters live on. `to(brotensor::Device)`
// migrates every owned tensor; `device()` reports where they currently are.

class MultiHeadAttention : public ICircuit {
public:
    MultiHeadAttention() = default;

    // num_heads must divide dim. n_slots = K (number of tokens).
    void init(int n_slots, int dim, int num_heads, uint64_t& rng_state);

    int n_slots()   const { return n_; }
    int dim()       const { return d_; }
    int num_heads() const { return h_; }
    int head_dim()  const { return dh_; }

    // X: (K, D). mask: length K (1 valid, 0 invalid), may be nullptr.
    // O: (K, D); resized if mis-shaped.
    void forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& O);
    void backward(const brotensor::Tensor& dO, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "MultiHeadAttention"; }
    int  num_params() const override {
        return Wq_.size() + Wk_.size() + Wv_.size() + Wo_.size();
    }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor&       Wq()  { return Wq_; }
    brotensor::Tensor&       Wk()  { return Wk_; }
    brotensor::Tensor&       Wv()  { return Wv_; }
    brotensor::Tensor&       Wo()  { return Wo_; }
    brotensor::Tensor&       dWq() { return dWq_; }
    brotensor::Tensor&       dWk() { return dWk_; }
    brotensor::Tensor&       dWv() { return dWv_; }
    brotensor::Tensor&       dWo() { return dWo_; }

private:
    int n_  = 0;
    int d_  = 0;
    int h_  = 1;
    int dh_ = 0;

    brotensor::Tensor Wq_, Wk_, Wv_, Wo_;
    brotensor::Tensor dWq_, dWk_, dWv_, dWo_;
    brotensor::Tensor vWq_, vWk_, vWv_, vWo_;
    brotensor::Tensor mWq_, mWk_, mWv_, mWo_;
    brotensor::Tensor vAWq_, vAWk_, vAWv_, vAWo_;

    // Caches for backward. The brotensor mha ops use a flat per-head layout:
    //   Qh_/Kh_/Vh_: (h * K, head_dim) — head h occupies rows [h*K, (h+1)*K)
    //   Attnh_:      (h * K, K)        — per-head softmax weights
    //   Yconcat_:    (K, D)            — pre-Wo concat of per-head outputs
    brotensor::Tensor X_cache_;   // (K, D)
    brotensor::Tensor Qh_, Kh_, Vh_;
    brotensor::Tensor Attnh_;
    brotensor::Tensor Yconcat_;
    // Validity-mask pointer used by the most recent forward; reused verbatim
    // by backward. The caller owns the mask buffer and must keep it alive (and
    // on the same device as X) between the forward and the matching backward.
    // No host copy is made — a host snapshot would be wrong for device masks.
    const float* mask_ptr_ = nullptr;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
