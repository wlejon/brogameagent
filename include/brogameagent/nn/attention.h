#pragma once

#include "circuits.h"
#include "tensor.h"

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

class ScaledDotProductAttention : public ICircuit {
public:
    ScaledDotProductAttention() = default;

    void init(int n_slots, int dim, uint64_t& rng_state, int num_heads = 1);

    int n_slots() const { return n_; }
    int dim()     const { return d_; }

    // X: (N, D). mask: length N (1 valid, 0 invalid), may be nullptr (all valid).
    // O: (N, D).
    void forward(const Tensor& X, const float* mask, Tensor& O);
    void backward(const Tensor& dO, Tensor& dX);

    const char* name() const override { return "ScaledDotProductAttention"; }
    int  num_params() const override { return Wq_.size() + Wk_.size() + Wv_.size() + Wo_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor&       Wq()       { return Wq_; }
    Tensor&       Wk()       { return Wk_; }
    Tensor&       Wv()       { return Wv_; }
    Tensor&       Wo()       { return Wo_; }
    Tensor&       dWq()      { return dWq_; }
    Tensor&       dWk()      { return dWk_; }
    Tensor&       dWv()      { return dWv_; }
    Tensor&       dWo()      { return dWo_; }

private:
    int n_ = 0, d_ = 0;
    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor dWq_, dWk_, dWv_, dWo_;
    Tensor vWq_, vWk_, vWv_, vWo_;

    // Caches for backward.
    Tensor X_cache_;       // (N, D)
    Tensor Q_, K_, V_;     // (N, D)
    Tensor Attn_;          // (N, N)
    Tensor Y_;             // (N, D) = Attn @ V
    std::vector<uint8_t> mask_cache_;
};

} // namespace brogameagent::nn
