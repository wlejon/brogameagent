#pragma once

#include "circuits.h"
#include <brotensor/tensor.h>

#include <cstdint>

namespace brogameagent::nn {

// ─── Embedding ─────────────────────────────────────────────────────────────
//
// (vocab, dim) weight table. Forward: copy row idx into out. Backward: sparse
// — only row idx of dW accumulates the upstream gradient. Xavier init.

class Embedding : public ICircuit {
public:
    Embedding() = default;
    Embedding(int vocab, int dim, uint64_t& rng_state) { init(vocab, dim, rng_state); }

    void init(int vocab, int dim, uint64_t& rng_state);

    void forward(int idx, brotensor::Tensor& out);
    void backward(int idx, const brotensor::Tensor& dY);

    int vocab() const { return W_.rows; }
    int dim()   const { return W_.cols; }

    const char* name() const override { return "Embedding"; }
    int  num_params() const override { return W_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor&       W()       { return W_; }
    const brotensor::Tensor& W() const { return W_; }
    brotensor::Tensor&       dW()       { return dW_; }

private:
    brotensor::Tensor W_;
    brotensor::Tensor dW_;
    brotensor::Tensor vW_;
    brotensor::Tensor mW_;
    brotensor::Tensor vAW_;
};

} // namespace brogameagent::nn
