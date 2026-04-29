#pragma once

#include "circuits.h"
#include "tensor.h"

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

    void forward(int idx, Tensor& out);
    void backward(int idx, const Tensor& dY);

    int vocab() const { return W_.rows; }
    int dim()   const { return W_.cols; }

    const char* name() const override { return "Embedding"; }
    int  num_params() const override { return W_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor&       W()       { return W_; }
    const Tensor& W() const { return W_; }
    Tensor&       dW()       { return dW_; }

private:
    Tensor W_;
    Tensor dW_;
    Tensor vW_;
    Tensor mW_;
    Tensor vAW_;
};

} // namespace brogameagent::nn
