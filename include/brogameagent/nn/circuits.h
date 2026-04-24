#pragma once

#include "ops.h"
#include "tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brogameagent::nn {

// ─── Circuit ───────────────────────────────────────────────────────────────
//
// A circuit is a named layer that owns its weights, its gradients, and its
// own forward/backward implementations. Circuits compose by being called
// directly in a net's forward() method — no autograd, no graph. Each
// circuit caches exactly the activations its own backward needs.
//
// Contract:
//   forward(x) writes output into `out` and caches whatever backward needs.
//   backward(dY, dX) uses cache + dY to produce dX and accumulate param grads.
//   zero_grad() clears per-step gradient accumulators.
//   step(lr, momentum) applies an SGD+momentum update, uses param_velocity.
//   num_params() reports trainable param count (for logging).
//   save_to(bytes) / load_from(bytes, &offset) serialize weights only.

class ICircuit {
public:
    virtual ~ICircuit() = default;
    virtual const char* name() const = 0;
    virtual int num_params() const = 0;
    virtual void zero_grad() = 0;
    virtual void sgd_step(float lr, float momentum) = 0;
    virtual void save_to(std::vector<uint8_t>& out) const = 0;
    virtual void load_from(const uint8_t* data, size_t& offset, size_t size) = 0;
};

// ─── Linear (dense fully-connected layer) ──────────────────────────────────

class Linear : public ICircuit {
public:
    Linear() = default;
    Linear(int in_dim, int out_dim, uint64_t& rng_state);

    void init(int in_dim, int out_dim, uint64_t& rng_state);

    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

    int in_dim() const  { return W_.cols; }
    int out_dim() const { return W_.rows; }

    const char* name() const override { return "Linear"; }
    int  num_params() const override { return W_.size() + b_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    // Inspect for tests / CLI.
    Tensor&       W()       { return W_; }
    const Tensor& W() const { return W_; }
    Tensor&       b()       { return b_; }
    const Tensor& b() const { return b_; }
    Tensor&       dW()       { return dW_; }
    const Tensor& dW() const { return dW_; }
    Tensor&       dB()       { return dB_; }
    const Tensor& dB() const { return dB_; }

private:
    Tensor W_, b_;
    Tensor dW_, dB_;
    Tensor vW_, vB_;   // SGD momentum velocities
    Tensor x_cache_;   // input stashed at forward, used by backward
};

// ─── ReLU / Tanh (stateless but cache input/output) ────────────────────────

class Relu : public ICircuit {
public:
    void forward(const Tensor& x, Tensor& y) {
        x_cache_ = x;
        relu_forward(x, y);
    }
    void backward(const Tensor& dY, Tensor& dX) {
        relu_backward(x_cache_, dY, dX);
    }
    const char* name() const override { return "ReLU"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    Tensor x_cache_;
};

class Tanh : public ICircuit {
public:
    void forward(const Tensor& x, Tensor& y) {
        tanh_forward(x, y);
        y_cache_ = y;
    }
    void backward(const Tensor& dY, Tensor& dX) {
        tanh_backward(y_cache_, dY, dX);
    }
    const char* name() const override { return "Tanh"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    Tensor y_cache_;
};

class Sigmoid : public ICircuit {
public:
    void forward(const Tensor& x, Tensor& y) {
        sigmoid_forward(x, y);
        y_cache_ = y;
    }
    void backward(const Tensor& dY, Tensor& dX) {
        sigmoid_backward(y_cache_, dY, dX);
    }
    const char* name() const override { return "Sigmoid"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    Tensor y_cache_;
};

// ─── Serialization helpers (tensor-level) ─────────────────────────────────

void tensor_write(const Tensor& t, std::vector<uint8_t>& out);
void tensor_read(Tensor& t, const uint8_t* data, size_t& offset, size_t size);

} // namespace brogameagent::nn
