#pragma once

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

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
//
// Device: brotensor::Tensor carries its own Device tag and brotensor ops
// dispatch on it at runtime, so a circuit has a single forward/backward that
// runs on whatever device its tensors live on. `to(Device)` migrates every
// tensor a circuit owns; `device()` reports where they currently are.

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

    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);
    // Explicit-input overload: accumulate dW/db using `x_input` instead of the
    // internal x_cache_ stashed by forward(). Used by callers that share a
    // single Linear across multiple forward/backward pairs (e.g. the per-slot
    // streams in DeepSetsEncoder).
    void backward(const brotensor::Tensor& x_input, const brotensor::Tensor& dY, brotensor::Tensor& dX);

    // Batched-training forward/backward. Cache slot is independent from the
    // single-sample forward() cache so the two paths can coexist.
    void forward_batched_train(const brotensor::Tensor& X_BD, brotensor::Tensor& Y_BD);
    void backward_batched(const brotensor::Tensor& dY_BD, brotensor::Tensor& dX_BD);

    int in_dim() const  { return W_.cols; }
    int out_dim() const { return W_.rows; }

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "Linear"; }
    int  num_params() const override { return W_.size() + b_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    // Adam optimizer step; uses per-parameter m/v moment buffers. `step` is a
    // 1-based step counter for bias correction.
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    // Inspect for tests / CLI.
    brotensor::Tensor&       W()       { return W_; }
    const brotensor::Tensor& W() const { return W_; }
    brotensor::Tensor&       b()       { return b_; }
    const brotensor::Tensor& b() const { return b_; }
    brotensor::Tensor&       dW()       { return dW_; }
    const brotensor::Tensor& dW() const { return dW_; }
    brotensor::Tensor&       dB()       { return dB_; }
    const brotensor::Tensor& dB() const { return dB_; }

private:
    brotensor::Tensor W_, b_;
    brotensor::Tensor dW_, dB_;
    brotensor::Tensor vW_, vB_;   // SGD momentum velocities
    // Adam moment buffers (m: first moment, v_a: second moment).
    brotensor::Tensor mW_, mB_;
    brotensor::Tensor vAW_, vAB_;
    brotensor::Tensor x_cache_;       // input stashed at forward(), used by backward()
    brotensor::Tensor x_cache_btr_;   // batched-train input cache, independent slot

    brotensor::Device device_ = brotensor::Device::CPU;
};

// ─── ReLU / Tanh (stateless but cache input/output) ────────────────────────

class Relu : public ICircuit {
public:
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
        x_cache_ = x;
        brotensor::relu_forward(x, y);
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::relu_backward(x_cache_, dY, dX);
    }
    const char* name() const override { return "ReLU"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void adam_step(float, float, float, float, int) {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    brotensor::Tensor x_cache_;
};

class Tanh : public ICircuit {
public:
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
        brotensor::tanh_forward(x, y);
        y_cache_ = y;
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::tanh_backward(y_cache_, dY, dX);
    }
    const char* name() const override { return "Tanh"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void adam_step(float, float, float, float, int) {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    brotensor::Tensor y_cache_;
};

class Sigmoid : public ICircuit {
public:
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
        brotensor::sigmoid_forward(x, y);
        y_cache_ = y;
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::sigmoid_backward(y_cache_, dY, dX);
    }
    const char* name() const override { return "Sigmoid"; }
    int  num_params() const override { return 0; }
    void zero_grad() override {}
    void sgd_step(float, float) override {}
    void adam_step(float, float, float, float, int) {}
    void save_to(std::vector<uint8_t>&) const override {}
    void load_from(const uint8_t*, size_t&, size_t) override {}
private:
    brotensor::Tensor y_cache_;
};

// ─── Serialization helpers (tensor-level) ─────────────────────────────────

void tensor_write(const brotensor::Tensor& t, std::vector<uint8_t>& out);
void tensor_read(brotensor::Tensor& t, const uint8_t* data, size_t& offset, size_t size);

// ─── Optimizer helpers ────────────────────────────────────────────────────
//
// adam_step_cpu performs a single Adam update in place over a flat parameter
// buffer. Kept as a named helper for call sites that predate the unified
// brotensor::adam_step; it forwards to brotensor::adam_step, which dispatches
// on the tensors' device.
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   param -= lr * m_hat / (sqrt(v_hat) + eps)   (m_hat/v_hat bias-corrected)
// `step` is a 1-based step counter. All four tensors must share shape/device.
void adam_step_cpu(brotensor::Tensor& param, const brotensor::Tensor& grad,
                   brotensor::Tensor& m, brotensor::Tensor& v,
                   float lr, float beta1, float beta2, float eps, int step);

} // namespace brogameagent::nn
