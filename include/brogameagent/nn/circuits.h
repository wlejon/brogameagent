#pragma once

#include <brotensor/device.h>
#include <brotensor/ops_cpu.h>
#include <brotensor/tensor.h>

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/tensor.h>
#endif

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

    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);
    // Explicit-input overload: accumulate dW/db using `x_input` instead of the
    // internal x_cache_ stashed by forward(). Mirrors the GPU API where the
    // forward input is passed explicitly to backward. Used by callers that
    // share a single Linear across multiple forward/backward pairs (e.g. the
    // per-slot streams in DeepSetsEncoder).
    void backward(const brotensor::Tensor& x_input, const brotensor::Tensor& dY, brotensor::Tensor& dX);

#ifdef BROTENSOR_HAS_GPU
    // GPU code path. Parameters must already be on brotensor::Device::GPU (call to()).
    // Caller must keep `x` alive until backward() (the layer caches a view).
    void forward(const brotensor::GpuTensor& x, brotensor::GpuTensor& y);
    void backward(const brotensor::GpuTensor& dY, brotensor::GpuTensor& dX);

    // Batched-training forward/backward. The caller must keep `X_BD` alive
    // until backward_batched (the layer caches a view). Cache slot is
    // independent from the single-sample `forward(GpuTensor)` cache so the
    // two paths can coexist without interfering.
    void forward_batched_train(const brotensor::GpuTensor& X_BD, brotensor::GpuTensor& Y_BD);
    void backward_batched(const brotensor::GpuTensor& dY_BD, brotensor::GpuTensor& dX_BD);
#endif

    int in_dim() const  { return W_.cols; }
    int out_dim() const { return W_.rows; }

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "Linear"; }
    int  num_params() const override { return W_.size() + b_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    // Adam optimizer step; uses per-parameter m/v moment buffers. `step` is a
    // 1-based step counter for bias correction. See adam_step_cpu().
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

#ifdef BROTENSOR_HAS_GPU
    brotensor::GpuTensor&       W_g()       { return W_g_; }
    brotensor::GpuTensor&       b_g()       { return b_g_; }
    brotensor::GpuTensor&       dW_g()      { return dW_g_; }
    brotensor::GpuTensor&       dB_g()      { return dB_g_; }
#endif

private:
    brotensor::Tensor W_, b_;
    brotensor::Tensor dW_, dB_;
    brotensor::Tensor vW_, vB_;   // SGD momentum velocities
    // Adam moment buffers (m: first moment, v_a: second moment).
    brotensor::Tensor mW_, mB_;
    brotensor::Tensor vAW_, vAB_;
    brotensor::Tensor x_cache_;   // input stashed at forward, used by backward

    brotensor::Device device_ = brotensor::Device::CPU;
#ifdef BROTENSOR_HAS_GPU
    // GPU mirrors. Allocated lazily on to(GPU). x_cache_g_ is a non-owning
    // view of the caller-provided x in forward(GpuTensor); backward consumes
    // it. The caller must keep x alive between forward and backward.
    brotensor::GpuTensor W_g_, b_g_;
    brotensor::GpuTensor dW_g_, dB_g_;
    brotensor::GpuTensor vW_g_, vB_g_;
    brotensor::GpuTensor mW_g_, mB_g_;
    brotensor::GpuTensor vAW_g_, vAB_g_;
    brotensor::GpuTensor x_cache_g_;     // non-owning view; lifetime = caller's x
    brotensor::GpuTensor x_cache_btr_g_; // batched-train view; independent from above
#endif
};

// ─── ReLU / Tanh (stateless but cache input/output) ────────────────────────

class Relu : public ICircuit {
public:
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
        x_cache_ = x;
        brotensor::relu_forward_cpu(x, y);
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::relu_backward_cpu(x_cache_, dY, dX);
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
        brotensor::tanh_forward_cpu(x, y);
        y_cache_ = y;
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::tanh_backward_cpu(y_cache_, dY, dX);
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
        brotensor::sigmoid_forward_cpu(x, y);
        y_cache_ = y;
    }
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
        brotensor::sigmoid_backward_cpu(y_cache_, dY, dX);
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
// buffer:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^step)
//   v_hat = v / (1 - beta2^step)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
// `step` is a 1-based step counter (the trainer increments it before calling).
// All four tensors must have identical shape.
void adam_step_cpu(brotensor::Tensor& param, const brotensor::Tensor& grad, brotensor::Tensor& m, brotensor::Tensor& v,
                   float lr, float beta1, float beta2, float eps, int step);

} // namespace brogameagent::nn
