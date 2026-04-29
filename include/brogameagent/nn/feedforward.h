#pragma once

#include "circuits.h"
#include "device.h"
#include "tensor.h"

#ifdef BGA_HAS_CUDA
#include "gpu/tensor.h"
#endif

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── FeedForward ───────────────────────────────────────────────────────────
//
// Position-wise 2-layer MLP: x → Linear(D, d_ff) → ReLU → Linear(d_ff, D).
// Operates on (K, D) input matrices: same parameters applied independently
// to each of the K rows. Parameters:
//   W1 (d_ff, D), b1 (d_ff)
//   W2 (D, d_ff), b2 (D)
// Activation is ReLU (deliberately simple; GELU can come later without
// touching call sites).
//
// GPU dispatch: device_ tracks where parameters live. `to(Device)` migrates
// host↔device. CPU forward/backward overloads are unchanged. The GPU path
// reuses the per-vector linear_*_gpu kernels by walking the K rows of the
// (K, D) input via GpuTensor::view() — simpler than introducing a batched
// linear primitive, and lets us keep this layer composed of existing ops.

class FeedForward : public ICircuit {
public:
    FeedForward() = default;

    FeedForward(const FeedForward& o) { copy_host_(o); }
    FeedForward& operator=(const FeedForward& o) {
        if (this != &o) copy_host_(o);
        return *this;
    }
    FeedForward(FeedForward&&) = default;
    FeedForward& operator=(FeedForward&&) = default;

    void init(int dim, int d_ff, uint64_t& rng_state);

    int dim()  const { return d_; }
    int d_ff() const { return df_; }

    // X: (K, D); Y: (K, D); resized if mis-shaped.
    void forward(const Tensor& X, Tensor& Y);
    void backward(const Tensor& dY, Tensor& dX);

#ifdef BGA_HAS_CUDA
    void forward(const gpu::GpuTensor& X, gpu::GpuTensor& Y);
    void backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX);

    // Inference-only batched forward. Input/output (R, D); FF is purely
    // position-wise so we just call batched Linear→ReLU→Linear over R rows
    // in three launches. Allocates two scratch (R, d_ff) tensors per call.
    void forward_inference_batched(const gpu::GpuTensor& X_RD,
                                    gpu::GpuTensor& Y_RD);
#endif

    Device device() const { return device_; }
    void to(Device d);

    const char* name() const override { return "FeedForward"; }
    int  num_params() const override {
        return W1_.size() + b1_.size() + W2_.size() + b2_.size();
    }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor& W1() { return W1_; } Tensor& b1() { return b1_; }
    Tensor& W2() { return W2_; } Tensor& b2() { return b2_; }
    Tensor& dW1() { return dW1_; } Tensor& dB1() { return dB1_; }
    Tensor& dW2() { return dW2_; } Tensor& dB2() { return dB2_; }

private:
    void copy_host_(const FeedForward& o) {
        d_ = o.d_; df_ = o.df_;
        W1_ = o.W1_; b1_ = o.b1_; W2_ = o.W2_; b2_ = o.b2_;
        dW1_ = o.dW1_; dB1_ = o.dB1_; dW2_ = o.dW2_; dB2_ = o.dB2_;
        vW1_ = o.vW1_; vB1_ = o.vB1_; vW2_ = o.vW2_; vB2_ = o.vB2_;
        mW1_ = o.mW1_; mB1_ = o.mB1_; mW2_ = o.mW2_; mB2_ = o.mB2_;
        vAW1_ = o.vAW1_; vAB1_ = o.vAB1_; vAW2_ = o.vAW2_; vAB2_ = o.vAB2_;
        X_cache_ = o.X_cache_;
        H_pre_ = o.H_pre_;
        H_post_ = o.H_post_;
        device_ = Device::CPU;
    }

    int d_  = 0;
    int df_ = 0;

    Tensor W1_, b1_, W2_, b2_;
    Tensor dW1_, dB1_, dW2_, dB2_;
    Tensor vW1_, vB1_, vW2_, vB2_;
    Tensor mW1_, mB1_, mW2_, mB2_;
    Tensor vAW1_, vAB1_, vAW2_, vAB2_;

    // Caches.
    Tensor X_cache_;   // (K, D)
    Tensor H_pre_;     // (K, d_ff) pre-activation
    Tensor H_post_;    // (K, d_ff) post-ReLU

    Device device_ = Device::CPU;
#ifdef BGA_HAS_CUDA
    gpu::GpuTensor W1_g_, b1_g_, W2_g_, b2_g_;
    gpu::GpuTensor dW1_g_, dB1_g_, dW2_g_, dB2_g_;
    gpu::GpuTensor vW1_g_, vB1_g_, vW2_g_, vB2_g_;
    gpu::GpuTensor mW1_g_, mB1_g_, mW2_g_, mB2_g_;
    gpu::GpuTensor vAW1_g_, vAB1_g_, vAW2_g_, vAB2_g_;
    // Forward caches mirroring the (K, D)/(K, df) host caches.
    gpu::GpuTensor X_cache_g_;
    gpu::GpuTensor H_pre_g_;
    gpu::GpuTensor H_post_g_;
#endif
};

} // namespace brogameagent::nn
