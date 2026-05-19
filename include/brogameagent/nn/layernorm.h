#pragma once

#include "circuits.h"
#include <brotensor/device.h>
#include <brotensor/tensor.h>

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/tensor.h>
#endif

#include <cstdint>

namespace brogameagent::nn {

// ─── LayerNorm ─────────────────────────────────────────────────────────────
//
// Per-vector normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
// Learnable gamma, beta of size N. Single-sample only (vector in, vector out).
//
// GPU dispatch: device_ tracks where parameters live. `to(brotensor::Device)` migrates
// host↔device. The CPU forward/backward overloads are unchanged. The GPU
// overloads call ::brotensor::layernorm_*_gpu.

class LayerNorm : public ICircuit {
public:
    LayerNorm() = default;
    LayerNorm(int n, float eps = 1e-5f) { init(n, eps); }

    // Copy semantics: copy the host-side state only. The destination always
    // starts on brotensor::Device::CPU (no GPU mirrors copied) — callers must to(GPU)
    // again if they want a device-resident copy. This preserves the move-only
    // GpuTensor invariant while keeping LayerNorm usable in std::vector and
    // assignable inside composite layers.
    LayerNorm(const LayerNorm& o) { copy_host_(o); }
    LayerNorm& operator=(const LayerNorm& o) { if (this != &o) copy_host_(o); return *this; }
    LayerNorm(LayerNorm&&) = default;
    LayerNorm& operator=(LayerNorm&&) = default;

    void init(int n, float eps = 1e-5f);

    // CPU code path — literally unchanged from pre-retrofit behavior.
    void forward(const brotensor::Tensor& x, brotensor::Tensor& y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

#ifdef BROTENSOR_HAS_GPU
    // GPU code path. Parameters must already be on brotensor::Device::GPU (call to()).
    void forward(const brotensor::GpuTensor& x, brotensor::GpuTensor& y);
    void backward(const brotensor::GpuTensor& dY, brotensor::GpuTensor& dX);
#endif

    int dim() const { return gamma_.size(); }

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "LayerNorm"; }
    int  num_params() const override { return gamma_.size() + beta_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    brotensor::Tensor&       gamma()       { return gamma_; }
    const brotensor::Tensor& gamma() const { return gamma_; }
    brotensor::Tensor&       beta()        { return beta_; }
    const brotensor::Tensor& beta()  const { return beta_; }
    brotensor::Tensor&       dGamma()       { return dGamma_; }
    brotensor::Tensor&       dBeta()        { return dBeta_; }

private:
    void copy_host_(const LayerNorm& o) {
        gamma_ = o.gamma_; beta_ = o.beta_;
        dGamma_ = o.dGamma_; dBeta_ = o.dBeta_;
        vGamma_ = o.vGamma_; vBeta_ = o.vBeta_;
        mGamma_ = o.mGamma_; mBeta_ = o.mBeta_;
        vAGamma_ = o.vAGamma_; vABeta_ = o.vABeta_;
        eps_ = o.eps_;
        xhat_ = o.xhat_;
        mean_ = o.mean_;
        rstd_ = o.rstd_;
        device_ = brotensor::Device::CPU;
        // GPU mirrors deliberately left default-constructed (empty).
    }

    brotensor::Tensor gamma_, beta_;
    brotensor::Tensor dGamma_, dBeta_;
    brotensor::Tensor vGamma_, vBeta_;
    // Adam moment buffers.
    brotensor::Tensor mGamma_, mBeta_;
    brotensor::Tensor vAGamma_, vABeta_;
    float eps_ = 1e-5f;
    // Caches for backward.
    brotensor::Tensor xhat_;     // normalized x
    float mean_ = 0.0f;
    float rstd_ = 0.0f;

    brotensor::Device device_ = brotensor::Device::CPU;
#ifdef BROTENSOR_HAS_GPU
    // GPU mirrors. Allocated lazily on first to(GPU); updated by to() in
    // either direction. Forward/backward also create xhat_g_ as needed.
    brotensor::GpuTensor gamma_g_, beta_g_;
    brotensor::GpuTensor dGamma_g_, dBeta_g_;
    brotensor::GpuTensor vGamma_g_, vBeta_g_;
    // Adam GPU mirrors.
    brotensor::GpuTensor mGamma_g_, mBeta_g_;
    brotensor::GpuTensor vAGamma_g_, vABeta_g_;
    brotensor::GpuTensor xhat_g_;
#endif
};

} // namespace brogameagent::nn
