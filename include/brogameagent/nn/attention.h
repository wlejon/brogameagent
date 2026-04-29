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

// ─── ScaledDotProductAttention ─────────────────────────────────────────────
//
// Single-head self-attention over N slots of dim D. Four learnable (D, D)
// projections Wq/Wk/Wv/Wo; no biases (simpler, works fine as a layer norm
// follows in typical stacks). Invalid rows (mask[k] == 0) are excluded from
// the softmax denominator (additive -inf on scores pre-softmax) and produce
// zero output rows — this matches the convention of softmax_forward's mask.
//
// GPU dispatch: device_ tracks where parameters live. `to(Device)` migrates
// host↔device. The CPU forward/backward overloads are unchanged. The GPU
// overloads call attention_forward_gpu / attention_backward_gpu and own a
// device-side mask buffer mirroring mask_cache_.

class ScaledDotProductAttention : public ICircuit {
public:
    ScaledDotProductAttention() = default;

    // Copy semantics: same rationale as LayerNorm — copy host state only,
    // destination starts on Device::CPU.
    ScaledDotProductAttention(const ScaledDotProductAttention& o) { copy_host_(o); }
    ScaledDotProductAttention& operator=(const ScaledDotProductAttention& o) {
        if (this != &o) copy_host_(o);
        return *this;
    }
    ScaledDotProductAttention(ScaledDotProductAttention&&) = default;
    ScaledDotProductAttention& operator=(ScaledDotProductAttention&&) = default;

    void init(int n_slots, int dim, uint64_t& rng_state, int num_heads = 1);

    int n_slots() const { return n_; }
    int dim()     const { return d_; }

    // CPU code path — unchanged.
    void forward(const Tensor& X, const float* mask, Tensor& O);
    void backward(const Tensor& dO, Tensor& dX);

#ifdef BGA_HAS_CUDA
    // GPU code path. mask_dev (length n_) is an optional device pointer.
    void forward(const gpu::GpuTensor& X, const float* mask_dev,
                 gpu::GpuTensor& O);
    void backward(const gpu::GpuTensor& dO, gpu::GpuTensor& dX);
#endif

    Device device() const { return device_; }
    void to(Device d);

    const char* name() const override { return "ScaledDotProductAttention"; }
    int  num_params() const override { return Wq_.size() + Wk_.size() + Wv_.size() + Wo_.size(); }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
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
    void copy_host_(const ScaledDotProductAttention& o) {
        n_ = o.n_; d_ = o.d_;
        Wq_ = o.Wq_; Wk_ = o.Wk_; Wv_ = o.Wv_; Wo_ = o.Wo_;
        dWq_ = o.dWq_; dWk_ = o.dWk_; dWv_ = o.dWv_; dWo_ = o.dWo_;
        vWq_ = o.vWq_; vWk_ = o.vWk_; vWv_ = o.vWv_; vWo_ = o.vWo_;
        mWq_ = o.mWq_; mWk_ = o.mWk_; mWv_ = o.mWv_; mWo_ = o.mWo_;
        vAWq_ = o.vAWq_; vAWk_ = o.vAWk_; vAWv_ = o.vAWv_; vAWo_ = o.vAWo_;
        X_cache_ = o.X_cache_;
        Q_ = o.Q_; K_ = o.K_; V_ = o.V_;
        Attn_ = o.Attn_;
        Y_ = o.Y_;
        mask_cache_ = o.mask_cache_;
        device_ = Device::CPU;
        // GPU mirrors / mask pointer left default.
    }

    int n_ = 0, d_ = 0;
    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor dWq_, dWk_, dWv_, dWo_;
    Tensor vWq_, vWk_, vWv_, vWo_;
    // Adam moment buffers.
    Tensor mWq_, mWk_, mWv_, mWo_;
    Tensor vAWq_, vAWk_, vAWv_, vAWo_;

    // Caches for backward.
    Tensor X_cache_;       // (N, D)
    Tensor Q_, K_, V_;     // (N, D)
    Tensor Attn_;          // (N, N)
    Tensor Y_;             // (N, D) = Attn @ V
    std::vector<uint8_t> mask_cache_;

    Device device_ = Device::CPU;
#ifdef BGA_HAS_CUDA
    gpu::GpuTensor Wq_g_, Wk_g_, Wv_g_, Wo_g_;
    gpu::GpuTensor dWq_g_, dWk_g_, dWv_g_, dWo_g_;
    gpu::GpuTensor vWq_g_, vWk_g_, vWv_g_, vWo_g_;
    // Adam GPU mirrors.
    gpu::GpuTensor mWq_g_, mWk_g_, mWv_g_, mWo_g_;
    gpu::GpuTensor vAWq_g_, vAWk_g_, vAWv_g_, vAWo_g_;
    gpu::GpuTensor X_cache_g_;
    gpu::GpuTensor Q_g_, K_g_, V_g_;
    gpu::GpuTensor Attn_g_;
    gpu::GpuTensor Y_g_;            // Y_pre_Wo
    // Mask pointer used by the most recent GPU forward; cached for backward.
    // Non-owning — caller manages lifetime (matches CPU API where the caller
    // owns the host mask buffer for the duration of forward+backward).
    const float* last_mask_dev_ = nullptr;
#endif
};

} // namespace brogameagent::nn
