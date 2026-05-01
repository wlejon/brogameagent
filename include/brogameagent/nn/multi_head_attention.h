#pragma once

#include "circuits.h"
#include "device.h"
#include "tensor.h"

#ifdef BGA_HAS_GPU
#include "gpu/tensor.h"
#endif

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
// Forward:
//   per head h: Q_h = X @ Wq_h^T, K_h = X @ Wk_h^T, V_h = X @ Wv_h^T
//   scores_h(i,j) = Q_h_i · K_h_j / sqrt(head_dim)
//   weights_h     = softmax_row(scores_h)        (mask aware)
//   Y_h           = weights_h @ V_h              (K, head_dim)
//   Y_concat      = concat over heads            (K, D)
//   O             = Y_concat @ Wo^T              (K, D)
//
// Mask semantics mirror ScaledDotProductAttention:
//   - mask is length K, 1 valid / 0 invalid (float*).
//   - Invalid keys are excluded from the softmax denominator.
//   - Invalid query rows produce zero output rows.
//
// Backward accumulates into dWq/dWk/dWv/dWo (does NOT overwrite) — caller
// is responsible for zero_grad between batches.
//
// GPU dispatch: device_ tracks where parameters live. `to(Device)` migrates
// host↔device. CPU forward/backward overloads are unchanged. The GPU
// overloads call mha_forward_gpu / mha_backward_gpu and own a device-side
// cache for Qh/Kh/Vh/Attnh/Yconcat.

class MultiHeadAttention : public ICircuit {
public:
    MultiHeadAttention() = default;

    // Copy semantics: copy host-side state only; destination starts on
    // Device::CPU. Mirrors ScaledDotProductAttention.
    MultiHeadAttention(const MultiHeadAttention& o) { copy_host_(o); }
    MultiHeadAttention& operator=(const MultiHeadAttention& o) {
        if (this != &o) copy_host_(o);
        return *this;
    }
    MultiHeadAttention(MultiHeadAttention&&) = default;
    MultiHeadAttention& operator=(MultiHeadAttention&&) = default;

    // num_heads must divide dim. n_slots = K (number of tokens).
    void init(int n_slots, int dim, int num_heads, uint64_t& rng_state);

    int n_slots()   const { return n_; }
    int dim()       const { return d_; }
    int num_heads() const { return h_; }
    int head_dim()  const { return dh_; }

    // X: (K, D). mask: length K (1 valid, 0 invalid), may be nullptr.
    // O: (K, D); resized if mis-shaped.
    void forward(const Tensor& X, const float* mask, Tensor& O);
    void backward(const Tensor& dO, Tensor& dX);

#ifdef BGA_HAS_GPU
    // GPU code path. mask_dev (length K) is an optional device pointer.
    void forward(const gpu::GpuTensor& X, const float* mask_dev,
                 gpu::GpuTensor& O);
    void backward(const gpu::GpuTensor& dO, gpu::GpuTensor& dX);

    // Inference-only batched forward. Input is (B*K, D); per-batch
    // attention is independent (softmax over K tokens within each batch).
    // Loops B times calling the existing single-sample GPU forward against
    // GpuTensor row-views — internal caches get clobbered each iteration
    // (acceptable; no backward is run). mask_R_dev is (B*K,) or null.
    void forward_inference_batched(const gpu::GpuTensor& X_RD,
                                    const float* mask_R_dev,
                                    gpu::GpuTensor& Y_RD,
                                    int B, int K);
#endif

    Device device() const { return device_; }
    void to(Device d);

    const char* name() const override { return "MultiHeadAttention"; }
    int  num_params() const override {
        return Wq_.size() + Wk_.size() + Wv_.size() + Wo_.size();
    }
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor&       Wq()  { return Wq_; }
    Tensor&       Wk()  { return Wk_; }
    Tensor&       Wv()  { return Wv_; }
    Tensor&       Wo()  { return Wo_; }
    Tensor&       dWq() { return dWq_; }
    Tensor&       dWk() { return dWk_; }
    Tensor&       dWv() { return dWv_; }
    Tensor&       dWo() { return dWo_; }

private:
    void copy_host_(const MultiHeadAttention& o) {
        n_ = o.n_; d_ = o.d_; h_ = o.h_; dh_ = o.dh_;
        Wq_ = o.Wq_; Wk_ = o.Wk_; Wv_ = o.Wv_; Wo_ = o.Wo_;
        dWq_ = o.dWq_; dWk_ = o.dWk_; dWv_ = o.dWv_; dWo_ = o.dWo_;
        vWq_ = o.vWq_; vWk_ = o.vWk_; vWv_ = o.vWv_; vWo_ = o.vWo_;
        mWq_ = o.mWq_; mWk_ = o.mWk_; mWv_ = o.mWv_; mWo_ = o.mWo_;
        vAWq_ = o.vAWq_; vAWk_ = o.vAWk_; vAWv_ = o.vAWv_; vAWo_ = o.vAWo_;
        X_cache_ = o.X_cache_;
        Qh_ = o.Qh_; Kh_ = o.Kh_; Vh_ = o.Vh_;
        Attnh_ = o.Attnh_;
        Yconcat_ = o.Yconcat_;
        mask_cache_ = o.mask_cache_;
        device_ = Device::CPU;
        // GPU mirrors deliberately left default-constructed.
    }

    int n_  = 0;
    int d_  = 0;
    int h_  = 1;
    int dh_ = 0;

    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor dWq_, dWk_, dWv_, dWo_;
    Tensor vWq_, vWk_, vWv_, vWo_;
    Tensor mWq_, mWk_, mWv_, mWo_;
    Tensor vAWq_, vAWk_, vAWv_, vAWo_;

    // Caches for backward.
    Tensor X_cache_;            // (K, D)
    // Per-head Q/K/V/Attn/Yh stored as (h * K, head_dim) and (h * K, K) flats.
    std::vector<Tensor> Qh_;    // h copies of (K, dh)
    std::vector<Tensor> Kh_;
    std::vector<Tensor> Vh_;
    std::vector<Tensor> Attnh_; // h copies of (K, K)
    Tensor Yconcat_;            // (K, D) — pre-Wo concat output
    std::vector<uint8_t> mask_cache_;

    Device device_ = Device::CPU;
#ifdef BGA_HAS_GPU
    gpu::GpuTensor Wq_g_, Wk_g_, Wv_g_, Wo_g_;
    gpu::GpuTensor dWq_g_, dWk_g_, dWv_g_, dWo_g_;
    gpu::GpuTensor vWq_g_, vWk_g_, vWv_g_, vWo_g_;
    gpu::GpuTensor mWq_g_, mWk_g_, mWv_g_, mWo_g_;
    gpu::GpuTensor vAWq_g_, vAWk_g_, vAWv_g_, vAWo_g_;
    // Forward caches (flat layout matching the GPU op contract):
    //   Qh_g_, Kh_g_, Vh_g_: (h * K, dh)
    //   Attnh_g_:            (h * K, K)
    //   Yconcat_g_:          (K, D)
    //   X_cache_g_:          (K, D)
    gpu::GpuTensor X_cache_g_;
    gpu::GpuTensor Qh_g_, Kh_g_, Vh_g_;
    gpu::GpuTensor Attnh_g_;
    gpu::GpuTensor Yconcat_g_;
    // Mask pointer used by the most recent GPU forward; cached for backward.
    // Non-owning — caller manages lifetime.
    const float* last_mask_dev_ = nullptr;
#endif
};

} // namespace brogameagent::nn
