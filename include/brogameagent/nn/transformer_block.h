#pragma once

#include "circuits.h"
#include "feedforward.h"
#include "multi_head_attention.h"
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── TransformerBlock ──────────────────────────────────────────────────────
//
// Composes a multi-head self-attention sub-layer and a position-wise
// feed-forward sub-layer with two LayerNorms and two residual connections.
// LayerNorm parameters (gamma, beta of size D) are shared across the K
// token positions — standard transformer convention.
//
// Pre-norm (default — better gradient flow for stacks):
//     a = x + MHA(LN1(x), mask)
//     y = a + FF(LN2(a))
//
// Post-norm (original "Attention Is All You Need" formulation):
//     a = LN1(x + MHA(x, mask))
//     y = LN2(a + FF(a))
//
// Device: composite layer. `to(brotensor::Device)` recurses into children
// (mha, ff) and the two RowLN sublayers. brotensor ops dispatch on the
// device their operand tensors live on, so a single forward/backward path
// covers every backend.

enum class NormPlacement { PreNorm, PostNorm };

class TransformerBlock : public ICircuit {
public:
    TransformerBlock() = default;

    struct Config {
        int dim       = 32;
        int num_heads = 4;
        int d_ff      = 64;
        int n_slots   = 0;                  // K — used to size caches
        float ln_eps  = 1e-5f;
        NormPlacement norm = NormPlacement::PreNorm;
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int dim()       const { return cfg_.dim; }
    int num_heads() const { return cfg_.num_heads; }
    int d_ff()      const { return cfg_.d_ff; }
    NormPlacement norm() const { return cfg_.norm; }

    // X: (K, D), mask length K (1 valid / 0 invalid) or nullptr.
    // Y: (K, D); resized if mis-shaped.
    void forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& Y);
    void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);

    brotensor::Device device() const { return device_; }
    void to(brotensor::Device d);

    const char* name() const override { return "TransformerBlock"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    // Serialization order: gamma1, beta1, MHA, gamma2, beta2, FF.
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    MultiHeadAttention& mha() { return mha_; }
    FeedForward&        ff()  { return ff_; }

    // Per-row LayerNorm helpers (gamma/beta of size D, shared across K rows).
    // Stores per-row xhat (K, D), mean[K], rstd[K] for backward.
    // Exposed publicly so TransformerEncoder can reuse the same primitive
    // for its optional final-LN.
    //
    // Device: device_ tracks where parameters live. The per-row mean/rstd
    // cache stays host-side; xhat is a device tensor migrated by to().
    struct RowLN {
        brotensor::Tensor gamma, beta;
        brotensor::Tensor dGamma, dBeta;
        brotensor::Tensor vGamma, vBeta;
        brotensor::Tensor mGamma, mBeta;
        brotensor::Tensor vAGamma, vABeta;
        // Caches.
        brotensor::Tensor xhat;            // (K, D)
        std::vector<float> mean;
        std::vector<float> rstd;
        float eps = 1e-5f;

        brotensor::Device device_ = brotensor::Device::CPU;

        void init(int D, float eps);
        void to(brotensor::Device d);
        void zero_grad();
        void sgd_step(float lr, float momentum);
        void adam_step(float lr, float beta1, float beta2, float eps, int step);
        // X, Y both (K, D).
        void forward(const brotensor::Tensor& X, brotensor::Tensor& Y);
        void backward(const brotensor::Tensor& dY, brotensor::Tensor& dX);
        void save_to(std::vector<uint8_t>& out) const;
        void load_from(const uint8_t* data, size_t& offset, size_t size);
        int num_params() const { return gamma.size() + beta.size(); }
    };

private:
    Config cfg_{};
    RowLN ln1_;
    RowLN ln2_;
    MultiHeadAttention mha_;
    FeedForward ff_;

    // Caches for backward (named per pre-norm path; post-norm reuses them
    // with adjusted semantics — see implementation comments).
    brotensor::Tensor X_cache_;
    brotensor::Tensor LN1_out_;     // pre-norm: LN1(x); post-norm: x + MHA(x)
    brotensor::Tensor MHA_out_;     // MHA(LN1_out)        (pre-norm) or MHA(x) (post-norm)
    brotensor::Tensor A_cache_;     // x + MHA_out (pre)   or  LN1(x + MHA(x)) (post)
    brotensor::Tensor LN2_out_;     // LN2(A) (pre)        or  A (post)
    brotensor::Tensor FF_out_;      // FF(LN2_out) (pre)   or  FF(A) (post)

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
