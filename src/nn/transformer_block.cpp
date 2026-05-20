#include "brogameagent/nn/transformer_block.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cmath>

namespace brogameagent::nn {

// ─── RowLN ─────────────────────────────────────────────────────────────────

void TransformerBlock::RowLN::init(int D, float e) {
    eps = e;
    gamma.resize(D, 1); beta.resize(D, 1);
    dGamma.resize(D, 1); dBeta.resize(D, 1);
    vGamma.resize(D, 1); vBeta.resize(D, 1);
    mGamma.resize(D, 1); mBeta.resize(D, 1);
    vAGamma.resize(D, 1); vABeta.resize(D, 1);
    for (int i = 0; i < D; ++i) { gamma[i] = 1.0f; beta[i] = 0.0f; }
    dGamma.zero(); dBeta.zero();
    vGamma.zero(); vBeta.zero();
    mGamma.zero(); mBeta.zero();
    vAGamma.zero(); vABeta.zero();
}

void TransformerBlock::RowLN::zero_grad() {
    dGamma.zero(); dBeta.zero();
}

void TransformerBlock::RowLN::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(gamma, dGamma, vGamma, lr, momentum);
    brotensor::sgd_step(beta,  dBeta,  vBeta,  lr, momentum);
}

void TransformerBlock::RowLN::adam_step(float lr, float beta1, float beta2,
                                        float eps_a, int step) {
    brotensor::adam_step(gamma, dGamma, mGamma, vAGamma, lr, beta1, beta2, eps_a, step);
    brotensor::adam_step(beta,  dBeta,  mBeta,  vABeta,  lr, beta1, beta2, eps_a, step);
}

void TransformerBlock::RowLN::forward(const brotensor::Tensor& X, brotensor::Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == gamma.size());
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    // xhat caches normalised x; keep it on the same device as the input.
    if (xhat.rows != K || xhat.cols != D || xhat.device != X.device) {
        xhat = brotensor::Tensor::empty_on(X.device, K, D);
    }
    mean.assign(K, 0.0f);
    rstd.assign(K, 0.0f);

    // Per-row layernorm via the per-vector device-dispatched op; mean/rstd are
    // host scalars populated by each call (one per row). Row views are
    // non-owning windows over each tensor's backing buffer.
    for (int i = 0; i < K; ++i) {
        const size_t off = static_cast<size_t>(i) * D;
        brotensor::Tensor x_row = brotensor::Tensor::view(
            X.device, static_cast<float*>(X.data) + off, D, 1);
        brotensor::Tensor y_row = brotensor::Tensor::view(
            Y.device, static_cast<float*>(Y.data) + off, D, 1);
        brotensor::Tensor xhat_row = brotensor::Tensor::view(
            xhat.device, static_cast<float*>(xhat.data) + off, D, 1);
        brotensor::layernorm_forward(x_row, gamma, beta,
                                     y_row, xhat_row,
                                     mean[i], rstd[i], eps);
    }
}

void TransformerBlock::RowLN::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    const int K = dY.rows;
    const int D = dY.cols;
    if (dX.rows != K || dX.cols != D || dX.device != dY.device) {
        dX = brotensor::Tensor::empty_on(dY.device, K, D);
    }
    for (int i = 0; i < K; ++i) {
        const size_t off = static_cast<size_t>(i) * D;
        brotensor::Tensor dy_row = brotensor::Tensor::view(
            dY.device, static_cast<float*>(dY.data) + off, D, 1);
        brotensor::Tensor xhat_row = brotensor::Tensor::view(
            xhat.device, static_cast<float*>(xhat.data) + off, D, 1);
        brotensor::Tensor dx_row = brotensor::Tensor::view(
            dX.device, static_cast<float*>(dX.data) + off, D, 1);
        brotensor::layernorm_backward(dy_row, xhat_row, gamma, rstd[i],
                                      dx_row, dGamma, dBeta);
    }
}

void TransformerBlock::RowLN::to(brotensor::Device d) {
    if (d == device_) return;
    gamma   = gamma.to(d);   beta   = beta.to(d);
    dGamma  = dGamma.to(d);  dBeta  = dBeta.to(d);
    vGamma  = vGamma.to(d);  vBeta  = vBeta.to(d);
    mGamma  = mGamma.to(d);  mBeta  = mBeta.to(d);
    vAGamma = vAGamma.to(d); vABeta = vABeta.to(d);
    if (xhat.size() > 0) xhat = xhat.to(d);
    device_ = d;
}

void TransformerBlock::RowLN::save_to(std::vector<uint8_t>& out) const {
    tensor_write(gamma, out);
    tensor_write(beta, out);
}

void TransformerBlock::RowLN::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(gamma, data, offset, size);
    tensor_read(beta, data, offset, size);
    const int D = gamma.size();
    dGamma.resize(D, 1); dBeta.resize(D, 1);
    vGamma.resize(D, 1); vBeta.resize(D, 1);
    mGamma.resize(D, 1); mBeta.resize(D, 1);
    vAGamma.resize(D, 1); vABeta.resize(D, 1);
    dGamma.zero(); dBeta.zero();
    vGamma.zero(); vBeta.zero();
    mGamma.zero(); mBeta.zero();
    vAGamma.zero(); vABeta.zero();
}

// ─── TransformerBlock ──────────────────────────────────────────────────────

void TransformerBlock::init(const Config& cfg, uint64_t& rng_state) {
    cfg_ = cfg;
    ln1_.init(cfg.dim, cfg.ln_eps);
    ln2_.init(cfg.dim, cfg.ln_eps);
    mha_.init(cfg.n_slots, cfg.dim, cfg.num_heads, rng_state);
    ff_.init(cfg.dim, cfg.d_ff, rng_state);
}

int TransformerBlock::num_params() const {
    return ln1_.num_params() + ln2_.num_params()
         + mha_.num_params() + ff_.num_params();
}

void TransformerBlock::forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == cfg_.dim);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    X_cache_ = X;

    // The mask (if any) is forwarded verbatim to MHA, which hands it straight
    // to the device-dispatched mha op — it must live on X's device. No host
    // snapshot is taken; the caller owns the buffer across forward/backward.
    const float* m = mask;  // pass through unchanged

    if (cfg_.norm == NormPlacement::PreNorm) {
        // a = x + MHA(LN1(x))
        ln1_.forward(X, LN1_out_);
        mha_.forward(LN1_out_, m, MHA_out_);
        A_cache_ = X;
        brotensor::add_inplace(A_cache_, MHA_out_);

        // y = a + FF(LN2(a))
        ln2_.forward(A_cache_, LN2_out_);
        ff_.forward(LN2_out_, FF_out_);
        Y = A_cache_;
        brotensor::add_inplace(Y, FF_out_);
    } else {
        // post-norm: a = LN1(x + MHA(x))
        mha_.forward(X, m, MHA_out_);
        brotensor::Tensor pre_ln1 = X;
        brotensor::add_inplace(pre_ln1, MHA_out_);
        ln1_.forward(pre_ln1, A_cache_);

        // y = LN2(a + FF(a))
        ff_.forward(A_cache_, FF_out_);
        brotensor::Tensor pre_ln2 = A_cache_;
        brotensor::add_inplace(pre_ln2, FF_out_);
        ln2_.forward(pre_ln2, Y);
    }
}

void TransformerBlock::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    const int K = X_cache_.rows;
    const int D = X_cache_.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);

    if (cfg_.norm == NormPlacement::PreNorm) {
        // y = a + FF(LN2(a))
        // dFF_out = dY ; dA += dY (residual)
        brotensor::Tensor dA = dY;

        brotensor::Tensor dLN2_out = brotensor::Tensor::zeros_on(dY.device, K, D);
        ff_.backward(dY, dLN2_out);

        brotensor::Tensor dA_from_ln = brotensor::Tensor::zeros_on(dY.device, K, D);
        ln2_.backward(dLN2_out, dA_from_ln);
        brotensor::add_inplace(dA, dA_from_ln);

        // a = x + MHA(LN1(x))
        // dMHA_out = dA → propagate through MHA + LN1. Compute that first
        // because the MHA backward consumes dA before we reuse it as dX.
        brotensor::Tensor dLN1_out = brotensor::Tensor::zeros_on(dY.device, K, D);
        mha_.backward(dA, dLN1_out);

        brotensor::Tensor dX_from_ln = brotensor::Tensor::zeros_on(dY.device, K, D);
        ln1_.backward(dLN1_out, dX_from_ln);
        // dX = dA (residual) + dX_from_ln.
        dX = dA;
        brotensor::add_inplace(dX, dX_from_ln);
    } else {
        // Post-norm.
        // y = LN2(a + FF(a))
        brotensor::Tensor d_pre_ln2 = brotensor::Tensor::zeros_on(dY.device, K, D);
        ln2_.backward(dY, d_pre_ln2);
        // pre_ln2 = a + FF(a)  → dA = d_pre_ln2 (residual) + FF.bwd(d_pre_ln2)
        brotensor::Tensor dA_ff = brotensor::Tensor::zeros_on(dY.device, K, D);
        ff_.backward(d_pre_ln2, dA_ff);
        brotensor::Tensor dA = d_pre_ln2;
        brotensor::add_inplace(dA, dA_ff);

        // a = LN1(pre_ln1) where pre_ln1 = x + MHA(x)
        brotensor::Tensor d_pre_ln1 = brotensor::Tensor::zeros_on(dY.device, K, D);
        ln1_.backward(dA, d_pre_ln1);

        // pre_ln1 = x + MHA(x) → dX (residual) + MHA.bwd
        brotensor::Tensor dX_from_mha = brotensor::Tensor::zeros_on(dY.device, K, D);
        mha_.backward(d_pre_ln1, dX_from_mha);
        dX = d_pre_ln1;
        brotensor::add_inplace(dX, dX_from_mha);
    }
}

void TransformerBlock::to(brotensor::Device d) {
    if (d == device_) return;
    ln1_.to(d);
    ln2_.to(d);
    mha_.to(d);
    ff_.to(d);
    // Activation caches — migrate so a forward after to() doesn't feed a
    // stale CPU-resident cache into a device op.
    X_cache_ = X_cache_.to(d);
    LN1_out_ = LN1_out_.to(d);
    MHA_out_ = MHA_out_.to(d);
    A_cache_ = A_cache_.to(d);
    LN2_out_ = LN2_out_.to(d);
    FF_out_  = FF_out_.to(d);
    device_ = d;
}

void TransformerBlock::zero_grad() {
    ln1_.zero_grad();
    ln2_.zero_grad();
    mha_.zero_grad();
    ff_.zero_grad();
}

void TransformerBlock::sgd_step(float lr, float momentum) {
    ln1_.sgd_step(lr, momentum);
    ln2_.sgd_step(lr, momentum);
    mha_.sgd_step(lr, momentum);
    ff_.sgd_step(lr, momentum);
}

void TransformerBlock::adam_step(float lr, float beta1, float beta2,
                                 float eps, int step) {
    ln1_.adam_step(lr, beta1, beta2, eps, step);
    ln2_.adam_step(lr, beta1, beta2, eps, step);
    mha_.adam_step(lr, beta1, beta2, eps, step);
    ff_.adam_step(lr, beta1, beta2, eps, step);
}

void TransformerBlock::save_to(std::vector<uint8_t>& out) const {
    // Order: gamma1/beta1, MHA(Wq,Wk,Wv,Wo), gamma2/beta2, FF(W1,b1,W2,b2).
    ln1_.save_to(out);
    mha_.save_to(out);
    ln2_.save_to(out);
    ff_.save_to(out);
}

void TransformerBlock::load_from(const uint8_t* data, size_t& offset, size_t size) {
    ln1_.load_from(data, offset, size);
    mha_.load_from(data, offset, size);
    ln2_.load_from(data, offset, size);
    ff_.load_from(data, offset, size);
}

} // namespace brogameagent::nn
