#include "brogameagent/nn/transformer_block.h"
#include "brogameagent/nn/ops.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        dGamma_g.zero(); dBeta_g.zero();
        return;
    }
#endif
    dGamma.zero(); dBeta.zero();
}

void TransformerBlock::RowLN::sgd_step(float lr, float momentum) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::sgd_step_gpu(gamma_g, dGamma_g, vGamma_g, lr, momentum);
        gpu::sgd_step_gpu(beta_g,  dBeta_g,  vBeta_g,  lr, momentum);
        return;
    }
#endif
    const int n = gamma.size();
    for (int i = 0; i < n; ++i) {
        vGamma[i] = momentum * vGamma[i] + dGamma[i];
        gamma[i] -= lr * vGamma[i];
        vBeta[i]  = momentum * vBeta[i] + dBeta[i];
        beta[i]  -= lr * vBeta[i];
    }
}

void TransformerBlock::RowLN::adam_step(float lr, float beta1, float beta2,
                                        float eps_a, int step) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::adam_step_gpu(gamma_g, dGamma_g, mGamma_g, vAGamma_g,
                           lr, beta1, beta2, eps_a, step);
        gpu::adam_step_gpu(beta_g,  dBeta_g,  mBeta_g,  vABeta_g,
                           lr, beta1, beta2, eps_a, step);
        return;
    }
#endif
    adam_step_cpu(gamma, dGamma, mGamma, vAGamma, lr, beta1, beta2, eps_a, step);
    adam_step_cpu(beta,  dBeta,  mBeta,  vABeta,  lr, beta1, beta2, eps_a, step);
}

void TransformerBlock::RowLN::forward(const Tensor& X, Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == gamma.size());
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    if (xhat.rows != K || xhat.cols != D) xhat.resize(K, D);
    mean.assign(K, 0.0f);
    rstd.assign(K, 0.0f);
    const float nf = static_cast<float>(D);
    for (int i = 0; i < K; ++i) {
        float m = 0.0f;
        for (int k = 0; k < D; ++k) m += X(i, k);
        m /= nf;
        float v = 0.0f;
        for (int k = 0; k < D; ++k) { float d = X(i, k) - m; v += d * d; }
        v /= nf;
        const float r = 1.0f / std::sqrt(v + eps);
        mean[i] = m;
        rstd[i] = r;
        for (int k = 0; k < D; ++k) {
            xhat(i, k) = (X(i, k) - m) * r;
            Y(i, k) = gamma[k] * xhat(i, k) + beta[k];
        }
    }
}

void TransformerBlock::RowLN::backward(const Tensor& dY, Tensor& dX) {
    const int K = dY.rows;
    const int D = dY.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    const float nf = static_cast<float>(D);
    for (int i = 0; i < K; ++i) {
        // Param grads.
        for (int k = 0; k < D; ++k) {
            dGamma[k] += dY(i, k) * xhat(i, k);
            dBeta[k]  += dY(i, k);
        }
        float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
        for (int k = 0; k < D; ++k) {
            const float dxh = dY(i, k) * gamma[k];
            sum_dxhat      += dxh;
            sum_dxhat_xhat += dxh * xhat(i, k);
        }
        const float scale = rstd[i] / nf;
        for (int k = 0; k < D; ++k) {
            const float dxh = dY(i, k) * gamma[k];
            dX(i, k) = scale * (nf * dxh - sum_dxhat - xhat(i, k) * sum_dxhat_xhat);
        }
    }
}

#ifdef BGA_HAS_CUDA
void TransformerBlock::RowLN::forward(const gpu::GpuTensor& X, gpu::GpuTensor& Y) {
    assert(device_ == Device::GPU);
    const int K = X.rows;
    const int D = X.cols;
    assert(D == gamma.size());
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    if (xhat_g.rows != K || xhat_g.cols != D) xhat_g.resize(K, D);
    mean.assign(K, 0.0f);
    rstd.assign(K, 0.0f);

    // Per-row layernorm via the per-vector kernel; mean/rstd are host scalars
    // populated by each call (one host↔device sync per row).
    for (int i = 0; i < K; ++i) {
        gpu::GpuTensor x_row = gpu::GpuTensor::view(
            const_cast<float*>(X.data) + static_cast<size_t>(i) * D, D, 1);
        gpu::GpuTensor y_row = gpu::GpuTensor::view(
            Y.data + static_cast<size_t>(i) * D, D, 1);
        gpu::GpuTensor xhat_row = gpu::GpuTensor::view(
            xhat_g.data + static_cast<size_t>(i) * D, D, 1);
        gpu::layernorm_forward_gpu(x_row, gamma_g, beta_g,
                                   y_row, xhat_row,
                                   mean[i], rstd[i], eps);
    }
}

void TransformerBlock::RowLN::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    const int K = dY.rows;
    const int D = dY.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    for (int i = 0; i < K; ++i) {
        gpu::GpuTensor dy_row = gpu::GpuTensor::view(
            const_cast<float*>(dY.data) + static_cast<size_t>(i) * D, D, 1);
        gpu::GpuTensor xhat_row = gpu::GpuTensor::view(
            xhat_g.data + static_cast<size_t>(i) * D, D, 1);
        gpu::GpuTensor dx_row = gpu::GpuTensor::view(
            dX.data + static_cast<size_t>(i) * D, D, 1);
        gpu::layernorm_backward_gpu(dy_row, xhat_row, gamma_g, rstd[i],
                                    dx_row, dGamma_g, dBeta_g);
    }
}

void TransformerBlock::RowLN::forward_inference_batched(
        const gpu::GpuTensor& X_RD, gpu::GpuTensor& Y_RD) {
    gpu::layernorm_forward_inference_batched_gpu(X_RD, gamma_g, beta_g, Y_RD, eps);
}
#endif

void TransformerBlock::RowLN::to(Device d) {
    if (d == device_) return;
    device_require_cuda("TransformerBlock::RowLN");
#ifdef BGA_HAS_CUDA
    if (d == Device::GPU) {
        gpu::upload(gamma, gamma_g); gpu::upload(beta, beta_g);
        gpu::upload(dGamma, dGamma_g); gpu::upload(dBeta, dBeta_g);
        gpu::upload(vGamma, vGamma_g); gpu::upload(vBeta, vBeta_g);
        gpu::upload(mGamma, mGamma_g); gpu::upload(mBeta, mBeta_g);
        gpu::upload(vAGamma, vAGamma_g); gpu::upload(vABeta, vABeta_g);
        device_ = Device::GPU;
    } else {
        gpu::download(gamma_g, gamma); gpu::download(beta_g, beta);
        gpu::download(dGamma_g, dGamma); gpu::download(dBeta_g, dBeta);
        gpu::download(vGamma_g, vGamma); gpu::download(vBeta_g, vBeta);
        gpu::download(mGamma_g, mGamma); gpu::download(mBeta_g, mBeta);
        gpu::download(vAGamma_g, vAGamma); gpu::download(vABeta_g, vABeta);
        gpu::cuda_sync();
        device_ = Device::CPU;
    }
#endif
}

void TransformerBlock::RowLN::save_to(std::vector<uint8_t>& out) const {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        auto* self = const_cast<RowLN*>(this);
        gpu::download(gamma_g, self->gamma);
        gpu::download(beta_g, self->beta);
        gpu::cuda_sync();
    }
#endif
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
    mGamma.zero(); mBeta.zero(); vAGamma.zero(); vABeta.zero();
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::upload(gamma, gamma_g); gpu::upload(beta, beta_g);
        gpu::upload(dGamma, dGamma_g); gpu::upload(dBeta, dBeta_g);
        gpu::upload(vGamma, vGamma_g); gpu::upload(vBeta, vBeta_g);
        gpu::upload(mGamma, mGamma_g); gpu::upload(mBeta, mBeta_g);
        gpu::upload(vAGamma, vAGamma_g); gpu::upload(vABeta, vABeta_g);
    }
#endif
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

void TransformerBlock::forward(const Tensor& X, const float* mask, Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == cfg_.dim);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    if (X_cache_.rows != K || X_cache_.cols != D) X_cache_.resize(K, D);
    X_cache_ = X;

    has_mask_ = (mask != nullptr);
    mask_cache_.assign(K, 1);
    if (mask) for (int i = 0; i < K; ++i) mask_cache_[i] = mask[i] > 0.5f ? 1 : 0;
    const float* m = mask;  // pass through unchanged

    if (cfg_.norm == NormPlacement::PreNorm) {
        // a = x + MHA(LN1(x))
        ln1_.forward(X, LN1_out_);
        mha_.forward(LN1_out_, m, MHA_out_);
        if (A_cache_.rows != K || A_cache_.cols != D) A_cache_.resize(K, D);
        for (int i = 0; i < K * D; ++i) A_cache_.data[i] = X.data[i] + MHA_out_.data[i];

        // y = a + FF(LN2(a))
        ln2_.forward(A_cache_, LN2_out_);
        ff_.forward(LN2_out_, FF_out_);
        for (int i = 0; i < K * D; ++i) Y.data[i] = A_cache_.data[i] + FF_out_.data[i];
    } else {
        // post-norm: a = LN1(x + MHA(x))
        mha_.forward(X, m, MHA_out_);
        // LN1_out_ here is repurposed to hold (x + MHA(x)) before LN1 — we
        // don't actually need it after the call so just feed via temp.
        Tensor pre_ln1(K, D);
        for (int i = 0; i < K * D; ++i) pre_ln1.data[i] = X.data[i] + MHA_out_.data[i];
        ln1_.forward(pre_ln1, A_cache_);

        // y = LN2(a + FF(a))
        ff_.forward(A_cache_, FF_out_);
        Tensor pre_ln2(K, D);
        for (int i = 0; i < K * D; ++i) pre_ln2.data[i] = A_cache_.data[i] + FF_out_.data[i];
        ln2_.forward(pre_ln2, Y);
    }
}

void TransformerBlock::backward(const Tensor& dY, Tensor& dX) {
    const int K = X_cache_.rows;
    const int D = X_cache_.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    const float* m = has_mask_ ? reinterpret_cast<const float*>(nullptr) : nullptr;
    // We cached mask_cache_ as bytes; rebuild a float* if needed.
    std::vector<float> mvec;
    const float* mask_ptr = nullptr;
    if (has_mask_) {
        mvec.assign(K, 0.0f);
        for (int i = 0; i < K; ++i) mvec[i] = mask_cache_[i] ? 1.0f : 0.0f;
        mask_ptr = mvec.data();
    }
    (void)m;

    if (cfg_.norm == NormPlacement::PreNorm) {
        // y = a + FF(LN2(a))
        // dFF_out = dY ; dA += dY (residual)
        Tensor dA(K, D);
        for (int i = 0; i < K * D; ++i) dA.data[i] = dY.data[i];

        Tensor dLN2_out(K, D);
        ff_.backward(dY, dLN2_out);

        Tensor dA_from_ln(K, D);
        ln2_.backward(dLN2_out, dA_from_ln);
        for (int i = 0; i < K * D; ++i) dA.data[i] += dA_from_ln.data[i];

        // a = x + MHA(LN1(x))
        // dX = dA (residual) ; dMHA_out = dA
        Tensor dX_local(K, D);
        for (int i = 0; i < K * D; ++i) dX_local.data[i] = dA.data[i];

        Tensor dLN1_out(K, D);
        // Note: MHA's forward used mask_ptr; backward does not need mask passed —
        // it consults its own cached mask.
        mha_.backward(dA, dLN1_out);

        Tensor dX_from_ln(K, D);
        ln1_.backward(dLN1_out, dX_from_ln);
        for (int i = 0; i < K * D; ++i) dX_local.data[i] += dX_from_ln.data[i];

        for (int i = 0; i < K * D; ++i) dX.data[i] = dX_local.data[i];
    } else {
        // Post-norm.
        // y = LN2(a + FF(a))
        Tensor d_pre_ln2(K, D);
        ln2_.backward(dY, d_pre_ln2);
        // pre_ln2 = a + FF(a)  → dA = d_pre_ln2 (residual) + FF.bwd(d_pre_ln2)
        Tensor dA_ff(K, D);
        ff_.backward(d_pre_ln2, dA_ff);
        Tensor dA(K, D);
        for (int i = 0; i < K * D; ++i) dA.data[i] = d_pre_ln2.data[i] + dA_ff.data[i];

        // a = LN1(pre_ln1) where pre_ln1 = x + MHA(x)
        Tensor d_pre_ln1(K, D);
        ln1_.backward(dA, d_pre_ln1);

        // pre_ln1 = x + MHA(x) → dX (residual) + MHA.bwd
        Tensor dX_local(K, D);
        for (int i = 0; i < K * D; ++i) dX_local.data[i] = d_pre_ln1.data[i];
        Tensor dX_from_mha(K, D);
        mha_.backward(d_pre_ln1, dX_from_mha);
        for (int i = 0; i < K * D; ++i) dX_local.data[i] += dX_from_mha.data[i];

        for (int i = 0; i < K * D; ++i) dX.data[i] = dX_local.data[i];
    }
    (void)mask_ptr;
}

#ifdef BGA_HAS_CUDA

namespace {
// Tiny add kernel (Y[i] += X[i]). Using add_inplace_gpu from ops.
} // namespace

void TransformerBlock::forward(const gpu::GpuTensor& X, const float* mask_dev,
                               gpu::GpuTensor& Y) {
    assert(device_ == Device::GPU);
    const int K = X.rows;
    const int D = X.cols;
    assert(D == cfg_.dim);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    if (LN1_out_g_.rows != K || LN1_out_g_.cols != D) LN1_out_g_.resize(K, D);
    if (MHA_out_g_.rows != K || MHA_out_g_.cols != D) MHA_out_g_.resize(K, D);
    if (A_cache_g_.rows != K || A_cache_g_.cols != D) A_cache_g_.resize(K, D);
    if (LN2_out_g_.rows != K || LN2_out_g_.cols != D) LN2_out_g_.resize(K, D);
    if (FF_out_g_.rows != K || FF_out_g_.cols != D) FF_out_g_.resize(K, D);

    // Track masking state for backward (mha caches the mask itself).
    has_mask_ = (mask_dev != nullptr);

    if (cfg_.norm == NormPlacement::PreNorm) {
        // a = x + MHA(LN1(x))
        ln1_.forward(X, LN1_out_g_);
        mha_.forward(LN1_out_g_, mask_dev, MHA_out_g_);
        A_cache_g_ = X.clone();
        gpu::add_inplace_gpu(A_cache_g_, MHA_out_g_);

        // y = a + FF(LN2(a))
        ln2_.forward(A_cache_g_, LN2_out_g_);
        ff_.forward(LN2_out_g_, FF_out_g_);
        // Y = A + FF_out
        // We can't directly memcpy; clone A then add FF_out_g_.
        Y = A_cache_g_.clone();
        gpu::add_inplace_gpu(Y, FF_out_g_);
    } else {
        // post-norm: a = LN1(x + MHA(x))
        mha_.forward(X, mask_dev, MHA_out_g_);
        gpu::GpuTensor pre_ln1 = X.clone();
        gpu::add_inplace_gpu(pre_ln1, MHA_out_g_);
        ln1_.forward(pre_ln1, A_cache_g_);

        // y = LN2(a + FF(a))
        ff_.forward(A_cache_g_, FF_out_g_);
        gpu::GpuTensor pre_ln2 = A_cache_g_.clone();
        gpu::add_inplace_gpu(pre_ln2, FF_out_g_);
        ln2_.forward(pre_ln2, Y);
    }
}

void TransformerBlock::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    const int K = dY.rows;
    const int D = dY.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);

    if (cfg_.norm == NormPlacement::PreNorm) {
        // y = a + FF(LN2(a))
        gpu::GpuTensor dA = dY.clone();
        gpu::GpuTensor dLN2_out(K, D);
        ff_.backward(dY, dLN2_out);
        gpu::GpuTensor dA_from_ln(K, D);
        ln2_.backward(dLN2_out, dA_from_ln);
        gpu::add_inplace_gpu(dA, dA_from_ln);

        // a = x + MHA(LN1(x))
        // dMHA_out = dA → propagate through MHA + LN1. Compute that first
        // because the MHA backward consumes dA before we reuse it as dX.
        gpu::GpuTensor dLN1_out(K, D);
        mha_.backward(dA, dLN1_out);
        gpu::GpuTensor dX_from_ln(K, D);
        ln1_.backward(dLN1_out, dX_from_ln);
        // dX = dA (residual) + dX_from_ln. Move dA in to free a buffer.
        dX = std::move(dA);
        gpu::add_inplace_gpu(dX, dX_from_ln);
    } else {
        // Post-norm.
        gpu::GpuTensor d_pre_ln2(K, D);
        ln2_.backward(dY, d_pre_ln2);
        gpu::GpuTensor dA_ff(K, D);
        ff_.backward(d_pre_ln2, dA_ff);
        gpu::GpuTensor dA = d_pre_ln2.clone();
        gpu::add_inplace_gpu(dA, dA_ff);

        gpu::GpuTensor d_pre_ln1(K, D);
        ln1_.backward(dA, d_pre_ln1);

        // dX = d_pre_ln1 (residual) + MHA.bwd. MHA consumes d_pre_ln1 first.
        gpu::GpuTensor dX_from_mha(K, D);
        mha_.backward(d_pre_ln1, dX_from_mha);
        dX = std::move(d_pre_ln1);
        gpu::add_inplace_gpu(dX, dX_from_mha);
    }
}

void TransformerBlock::forward_inference_batched(
        const gpu::GpuTensor& X_RD, const float* mask_R_dev,
        gpu::GpuTensor& Y_RD, int B, int K) {
    const int D = cfg_.dim;
    const int R = B * K;
    if (Y_RD.rows != R || Y_RD.cols != D) Y_RD.resize(R, D);
    if (R == 0) return;

    if (cfg_.norm == NormPlacement::PreNorm) {
        // a = X + MHA(LN1(X), mask)
        gpu::GpuTensor LN_out(R, D);
        gpu::GpuTensor MHA_out(R, D);
        ln1_.forward_inference_batched(X_RD, LN_out);
        mha_.forward_inference_batched(LN_out, mask_R_dev, MHA_out, B, K);
        gpu::copy_d2d_gpu(X_RD, 0, Y_RD, 0, R * D);
        gpu::add_inplace_gpu(Y_RD, MHA_out);
        // Y += FF(LN2(Y))
        gpu::GpuTensor LN2_out(R, D);
        gpu::GpuTensor FF_out (R, D);
        ln2_.forward_inference_batched(Y_RD, LN2_out);
        ff_ .forward_inference_batched(LN2_out, FF_out);
        gpu::add_inplace_gpu(Y_RD, FF_out);
    } else {
        // post-norm: a = LN1(X + MHA(X, mask)); Y = LN2(a + FF(a))
        gpu::GpuTensor MHA_out(R, D);
        mha_.forward_inference_batched(X_RD, mask_R_dev, MHA_out, B, K);
        gpu::GpuTensor tmp(R, D);
        gpu::copy_d2d_gpu(X_RD, 0, tmp, 0, R * D);
        gpu::add_inplace_gpu(tmp, MHA_out);
        gpu::GpuTensor a(R, D);
        ln1_.forward_inference_batched(tmp, a);
        gpu::GpuTensor FF_out(R, D);
        ff_.forward_inference_batched(a, FF_out);
        gpu::copy_d2d_gpu(a, 0, tmp, 0, R * D);
        gpu::add_inplace_gpu(tmp, FF_out);
        ln2_.forward_inference_batched(tmp, Y_RD);
    }
}
#endif

void TransformerBlock::to(Device d) {
    if (d == device_) return;
    device_require_cuda("TransformerBlock");
    ln1_.to(d);
    ln2_.to(d);
    mha_.to(d);
    ff_.to(d);
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
