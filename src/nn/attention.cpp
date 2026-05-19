#include "brogameagent/nn/attention.h"
#include <brotensor/ops_cpu.h>

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#endif

#include <cassert>
#include <cmath>
#include <cstring>

namespace brogameagent::nn {

// Y = X @ W^T  where X:(N,D_in), W:(D_out,D_in), Y:(N,D_out)
static void matmul_xwT(const brotensor::Tensor& X, const brotensor::Tensor& W, brotensor::Tensor& Y) {
    const int N = X.rows, D_in = X.cols, D_out = W.rows;
    assert(W.cols == D_in);
    assert(Y.rows == N && Y.cols == D_out);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D_out; ++j) {
            float acc = 0.0f;
            const float* xr = X.ptr() + static_cast<size_t>(i) * D_in;
            const float* wr = W.ptr() + static_cast<size_t>(j) * D_in;
            for (int k = 0; k < D_in; ++k) acc += xr[k] * wr[k];
            Y(i, j) = acc;
        }
    }
}

void ScaledDotProductAttention::init(int n_slots, int dim, uint64_t& rng_state, int /*num_heads*/) {
    n_ = n_slots;
    d_ = dim;
    Wq_.resize(d_, d_); Wk_.resize(d_, d_); Wv_.resize(d_, d_); Wo_.resize(d_, d_);
    dWq_.resize(d_, d_); dWk_.resize(d_, d_); dWv_.resize(d_, d_); dWo_.resize(d_, d_);
    vWq_.resize(d_, d_); vWk_.resize(d_, d_); vWv_.resize(d_, d_); vWo_.resize(d_, d_);
    mWq_.resize(d_, d_); mWk_.resize(d_, d_); mWv_.resize(d_, d_); mWo_.resize(d_, d_);
    vAWq_.resize(d_, d_); vAWk_.resize(d_, d_); vAWv_.resize(d_, d_); vAWo_.resize(d_, d_);
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();
    brotensor::xavier_init_cpu(Wq_, rng_state);
    brotensor::xavier_init_cpu(Wk_, rng_state);
    brotensor::xavier_init_cpu(Wv_, rng_state);
    brotensor::xavier_init_cpu(Wo_, rng_state);

    X_cache_.resize(n_, d_);
    Q_.resize(n_, d_); K_.resize(n_, d_); V_.resize(n_, d_);
    Attn_.resize(n_, n_);
    Y_.resize(n_, d_);
    mask_cache_.assign(n_, 1);
}

void ScaledDotProductAttention::forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& O) {
    assert(X.rows == n_ && X.cols == d_);
    assert(O.rows == n_ && O.cols == d_);
    X_cache_ = X;
    mask_cache_.assign(n_, 1);
    if (mask) for (int i = 0; i < n_; ++i) mask_cache_[i] = mask[i] > 0.5f ? 1 : 0;

    // Q = X Wq^T, K = X Wk^T, V = X Wv^T.
    matmul_xwT(X, Wq_, Q_);
    matmul_xwT(X, Wk_, K_);
    matmul_xwT(X, Wv_, V_);

    // scores(i,j) = Q_i . K_j / sqrt(D)
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(d_));
    brotensor::Tensor scores(n_, n_);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d_; ++k) s += Q_(i, k) * K_(j, k);
            scores(i, j) = s * inv_sqrtd;
        }
    }

    // Row-softmax with column mask (invalid keys excluded). Invalid rows get
    // all-zero attention and zero outputs.
    brotensor::Tensor row_logits(n_, 1), row_probs(n_, 1);
    std::vector<float> col_mask(n_);
    for (int j = 0; j < n_; ++j) col_mask[j] = mask_cache_[j] ? 1.0f : 0.0f;
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) {
            for (int j = 0; j < n_; ++j) Attn_(i, j) = 0.0f;
            continue;
        }
        for (int j = 0; j < n_; ++j) row_logits[j] = scores(i, j);
        brotensor::softmax_forward_cpu(row_logits, row_probs, col_mask.data());
        for (int j = 0; j < n_; ++j) Attn_(i, j) = row_probs[j];
    }

    // Y = Attn @ V
    for (int i = 0; i < n_; ++i) {
        for (int k = 0; k < d_; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < n_; ++j) acc += Attn_(i, j) * V_(j, k);
            Y_(i, k) = acc;
        }
    }

    // O = Y @ Wo^T, zero invalid rows.
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) {
            for (int c = 0; c < d_; ++c) O(i, c) = 0.0f;
            continue;
        }
        for (int c = 0; c < d_; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < d_; ++k) acc += Y_(i, k) * Wo_(c, k);
            O(i, c) = acc;
        }
    }
}

void ScaledDotProductAttention::backward(const brotensor::Tensor& dO, brotensor::Tensor& dX) {
    assert(dO.rows == n_ && dO.cols == d_);
    assert(dX.rows == n_ && dX.cols == d_);
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(d_));

    // Zero dO on invalid rows (they had zero output).
    // dO_masked is effectively dO with invalid rows set to 0.
    brotensor::Tensor dY(n_, d_);   // grad wrt Y (before Wo)
    dY.zero();
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) continue;
        // O_i = Wo @ Y_i   (treating vectors)
        // dWo += dO_i outer Y_i ; dY_i = Wo^T @ dO_i
        for (int c = 0; c < d_; ++c) {
            const float g = dO(i, c);
            for (int k = 0; k < d_; ++k) {
                dWo_(c, k) += g * Y_(i, k);
                dY(i, k)   += Wo_(c, k) * g;
            }
        }
    }

    // Y = Attn @ V
    // dAttn(i,j) = sum_k dY(i,k) * V(j,k)
    // dV(j,k)   = sum_i Attn(i,j) * dY(i,k)
    brotensor::Tensor dAttn(n_, n_); dAttn.zero();
    brotensor::Tensor dV(n_, d_); dV.zero();
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d_; ++k) s += dY(i, k) * V_(j, k);
            dAttn(i, j) = s;
        }
    }
    for (int j = 0; j < n_; ++j) {
        for (int k = 0; k < d_; ++k) {
            float s = 0.0f;
            for (int i = 0; i < n_; ++i) s += Attn_(i, j) * dY(i, k);
            dV(j, k) = s;
        }
    }

    // dScores: row-wise softmax backward.
    brotensor::Tensor dScores(n_, n_); dScores.zero();
    brotensor::Tensor row_p(n_, 1), row_dp(n_, 1), row_dz(n_, 1);
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) continue;
        for (int j = 0; j < n_; ++j) { row_p[j] = Attn_(i, j); row_dp[j] = dAttn(i, j); }
        brotensor::softmax_backward_cpu(row_p, row_dp, row_dz);
        for (int j = 0; j < n_; ++j) {
            if (!mask_cache_[j]) { dScores(i, j) = 0.0f; continue; }
            dScores(i, j) = row_dz[j] * inv_sqrtd;
        }
    }

    // scores(i,j) = Q_i . K_j
    // dQ(i,k) = sum_j dScores(i,j) * K(j,k)
    // dK(j,k) = sum_i dScores(i,j) * Q(i,k)
    brotensor::Tensor dQ(n_, d_); dQ.zero();
    brotensor::Tensor dK(n_, d_); dK.zero();
    for (int i = 0; i < n_; ++i) {
        for (int k = 0; k < d_; ++k) {
            float s = 0.0f;
            for (int j = 0; j < n_; ++j) s += dScores(i, j) * K_(j, k);
            dQ(i, k) = s;
        }
    }
    for (int j = 0; j < n_; ++j) {
        for (int k = 0; k < d_; ++k) {
            float s = 0.0f;
            for (int i = 0; i < n_; ++i) s += dScores(i, j) * Q_(i, k);
            dK(j, k) = s;
        }
    }

    // Q = X Wq^T -> dWq += dQ^T @ X ; dX += dQ @ Wq  (likewise for K, V).
    dX.zero();
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < d_; ++j) {
            const float gq = dQ(i, j);
            const float gk = dK(i, j);
            const float gv = dV(i, j);
            for (int k = 0; k < d_; ++k) {
                dWq_(j, k) += gq * X_cache_(i, k);
                dWk_(j, k) += gk * X_cache_(i, k);
                dWv_(j, k) += gv * X_cache_(i, k);
                dX(i, k)   += gq * Wq_(j, k) + gk * Wk_(j, k) + gv * Wv_(j, k);
            }
        }
    }
}

#ifdef BROTENSOR_HAS_GPU
void ScaledDotProductAttention::forward(const brotensor::GpuTensor& X,
                                        const float* mask_dev,
                                        brotensor::GpuTensor& O) {
    assert(device_ == brotensor::Device::GPU);
    last_mask_dev_ = mask_dev;
    // Ensure cache mirrors are sized.
    if (X_cache_g_.rows != n_ || X_cache_g_.cols != d_) X_cache_g_.resize(n_, d_);
    if (Q_g_.rows != n_ || Q_g_.cols != d_) Q_g_.resize(n_, d_);
    if (K_g_.rows != n_ || K_g_.cols != d_) K_g_.resize(n_, d_);
    if (V_g_.rows != n_ || V_g_.cols != d_) V_g_.resize(n_, d_);
    if (Attn_g_.rows != n_ || Attn_g_.cols != n_) Attn_g_.resize(n_, n_);
    if (Y_g_.rows != n_ || Y_g_.cols != d_) Y_g_.resize(n_, d_);
    // X_cache_g_ holds the forward input by value-copy via clone-on-attention?
    // The kernel API takes X directly and reads it for backward via the
    // separate gX argument. We keep X_cache_g_ as a clone so the layer owns
    // a stable reference for backward (matches CPU semantics where X_cache_
    // shadows the caller's X).
    X_cache_g_ = X.clone();
    brotensor::attention_forward_gpu(X, Wq_g_, Wk_g_, Wv_g_, Wo_g_,
                               mask_dev,
                               Q_g_, K_g_, V_g_, Attn_g_, Y_g_, O);
}

void ScaledDotProductAttention::backward(const brotensor::GpuTensor& dO,
                                         brotensor::GpuTensor& dX) {
    assert(device_ == brotensor::Device::GPU);
    brotensor::attention_backward_gpu(dO, X_cache_g_, Q_g_, K_g_, V_g_, Attn_g_, Y_g_,
                                Wq_g_, Wk_g_, Wv_g_, Wo_g_,
                                last_mask_dev_,
                                dX,
                                dWq_g_, dWk_g_, dWv_g_, dWo_g_);
}
#endif

void ScaledDotProductAttention::to(brotensor::Device d) {
    if (d == device_) return;
    brotensor::device_require_gpu("ScaledDotProductAttention");
#ifdef BROTENSOR_HAS_GPU
    if (d == brotensor::Device::GPU) {
        brotensor::upload(Wq_, Wq_g_); brotensor::upload(Wk_, Wk_g_);
        brotensor::upload(Wv_, Wv_g_); brotensor::upload(Wo_, Wo_g_);
        brotensor::upload(dWq_, dWq_g_); brotensor::upload(dWk_, dWk_g_);
        brotensor::upload(dWv_, dWv_g_); brotensor::upload(dWo_, dWo_g_);
        brotensor::upload(vWq_, vWq_g_); brotensor::upload(vWk_, vWk_g_);
        brotensor::upload(vWv_, vWv_g_); brotensor::upload(vWo_, vWo_g_);
        brotensor::upload(mWq_, mWq_g_); brotensor::upload(mWk_, mWk_g_);
        brotensor::upload(mWv_, mWv_g_); brotensor::upload(mWo_, mWo_g_);
        brotensor::upload(vAWq_, vAWq_g_); brotensor::upload(vAWk_, vAWk_g_);
        brotensor::upload(vAWv_, vAWv_g_); brotensor::upload(vAWo_, vAWo_g_);
        device_ = brotensor::Device::GPU;
    } else {
        brotensor::download(Wq_g_, Wq_); brotensor::download(Wk_g_, Wk_);
        brotensor::download(Wv_g_, Wv_); brotensor::download(Wo_g_, Wo_);
        brotensor::download(dWq_g_, dWq_); brotensor::download(dWk_g_, dWk_);
        brotensor::download(dWv_g_, dWv_); brotensor::download(dWo_g_, dWo_);
        brotensor::download(vWq_g_, vWq_); brotensor::download(vWk_g_, vWk_);
        brotensor::download(vWv_g_, vWv_); brotensor::download(vWo_g_, vWo_);
        brotensor::download(mWq_g_, mWq_); brotensor::download(mWk_g_, mWk_);
        brotensor::download(mWv_g_, mWv_); brotensor::download(mWo_g_, mWo_);
        brotensor::download(vAWq_g_, vAWq_); brotensor::download(vAWk_g_, vAWk_);
        brotensor::download(vAWv_g_, vAWv_); brotensor::download(vAWo_g_, vAWo_);
        brotensor::cuda_sync();
        device_ = brotensor::Device::CPU;
    }
#endif
}

void ScaledDotProductAttention::zero_grad() {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        dWq_g_.zero(); dWk_g_.zero(); dWv_g_.zero(); dWo_g_.zero();
        return;
    }
#endif
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
}

static void sgd_mat(brotensor::Tensor& W, brotensor::Tensor& vW, const brotensor::Tensor& dW, float lr, float momentum) {
    const int n = W.size();
    float* w = W.ptr(); float* v = vW.ptr(); const float* g = dW.ptr();
    for (int i = 0; i < n; ++i) {
        v[i] = momentum * v[i] + g[i];
        w[i] -= lr * v[i];
    }
}

void ScaledDotProductAttention::sgd_step(float lr, float momentum) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        brotensor::sgd_step_gpu(Wq_g_, dWq_g_, vWq_g_, lr, momentum);
        brotensor::sgd_step_gpu(Wk_g_, dWk_g_, vWk_g_, lr, momentum);
        brotensor::sgd_step_gpu(Wv_g_, dWv_g_, vWv_g_, lr, momentum);
        brotensor::sgd_step_gpu(Wo_g_, dWo_g_, vWo_g_, lr, momentum);
        return;
    }
#endif
    sgd_mat(Wq_, vWq_, dWq_, lr, momentum);
    sgd_mat(Wk_, vWk_, dWk_, lr, momentum);
    sgd_mat(Wv_, vWv_, dWv_, lr, momentum);
    sgd_mat(Wo_, vWo_, dWo_, lr, momentum);
}

void ScaledDotProductAttention::adam_step(float lr, float beta1, float beta2,
                                          float eps, int step) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        brotensor::adam_step_gpu(Wq_g_, dWq_g_, mWq_g_, vAWq_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(Wk_g_, dWk_g_, mWk_g_, vAWk_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(Wv_g_, dWv_g_, mWv_g_, vAWv_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(Wo_g_, dWo_g_, mWo_g_, vAWo_g_, lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(Wq_, dWq_, mWq_, vAWq_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wk_, dWk_, mWk_, vAWk_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wv_, dWv_, mWv_, vAWv_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wo_, dWo_, mWo_, vAWo_, lr, beta1, beta2, eps, step);
}

void ScaledDotProductAttention::save_to(std::vector<uint8_t>& out) const {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        auto* self = const_cast<ScaledDotProductAttention*>(this);
        brotensor::download(Wq_g_, self->Wq_); brotensor::download(Wk_g_, self->Wk_);
        brotensor::download(Wv_g_, self->Wv_); brotensor::download(Wo_g_, self->Wo_);
        brotensor::cuda_sync();
    }
#endif
    tensor_write(Wq_, out);
    tensor_write(Wk_, out);
    tensor_write(Wv_, out);
    tensor_write(Wo_, out);
}

void ScaledDotProductAttention::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(Wq_, data, offset, size);
    tensor_read(Wk_, data, offset, size);
    tensor_read(Wv_, data, offset, size);
    tensor_read(Wo_, data, offset, size);
    d_ = Wq_.rows;
    dWq_.resize(d_, d_); dWk_.resize(d_, d_); dWv_.resize(d_, d_); dWo_.resize(d_, d_);
    vWq_.resize(d_, d_); vWk_.resize(d_, d_); vWv_.resize(d_, d_); vWo_.resize(d_, d_);
    mWq_.resize(d_, d_); mWk_.resize(d_, d_); mWv_.resize(d_, d_); mWo_.resize(d_, d_);
    vAWq_.resize(d_, d_); vAWk_.resize(d_, d_); vAWv_.resize(d_, d_); vAWo_.resize(d_, d_);
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        brotensor::upload(Wq_, Wq_g_); brotensor::upload(Wk_, Wk_g_);
        brotensor::upload(Wv_, Wv_g_); brotensor::upload(Wo_, Wo_g_);
        brotensor::upload(dWq_, dWq_g_); brotensor::upload(dWk_, dWk_g_);
        brotensor::upload(dWv_, dWv_g_); brotensor::upload(dWo_, dWo_g_);
        brotensor::upload(vWq_, vWq_g_); brotensor::upload(vWk_, vWk_g_);
        brotensor::upload(vWv_, vWv_g_); brotensor::upload(vWo_, vWo_g_);
        brotensor::upload(mWq_, mWq_g_); brotensor::upload(mWk_, mWk_g_);
        brotensor::upload(mWv_, mWv_g_); brotensor::upload(mWo_, mWo_g_);
        brotensor::upload(vAWq_, vAWq_g_); brotensor::upload(vAWk_, vAWk_g_);
        brotensor::upload(vAWv_, vAWv_g_); brotensor::upload(vAWo_, vAWo_g_);
    }
#endif
}

} // namespace brogameagent::nn
