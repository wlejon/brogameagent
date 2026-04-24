#include "brogameagent/nn/attention.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace brogameagent::nn {

// Y = X @ W^T  where X:(N,D_in), W:(D_out,D_in), Y:(N,D_out)
static void matmul_xwT(const Tensor& X, const Tensor& W, Tensor& Y) {
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
    xavier_init(Wq_, rng_state);
    xavier_init(Wk_, rng_state);
    xavier_init(Wv_, rng_state);
    xavier_init(Wo_, rng_state);

    X_cache_.resize(n_, d_);
    Q_.resize(n_, d_); K_.resize(n_, d_); V_.resize(n_, d_);
    Attn_.resize(n_, n_);
    Y_.resize(n_, d_);
    mask_cache_.assign(n_, 1);
}

void ScaledDotProductAttention::forward(const Tensor& X, const float* mask, Tensor& O) {
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
    Tensor scores(n_, n_);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d_; ++k) s += Q_(i, k) * K_(j, k);
            scores(i, j) = s * inv_sqrtd;
        }
    }

    // Row-softmax with column mask (invalid keys excluded). Invalid rows get
    // all-zero attention and zero outputs.
    Tensor row_logits(n_, 1), row_probs(n_, 1);
    std::vector<float> col_mask(n_);
    for (int j = 0; j < n_; ++j) col_mask[j] = mask_cache_[j] ? 1.0f : 0.0f;
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) {
            for (int j = 0; j < n_; ++j) Attn_(i, j) = 0.0f;
            continue;
        }
        for (int j = 0; j < n_; ++j) row_logits[j] = scores(i, j);
        softmax_forward(row_logits, row_probs, col_mask.data());
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

void ScaledDotProductAttention::backward(const Tensor& dO, Tensor& dX) {
    assert(dO.rows == n_ && dO.cols == d_);
    assert(dX.rows == n_ && dX.cols == d_);
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(d_));

    // Zero dO on invalid rows (they had zero output).
    // dO_masked is effectively dO with invalid rows set to 0.
    Tensor dY(n_, d_);   // grad wrt Y (before Wo)
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
    Tensor dAttn(n_, n_); dAttn.zero();
    Tensor dV(n_, d_); dV.zero();
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
    Tensor dScores(n_, n_); dScores.zero();
    Tensor row_p(n_, 1), row_dp(n_, 1), row_dz(n_, 1);
    for (int i = 0; i < n_; ++i) {
        if (!mask_cache_[i]) continue;
        for (int j = 0; j < n_; ++j) { row_p[j] = Attn_(i, j); row_dp[j] = dAttn(i, j); }
        softmax_backward(row_p, row_dp, row_dz);
        for (int j = 0; j < n_; ++j) {
            if (!mask_cache_[j]) { dScores(i, j) = 0.0f; continue; }
            dScores(i, j) = row_dz[j] * inv_sqrtd;
        }
    }

    // scores(i,j) = Q_i . K_j
    // dQ(i,k) = sum_j dScores(i,j) * K(j,k)
    // dK(j,k) = sum_i dScores(i,j) * Q(i,k)
    Tensor dQ(n_, d_); dQ.zero();
    Tensor dK(n_, d_); dK.zero();
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

void ScaledDotProductAttention::zero_grad() {
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
}

static void sgd_mat(Tensor& W, Tensor& vW, const Tensor& dW, float lr, float momentum) {
    const int n = W.size();
    float* w = W.ptr(); float* v = vW.ptr(); const float* g = dW.ptr();
    for (int i = 0; i < n; ++i) {
        v[i] = momentum * v[i] + g[i];
        w[i] -= lr * v[i];
    }
}

void ScaledDotProductAttention::sgd_step(float lr, float momentum) {
    sgd_mat(Wq_, vWq_, dWq_, lr, momentum);
    sgd_mat(Wk_, vWk_, dWk_, lr, momentum);
    sgd_mat(Wv_, vWv_, dWv_, lr, momentum);
    sgd_mat(Wo_, vWo_, dWo_, lr, momentum);
}

void ScaledDotProductAttention::save_to(std::vector<uint8_t>& out) const {
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
}

} // namespace brogameagent::nn
