#include "brogameagent/nn/multi_head_attention.h"
#include "brogameagent/nn/ops.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

#include <cassert>
#include <cmath>
#include <cstring>

namespace brogameagent::nn {

// Per-head matmul Y(i, j) = sum_k X(i, k) * W(j_off + j, k), where W has
// per-head row slice [j_off, j_off + Dh) of a (D, D) matrix.
//   X:  (K, D)
//   W:  (D, D)
//   Y:  (K, Dh)
static void matmul_head_xwT(const Tensor& X, const Tensor& W,
                            int row_off, Tensor& Y) {
    const int K = X.rows, D = X.cols, Dh = Y.cols;
    assert(W.rows == D && W.cols == D);
    assert(Y.rows == K);
    for (int i = 0; i < K; ++i) {
        const float* xr = X.ptr() + static_cast<size_t>(i) * D;
        for (int j = 0; j < Dh; ++j) {
            const float* wr = W.ptr() + static_cast<size_t>(row_off + j) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
            Y(i, j) = acc;
        }
    }
}

void MultiHeadAttention::init(int n_slots, int dim, int num_heads,
                              uint64_t& rng_state) {
    assert(num_heads >= 1);
    assert(dim % num_heads == 0 && "dim must be divisible by num_heads");
    n_  = n_slots;
    d_  = dim;
    h_  = num_heads;
    dh_ = d_ / h_;

    Wq_.resize(d_, d_);  Wk_.resize(d_, d_);  Wv_.resize(d_, d_);  Wo_.resize(d_, d_);
    dWq_.resize(d_, d_); dWk_.resize(d_, d_); dWv_.resize(d_, d_); dWo_.resize(d_, d_);
    vWq_.resize(d_, d_); vWk_.resize(d_, d_); vWv_.resize(d_, d_); vWo_.resize(d_, d_);
    mWq_.resize(d_, d_); mWk_.resize(d_, d_); mWv_.resize(d_, d_); mWo_.resize(d_, d_);
    vAWq_.resize(d_, d_); vAWk_.resize(d_, d_); vAWv_.resize(d_, d_); vAWo_.resize(d_, d_);
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();

    xavier_init(Wq_, rng_state);
    xavier_init(Wk_, rng_state);
    xavier_init(Wv_, rng_state);
    xavier_init(Wo_, rng_state);

    X_cache_.resize(n_, d_);
    Qh_.assign(h_, Tensor(n_, dh_));
    Kh_.assign(h_, Tensor(n_, dh_));
    Vh_.assign(h_, Tensor(n_, dh_));
    Attnh_.assign(h_, Tensor(n_, n_));
    Yconcat_.resize(n_, d_);
    mask_cache_.assign(n_, 1);
}

void MultiHeadAttention::forward(const Tensor& X, const float* mask, Tensor& O) {
    const int K  = X.rows;
    const int D  = X.cols;
    assert(D == d_);
    n_ = K;
    if (X_cache_.rows != K || X_cache_.cols != D) X_cache_.resize(K, D);
    for (auto& Q : Qh_)    if (Q.rows != K || Q.cols != dh_) Q.resize(K, dh_);
    for (auto& Kk : Kh_)   if (Kk.rows != K || Kk.cols != dh_) Kk.resize(K, dh_);
    for (auto& V : Vh_)    if (V.rows != K || V.cols != dh_) V.resize(K, dh_);
    for (auto& A : Attnh_) if (A.rows != K || A.cols != K)    A.resize(K, K);
    if (Yconcat_.rows != K || Yconcat_.cols != D) Yconcat_.resize(K, D);
    if (O.rows != K || O.cols != D) O.resize(K, D);

    X_cache_ = X;
    mask_cache_.assign(K, 1);
    if (mask) for (int i = 0; i < K; ++i) mask_cache_[i] = mask[i] > 0.5f ? 1 : 0;

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh_));

    std::vector<float> col_mask(K);
    for (int j = 0; j < K; ++j) col_mask[j] = mask_cache_[j] ? 1.0f : 0.0f;

    // Per-head Q/K/V projections, scores, softmax, attn@V.
    for (int hh = 0; hh < h_; ++hh) {
        const int row_off = hh * dh_;
        matmul_head_xwT(X, Wq_, row_off, Qh_[hh]);
        matmul_head_xwT(X, Wk_, row_off, Kh_[hh]);
        matmul_head_xwT(X, Wv_, row_off, Vh_[hh]);

        // scores (K, K)
        Tensor scores(K, K);
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                float s = 0.0f;
                for (int k = 0; k < dh_; ++k) s += Qh_[hh](i, k) * Kh_[hh](j, k);
                scores(i, j) = s * inv_sqrtdh;
            }
        }

        // Row-softmax with column mask (invalid keys excluded). Invalid rows
        // get all-zero attention.
        Tensor row_logits(K, 1), row_probs(K, 1);
        for (int i = 0; i < K; ++i) {
            if (!mask_cache_[i]) {
                for (int j = 0; j < K; ++j) Attnh_[hh](i, j) = 0.0f;
                continue;
            }
            for (int j = 0; j < K; ++j) row_logits[j] = scores(i, j);
            softmax_forward(row_logits, row_probs, col_mask.data());
            for (int j = 0; j < K; ++j) Attnh_[hh](i, j) = row_probs[j];
        }

        // Yh = Attn @ V → write into Yconcat columns [row_off, row_off+dh).
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < dh_; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) acc += Attnh_[hh](i, j) * Vh_[hh](j, k);
                Yconcat_(i, row_off + k) = acc;
            }
        }
    }

    // O = Yconcat @ Wo^T, zero invalid query rows.
    for (int i = 0; i < K; ++i) {
        if (!mask_cache_[i]) {
            for (int c = 0; c < D; ++c) O(i, c) = 0.0f;
            continue;
        }
        for (int c = 0; c < D; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += Yconcat_(i, k) * Wo_(c, k);
            O(i, c) = acc;
        }
    }
}

void MultiHeadAttention::backward(const Tensor& dO, Tensor& dX) {
    const int K  = X_cache_.rows;
    const int D  = X_cache_.cols;
    assert(dO.rows == K && dO.cols == D);
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    dX.zero();

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh_));

    // dWo (D, D) and dY_concat (K, D).  O = Y_concat @ Wo^T (with row mask).
    // dWo(c, k) += sum_i mask_i * dO(i, c) * Y_concat(i, k)
    // dY_concat(i, k) = mask_i * sum_c Wo(c, k) * dO(i, c)
    Tensor dYconcat(K, D); dYconcat.zero();
    for (int i = 0; i < K; ++i) {
        if (!mask_cache_[i]) continue;
        for (int c = 0; c < D; ++c) {
            const float g = dO(i, c);
            for (int k = 0; k < D; ++k) {
                dWo_(c, k)       += g * Yconcat_(i, k);
                dYconcat(i, k)   += Wo_(c, k) * g;
            }
        }
    }

    // Per-head backward.
    for (int hh = 0; hh < h_; ++hh) {
        const int row_off = hh * dh_;
        // Slice out dY for this head: (K, dh)
        Tensor dYh(K, dh_);
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < dh_; ++k) dYh(i, k) = dYconcat(i, row_off + k);
        }

        // Yh = Attnh @ Vh
        // dAttn(i,j) = sum_k dYh(i,k) * Vh(j,k)
        // dVh(j,k)   = sum_i Attnh(i,j) * dYh(i,k)
        Tensor dAttn(K, K); dAttn.zero();
        Tensor dVh(K, dh_); dVh.zero();
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                float s = 0.0f;
                for (int k = 0; k < dh_; ++k) s += dYh(i, k) * Vh_[hh](j, k);
                dAttn(i, j) = s;
            }
        }
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < dh_; ++k) {
                float s = 0.0f;
                for (int i = 0; i < K; ++i) s += Attnh_[hh](i, j) * dYh(i, k);
                dVh(j, k) = s;
            }
        }

        // dScores via per-row softmax backward, scaled by inv_sqrtdh.
        Tensor dScores(K, K); dScores.zero();
        Tensor row_p(K, 1), row_dp(K, 1), row_dz(K, 1);
        for (int i = 0; i < K; ++i) {
            if (!mask_cache_[i]) continue;
            for (int j = 0; j < K; ++j) { row_p[j] = Attnh_[hh](i, j); row_dp[j] = dAttn(i, j); }
            softmax_backward(row_p, row_dp, row_dz);
            for (int j = 0; j < K; ++j) {
                if (!mask_cache_[j]) { dScores(i, j) = 0.0f; continue; }
                dScores(i, j) = row_dz[j] * inv_sqrtdh;
            }
        }

        // dQh(i,k) = sum_j dScores(i,j) * Kh(j,k)
        // dKh(j,k) = sum_i dScores(i,j) * Qh(i,k)
        Tensor dQh(K, dh_); dQh.zero();
        Tensor dKh(K, dh_); dKh.zero();
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < dh_; ++k) {
                float s = 0.0f;
                for (int j = 0; j < K; ++j) s += dScores(i, j) * Kh_[hh](j, k);
                dQh(i, k) = s;
            }
        }
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < dh_; ++k) {
                float s = 0.0f;
                for (int i = 0; i < K; ++i) s += dScores(i, j) * Qh_[hh](i, k);
                dKh(j, k) = s;
            }
        }

        // Project gradients back through the per-head input projections.
        //   Q_h = X @ Wq[row_off..]^T   (rows row_off+j of Wq)
        //   dWq(row_off+j, k) += sum_i dQh(i, j) * X(i, k)
        //   dX(i, k)         += sum_j dQh(i, j) * Wq(row_off+j, k)
        for (int j = 0; j < dh_; ++j) {
            const int wrow = row_off + j;
            for (int i = 0; i < K; ++i) {
                const float gq = dQh(i, j);
                const float gk = dKh(i, j);
                const float gv = dVh(i, j);
                for (int k = 0; k < D; ++k) {
                    const float xv = X_cache_(i, k);
                    dWq_(wrow, k) += gq * xv;
                    dWk_(wrow, k) += gk * xv;
                    dWv_(wrow, k) += gv * xv;
                    dX(i, k) += gq * Wq_(wrow, k)
                              + gk * Wk_(wrow, k)
                              + gv * Wv_(wrow, k);
                }
            }
        }
    }
}

#ifdef BGA_HAS_CUDA
void MultiHeadAttention::forward(const gpu::GpuTensor& X,
                                 const float* mask_dev,
                                 gpu::GpuTensor& O) {
    assert(device_ == Device::GPU);
    const int K = X.rows;
    const int D = X.cols;
    assert(D == d_);
    n_ = K;
    last_mask_dev_ = mask_dev;
    if (X_cache_g_.rows != K || X_cache_g_.cols != D) X_cache_g_.resize(K, D);
    if (Qh_g_.rows != h_ * K || Qh_g_.cols != dh_) Qh_g_.resize(h_ * K, dh_);
    if (Kh_g_.rows != h_ * K || Kh_g_.cols != dh_) Kh_g_.resize(h_ * K, dh_);
    if (Vh_g_.rows != h_ * K || Vh_g_.cols != dh_) Vh_g_.resize(h_ * K, dh_);
    if (Attnh_g_.rows != h_ * K || Attnh_g_.cols != K) Attnh_g_.resize(h_ * K, K);
    if (Yconcat_g_.rows != K || Yconcat_g_.cols != D) Yconcat_g_.resize(K, D);
    // Clone X for backward (matches CPU semantics where X_cache_ shadows input).
    X_cache_g_ = X.clone();
    gpu::mha_forward_gpu(X, Wq_g_, Wk_g_, Wv_g_, Wo_g_,
                         mask_dev, h_,
                         Qh_g_, Kh_g_, Vh_g_, Attnh_g_, Yconcat_g_, O);
}

void MultiHeadAttention::backward(const gpu::GpuTensor& dO, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    gpu::mha_backward_gpu(dO, X_cache_g_, Qh_g_, Kh_g_, Vh_g_, Attnh_g_, Yconcat_g_,
                          Wq_g_, Wk_g_, Wv_g_, Wo_g_,
                          last_mask_dev_, h_,
                          dX, dWq_g_, dWk_g_, dWv_g_, dWo_g_);
}
#endif

void MultiHeadAttention::to(Device d) {
    if (d == device_) return;
    device_require_cuda("MultiHeadAttention");
#ifdef BGA_HAS_CUDA
    if (d == Device::GPU) {
        gpu::upload(Wq_, Wq_g_); gpu::upload(Wk_, Wk_g_);
        gpu::upload(Wv_, Wv_g_); gpu::upload(Wo_, Wo_g_);
        gpu::upload(dWq_, dWq_g_); gpu::upload(dWk_, dWk_g_);
        gpu::upload(dWv_, dWv_g_); gpu::upload(dWo_, dWo_g_);
        gpu::upload(vWq_, vWq_g_); gpu::upload(vWk_, vWk_g_);
        gpu::upload(vWv_, vWv_g_); gpu::upload(vWo_, vWo_g_);
        gpu::upload(mWq_, mWq_g_); gpu::upload(mWk_, mWk_g_);
        gpu::upload(mWv_, mWv_g_); gpu::upload(mWo_, mWo_g_);
        gpu::upload(vAWq_, vAWq_g_); gpu::upload(vAWk_, vAWk_g_);
        gpu::upload(vAWv_, vAWv_g_); gpu::upload(vAWo_, vAWo_g_);
        device_ = Device::GPU;
    } else {
        gpu::download(Wq_g_, Wq_); gpu::download(Wk_g_, Wk_);
        gpu::download(Wv_g_, Wv_); gpu::download(Wo_g_, Wo_);
        gpu::download(dWq_g_, dWq_); gpu::download(dWk_g_, dWk_);
        gpu::download(dWv_g_, dWv_); gpu::download(dWo_g_, dWo_);
        gpu::download(vWq_g_, vWq_); gpu::download(vWk_g_, vWk_);
        gpu::download(vWv_g_, vWv_); gpu::download(vWo_g_, vWo_);
        gpu::download(mWq_g_, mWq_); gpu::download(mWk_g_, mWk_);
        gpu::download(mWv_g_, mWv_); gpu::download(mWo_g_, mWo_);
        gpu::download(vAWq_g_, vAWq_); gpu::download(vAWk_g_, vAWk_);
        gpu::download(vAWv_g_, vAWv_); gpu::download(vAWo_g_, vAWo_);
        gpu::cuda_sync();
        device_ = Device::CPU;
    }
#endif
}

void MultiHeadAttention::zero_grad() {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        dWq_g_.zero(); dWk_g_.zero(); dWv_g_.zero(); dWo_g_.zero();
        return;
    }
#endif
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
}

static void sgd_mat_(Tensor& W, Tensor& vW, const Tensor& dW, float lr, float momentum) {
    const int n = W.size();
    float* w = W.ptr(); float* v = vW.ptr(); const float* g = dW.ptr();
    for (int i = 0; i < n; ++i) {
        v[i] = momentum * v[i] + g[i];
        w[i] -= lr * v[i];
    }
}

void MultiHeadAttention::sgd_step(float lr, float momentum) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::sgd_step_gpu(Wq_g_, dWq_g_, vWq_g_, lr, momentum);
        gpu::sgd_step_gpu(Wk_g_, dWk_g_, vWk_g_, lr, momentum);
        gpu::sgd_step_gpu(Wv_g_, dWv_g_, vWv_g_, lr, momentum);
        gpu::sgd_step_gpu(Wo_g_, dWo_g_, vWo_g_, lr, momentum);
        return;
    }
#endif
    sgd_mat_(Wq_, vWq_, dWq_, lr, momentum);
    sgd_mat_(Wk_, vWk_, dWk_, lr, momentum);
    sgd_mat_(Wv_, vWv_, dWv_, lr, momentum);
    sgd_mat_(Wo_, vWo_, dWo_, lr, momentum);
}

void MultiHeadAttention::adam_step(float lr, float beta1, float beta2,
                                   float eps, int step) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::adam_step_gpu(Wq_g_, dWq_g_, mWq_g_, vAWq_g_, lr, beta1, beta2, eps, step);
        gpu::adam_step_gpu(Wk_g_, dWk_g_, mWk_g_, vAWk_g_, lr, beta1, beta2, eps, step);
        gpu::adam_step_gpu(Wv_g_, dWv_g_, mWv_g_, vAWv_g_, lr, beta1, beta2, eps, step);
        gpu::adam_step_gpu(Wo_g_, dWo_g_, mWo_g_, vAWo_g_, lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(Wq_, dWq_, mWq_, vAWq_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wk_, dWk_, mWk_, vAWk_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wv_, dWv_, mWv_, vAWv_, lr, beta1, beta2, eps, step);
    adam_step_cpu(Wo_, dWo_, mWo_, vAWo_, lr, beta1, beta2, eps, step);
}

void MultiHeadAttention::save_to(std::vector<uint8_t>& out) const {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        auto* self = const_cast<MultiHeadAttention*>(this);
        gpu::download(Wq_g_, self->Wq_); gpu::download(Wk_g_, self->Wk_);
        gpu::download(Wv_g_, self->Wv_); gpu::download(Wo_g_, self->Wo_);
        gpu::cuda_sync();
    }
#endif
    // Mirrors ScaledDotProductAttention: just the four weight matrices.
    // num_heads is recovered from the caller's init() before load_from().
    tensor_write(Wq_, out);
    tensor_write(Wk_, out);
    tensor_write(Wv_, out);
    tensor_write(Wo_, out);
}

void MultiHeadAttention::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(Wq_, data, offset, size);
    tensor_read(Wk_, data, offset, size);
    tensor_read(Wv_, data, offset, size);
    tensor_read(Wo_, data, offset, size);
    d_  = Wq_.rows;
    if (h_ < 1) h_ = 1;
    dh_ = d_ / h_;
    dWq_.resize(d_, d_); dWk_.resize(d_, d_); dWv_.resize(d_, d_); dWo_.resize(d_, d_);
    vWq_.resize(d_, d_); vWk_.resize(d_, d_); vWv_.resize(d_, d_); vWo_.resize(d_, d_);
    mWq_.resize(d_, d_); mWk_.resize(d_, d_); mWv_.resize(d_, d_); mWo_.resize(d_, d_);
    vAWq_.resize(d_, d_); vAWk_.resize(d_, d_); vAWv_.resize(d_, d_); vAWo_.resize(d_, d_);
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();
    Qh_.assign(h_, Tensor(n_, dh_));
    Kh_.assign(h_, Tensor(n_, dh_));
    Vh_.assign(h_, Tensor(n_, dh_));
    Attnh_.assign(h_, Tensor(n_, n_));
    Yconcat_.resize(n_, d_);
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::upload(Wq_, Wq_g_); gpu::upload(Wk_, Wk_g_);
        gpu::upload(Wv_, Wv_g_); gpu::upload(Wo_, Wo_g_);
        gpu::upload(dWq_, dWq_g_); gpu::upload(dWk_, dWk_g_);
        gpu::upload(dWv_, dWv_g_); gpu::upload(dWo_, dWo_g_);
        gpu::upload(vWq_, vWq_g_); gpu::upload(vWk_, vWk_g_);
        gpu::upload(vWv_, vWv_g_); gpu::upload(vWo_, vWo_g_);
        gpu::upload(mWq_, mWq_g_); gpu::upload(mWk_, mWk_g_);
        gpu::upload(mWv_, mWv_g_); gpu::upload(mWo_, mWo_g_);
        gpu::upload(vAWq_, vAWq_g_); gpu::upload(vAWk_, vAWk_g_);
        gpu::upload(vAWv_, vAWv_g_); gpu::upload(vAWo_, vAWo_g_);
    }
#endif
}

} // namespace brogameagent::nn
