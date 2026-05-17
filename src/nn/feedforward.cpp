#include "brogameagent/nn/feedforward.h"
#include "brogameagent/nn/ops.h"

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brogameagent/nn/gpu_glue.h>
#endif

#include <cassert>

namespace brogameagent::nn {

void FeedForward::init(int dim, int d_ff, uint64_t& rng_state) {
    d_  = dim;
    df_ = d_ff;
    W1_.resize(df_, d_); b1_.resize(df_, 1);
    W2_.resize(d_, df_); b2_.resize(d_, 1);
    dW1_.resize(df_, d_); dB1_.resize(df_, 1);
    dW2_.resize(d_, df_); dB2_.resize(d_, 1);
    vW1_.resize(df_, d_); vB1_.resize(df_, 1);
    vW2_.resize(d_, df_); vB2_.resize(d_, 1);
    mW1_.resize(df_, d_); mB1_.resize(df_, 1);
    mW2_.resize(d_, df_); mB2_.resize(d_, 1);
    vAW1_.resize(df_, d_); vAB1_.resize(df_, 1);
    vAW2_.resize(d_, df_); vAB2_.resize(d_, 1);
    xavier_init(W1_, rng_state);
    xavier_init(W2_, rng_state);
    b1_.zero(); b2_.zero();
    dW1_.zero(); dB1_.zero(); dW2_.zero(); dB2_.zero();
    vW1_.zero(); vB1_.zero(); vW2_.zero(); vB2_.zero();
    mW1_.zero(); mB1_.zero(); mW2_.zero(); mB2_.zero();
    vAW1_.zero(); vAB1_.zero(); vAW2_.zero(); vAB2_.zero();
}

void FeedForward::forward(const Tensor& X, Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == d_);
    if (X_cache_.rows != K || X_cache_.cols != D) X_cache_.resize(K, D);
    if (H_pre_.rows != K || H_pre_.cols != df_) H_pre_.resize(K, df_);
    if (H_post_.rows != K || H_post_.cols != df_) H_post_.resize(K, df_);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    X_cache_ = X;

    // Layer 1: H_pre = X @ W1^T + b1   (W1 is (df, D))
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < df_; ++j) {
            float acc = b1_[j];
            const float* xr = X.ptr() + static_cast<size_t>(i) * D;
            const float* wr = W1_.ptr() + static_cast<size_t>(j) * D;
            for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
            H_pre_(i, j) = acc;
            H_post_(i, j) = acc > 0.0f ? acc : 0.0f;
        }
    }

    // Layer 2: Y = H_post @ W2^T + b2  (W2 is (D, df))
    for (int i = 0; i < K; ++i) {
        for (int c = 0; c < D; ++c) {
            float acc = b2_[c];
            const float* hr = H_post_.ptr() + static_cast<size_t>(i) * df_;
            const float* wr = W2_.ptr() + static_cast<size_t>(c) * df_;
            for (int k = 0; k < df_; ++k) acc += hr[k] * wr[k];
            Y(i, c) = acc;
        }
    }
}

void FeedForward::backward(const Tensor& dY, Tensor& dX) {
    const int K = X_cache_.rows;
    const int D = X_cache_.cols;
    assert(dY.rows == K && dY.cols == D);
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    dX.zero();

    // Layer 2 backward: Y = H_post @ W2^T + b2
    //   dW2(c, k) += sum_i dY(i, c) * H_post(i, k)
    //   dB2(c)    += sum_i dY(i, c)
    //   dH_post(i, k) = sum_c dY(i, c) * W2(c, k)
    Tensor dHpost(K, df_); dHpost.zero();
    for (int i = 0; i < K; ++i) {
        for (int c = 0; c < D; ++c) {
            const float g = dY(i, c);
            dB2_[c] += g;
            for (int k = 0; k < df_; ++k) {
                dW2_(c, k)  += g * H_post_(i, k);
                dHpost(i, k) += W2_(c, k) * g;
            }
        }
    }

    // ReLU backward: dH_pre(i, k) = (H_pre > 0) * dH_post(i, k)
    Tensor dHpre(K, df_);
    for (int i = 0; i < K; ++i) {
        for (int k = 0; k < df_; ++k) {
            dHpre(i, k) = H_pre_(i, k) > 0.0f ? dHpost(i, k) : 0.0f;
        }
    }

    // Layer 1 backward: H_pre = X @ W1^T + b1
    //   dW1(j, k) += sum_i dH_pre(i, j) * X(i, k)
    //   dB1(j)    += sum_i dH_pre(i, j)
    //   dX(i, k)  += sum_j dH_pre(i, j) * W1(j, k)
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < df_; ++j) {
            const float g = dHpre(i, j);
            dB1_[j] += g;
            for (int k = 0; k < D; ++k) {
                dW1_(j, k) += g * X_cache_(i, k);
                dX(i, k)   += W1_(j, k) * g;
            }
        }
    }
}

#ifdef BROTENSOR_HAS_GPU
void FeedForward::forward(const brotensor::GpuTensor& X, brotensor::GpuTensor& Y) {
    assert(device_ == Device::GPU);
    const int K = X.rows;
    const int D = X.cols;
    assert(D == d_);
    if (X_cache_g_.rows != K || X_cache_g_.cols != D) X_cache_g_.resize(K, D);
    if (H_pre_g_.rows != K || H_pre_g_.cols != df_) H_pre_g_.resize(K, df_);
    if (H_post_g_.rows != K || H_post_g_.cols != df_) H_post_g_.resize(K, df_);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);
    // Cache X for backward.
    X_cache_g_ = X.clone();

    // Per-row linear forward (loop over K rows, view each as a (D,1)/(df,1)).
    for (int i = 0; i < K; ++i) {
        brotensor::GpuTensor x_row = brotensor::GpuTensor::view(
            X.data + static_cast<size_t>(i) * D, D, 1);
        brotensor::GpuTensor h_row = brotensor::GpuTensor::view(
            H_pre_g_.data + static_cast<size_t>(i) * df_, df_, 1);
        brotensor::linear_forward_gpu(W1_g_, b1_g_, x_row, h_row);
    }
    // Flat ReLU over (K * df) entries.
    brotensor::relu_forward_gpu(H_pre_g_, H_post_g_);

    // Layer 2 per-row.
    for (int i = 0; i < K; ++i) {
        brotensor::GpuTensor h_row = brotensor::GpuTensor::view(
            H_post_g_.data + static_cast<size_t>(i) * df_, df_, 1);
        brotensor::GpuTensor y_row = brotensor::GpuTensor::view(
            Y.data + static_cast<size_t>(i) * D, D, 1);
        brotensor::linear_forward_gpu(W2_g_, b2_g_, h_row, y_row);
    }
}

void FeedForward::backward(const brotensor::GpuTensor& dY, brotensor::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    const int K = X_cache_g_.rows;
    const int D = X_cache_g_.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);

    // dHpost (K, df), dHpre (K, df) — temporaries.
    brotensor::GpuTensor dHpost(K, df_);
    brotensor::GpuTensor dHpre(K, df_);

    // Layer 2 backward, per-row. dHpost rows are *overwritten* by linear_backward.
    for (int i = 0; i < K; ++i) {
        brotensor::GpuTensor h_row = brotensor::GpuTensor::view(
            H_post_g_.data + static_cast<size_t>(i) * df_, df_, 1);
        brotensor::GpuTensor dy_row = brotensor::GpuTensor::view(
            const_cast<float*>(dY.data) + static_cast<size_t>(i) * D, D, 1);
        brotensor::GpuTensor dHpost_row = brotensor::GpuTensor::view(
            dHpost.data + static_cast<size_t>(i) * df_, df_, 1);
        brotensor::linear_backward_gpu(W2_g_, h_row, dy_row,
                                 dHpost_row, dW2_g_, dB2_g_);
    }

    // ReLU backward over flat (K * df).
    brotensor::relu_backward_gpu(H_pre_g_, dHpost, dHpre);

    // Layer 1 backward, per-row. dX rows overwritten.
    for (int i = 0; i < K; ++i) {
        brotensor::GpuTensor x_row = brotensor::GpuTensor::view(
            X_cache_g_.data + static_cast<size_t>(i) * D, D, 1);
        brotensor::GpuTensor dHpre_row = brotensor::GpuTensor::view(
            dHpre.data + static_cast<size_t>(i) * df_, df_, 1);
        brotensor::GpuTensor dX_row = brotensor::GpuTensor::view(
            dX.data + static_cast<size_t>(i) * D, D, 1);
        brotensor::linear_backward_gpu(W1_g_, x_row, dHpre_row,
                                 dX_row, dW1_g_, dB1_g_);
    }
}

void FeedForward::forward_inference_batched(const brotensor::GpuTensor& X_RD,
                                            brotensor::GpuTensor& Y_RD) {
    assert(device_ == Device::GPU);
    const int R = X_RD.rows;
    if (Y_RD.rows != R || Y_RD.cols != d_) Y_RD.resize(R, d_);
    if (R == 0) return;
    brotensor::GpuTensor H_pre (R, df_);
    brotensor::GpuTensor H_post(R, df_);
    brotensor::linear_forward_batched_gpu(W1_g_, b1_g_, X_RD, H_pre);
    brotensor::relu_forward_batched_gpu(H_pre, H_post);
    brotensor::linear_forward_batched_gpu(W2_g_, b2_g_, H_post, Y_RD);
}
#endif

void FeedForward::to(Device d) {
    if (d == device_) return;
    device_require_cuda("FeedForward");
#ifdef BROTENSOR_HAS_GPU
    if (d == Device::GPU) {
        upload_to(W1_, W1_g_); upload_to(b1_, b1_g_);
        upload_to(W2_, W2_g_); upload_to(b2_, b2_g_);
        upload_to(dW1_, dW1_g_); upload_to(dB1_, dB1_g_);
        upload_to(dW2_, dW2_g_); upload_to(dB2_, dB2_g_);
        upload_to(vW1_, vW1_g_); upload_to(vB1_, vB1_g_);
        upload_to(vW2_, vW2_g_); upload_to(vB2_, vB2_g_);
        upload_to(mW1_, mW1_g_); upload_to(mB1_, mB1_g_);
        upload_to(mW2_, mW2_g_); upload_to(mB2_, mB2_g_);
        upload_to(vAW1_, vAW1_g_); upload_to(vAB1_, vAB1_g_);
        upload_to(vAW2_, vAW2_g_); upload_to(vAB2_, vAB2_g_);
        device_ = Device::GPU;
    } else {
        download_to(W1_g_, W1_); download_to(b1_g_, b1_);
        download_to(W2_g_, W2_); download_to(b2_g_, b2_);
        download_to(dW1_g_, dW1_); download_to(dB1_g_, dB1_);
        download_to(dW2_g_, dW2_); download_to(dB2_g_, dB2_);
        download_to(vW1_g_, vW1_); download_to(vB1_g_, vB1_);
        download_to(vW2_g_, vW2_); download_to(vB2_g_, vB2_);
        download_to(mW1_g_, mW1_); download_to(mB1_g_, mB1_);
        download_to(mW2_g_, mW2_); download_to(mB2_g_, mB2_);
        download_to(vAW1_g_, vAW1_); download_to(vAB1_g_, vAB1_);
        download_to(vAW2_g_, vAW2_); download_to(vAB2_g_, vAB2_);
        brotensor::cuda_sync();
        device_ = Device::CPU;
    }
#endif
}

void FeedForward::zero_grad() {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        dW1_g_.zero(); dB1_g_.zero(); dW2_g_.zero(); dB2_g_.zero();
        return;
    }
#endif
    dW1_.zero(); dB1_.zero(); dW2_.zero(); dB2_.zero();
}

static void sgd_buf_(Tensor& W, Tensor& vW, const Tensor& dW,
                     float lr, float momentum) {
    const int n = W.size();
    float* w = W.ptr(); float* v = vW.ptr(); const float* g = dW.ptr();
    for (int i = 0; i < n; ++i) {
        v[i] = momentum * v[i] + g[i];
        w[i] -= lr * v[i];
    }
}

void FeedForward::sgd_step(float lr, float momentum) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        brotensor::sgd_step_gpu(W1_g_, dW1_g_, vW1_g_, lr, momentum);
        brotensor::sgd_step_gpu(b1_g_, dB1_g_, vB1_g_, lr, momentum);
        brotensor::sgd_step_gpu(W2_g_, dW2_g_, vW2_g_, lr, momentum);
        brotensor::sgd_step_gpu(b2_g_, dB2_g_, vB2_g_, lr, momentum);
        return;
    }
#endif
    sgd_buf_(W1_, vW1_, dW1_, lr, momentum);
    sgd_buf_(b1_, vB1_, dB1_, lr, momentum);
    sgd_buf_(W2_, vW2_, dW2_, lr, momentum);
    sgd_buf_(b2_, vB2_, dB2_, lr, momentum);
}

void FeedForward::adam_step(float lr, float beta1, float beta2, float eps, int step) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        brotensor::adam_step_gpu(W1_g_, dW1_g_, mW1_g_, vAW1_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(b1_g_, dB1_g_, mB1_g_, vAB1_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(W2_g_, dW2_g_, mW2_g_, vAW2_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(b2_g_, dB2_g_, mB2_g_, vAB2_g_, lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(W1_, dW1_, mW1_, vAW1_, lr, beta1, beta2, eps, step);
    adam_step_cpu(b1_, dB1_, mB1_, vAB1_, lr, beta1, beta2, eps, step);
    adam_step_cpu(W2_, dW2_, mW2_, vAW2_, lr, beta1, beta2, eps, step);
    adam_step_cpu(b2_, dB2_, mB2_, vAB2_, lr, beta1, beta2, eps, step);
}

void FeedForward::save_to(std::vector<uint8_t>& out) const {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        auto* self = const_cast<FeedForward*>(this);
        download_to(W1_g_, self->W1_); download_to(b1_g_, self->b1_);
        download_to(W2_g_, self->W2_); download_to(b2_g_, self->b2_);
        brotensor::cuda_sync();
    }
#endif
    tensor_write(W1_, out); tensor_write(b1_, out);
    tensor_write(W2_, out); tensor_write(b2_, out);
}

void FeedForward::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(W1_, data, offset, size);
    tensor_read(b1_, data, offset, size);
    tensor_read(W2_, data, offset, size);
    tensor_read(b2_, data, offset, size);
    df_ = W1_.rows;
    d_  = W1_.cols;
    dW1_.resize(df_, d_); dB1_.resize(df_, 1);
    dW2_.resize(d_, df_); dB2_.resize(d_, 1);
    vW1_.resize(df_, d_); vB1_.resize(df_, 1);
    vW2_.resize(d_, df_); vB2_.resize(d_, 1);
    mW1_.resize(df_, d_); mB1_.resize(df_, 1);
    mW2_.resize(d_, df_); mB2_.resize(d_, 1);
    vAW1_.resize(df_, d_); vAB1_.resize(df_, 1);
    vAW2_.resize(d_, df_); vAB2_.resize(d_, 1);
    mW1_.zero(); mB1_.zero(); mW2_.zero(); mB2_.zero();
    vAW1_.zero(); vAB1_.zero(); vAW2_.zero(); vAB2_.zero();
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        upload_to(W1_, W1_g_); upload_to(b1_, b1_g_);
        upload_to(W2_, W2_g_); upload_to(b2_, b2_g_);
        upload_to(dW1_, dW1_g_); upload_to(dB1_, dB1_g_);
        upload_to(dW2_, dW2_g_); upload_to(dB2_, dB2_g_);
        upload_to(vW1_, vW1_g_); upload_to(vB1_, vB1_g_);
        upload_to(vW2_, vW2_g_); upload_to(vB2_, vB2_g_);
        upload_to(mW1_, mW1_g_); upload_to(mB1_, mB1_g_);
        upload_to(mW2_, mW2_g_); upload_to(mB2_, mB2_g_);
        upload_to(vAW1_, vAW1_g_); upload_to(vAB1_, vAB1_g_);
        upload_to(vAW2_, vAW2_g_); upload_to(vAB2_, vAB2_g_);
    }
#endif
}

} // namespace brogameagent::nn
