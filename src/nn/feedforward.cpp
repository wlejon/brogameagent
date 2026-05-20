#include "brogameagent/nn/feedforward.h"

#include <brotensor/ops.h>

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
    brotensor::xavier_init(W1_, rng_state);
    brotensor::xavier_init(W2_, rng_state);
    // resize() leaves contents undefined — zero every accumulator explicitly.
    b1_.zero(); b2_.zero();
    dW1_.zero(); dB1_.zero(); dW2_.zero(); dB2_.zero();
    vW1_.zero(); vB1_.zero(); vW2_.zero(); vB2_.zero();
    mW1_.zero(); mB1_.zero(); mW2_.zero(); mB2_.zero();
    vAW1_.zero(); vAB1_.zero(); vAW2_.zero(); vAB2_.zero();
}

void FeedForward::forward(const brotensor::Tensor& X, brotensor::Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == d_);
    (void)D;
    X_cache_ = X;

    // Layer 1: H_pre = X @ W1^T + b1 (per row), then ReLU.
    brotensor::linear_forward_batched(W1_, b1_, X, H_pre_);
    brotensor::relu_forward_batched(H_pre_, H_post_);

    // Layer 2: Y = H_post @ W2^T + b2 (per row).
    brotensor::linear_forward_batched(W2_, b2_, H_post_, Y);
    (void)K;
}

void FeedForward::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    const int K = X_cache_.rows;
    const int D = X_cache_.cols;
    assert(dY.rows == K && dY.cols == D);
    (void)D;

    // Layer 2 backward: Y = H_post @ W2^T + b2.
    brotensor::Tensor dHpost;
    brotensor::linear_backward_batched(W2_, H_post_, dY, dHpost, dW2_, dB2_);

    // ReLU backward: dH_pre = (H_pre > 0) * dH_post.
    brotensor::Tensor dHpre;
    brotensor::relu_backward_batched(H_pre_, dHpost, dHpre);

    // Layer 1 backward: H_pre = X @ W1^T + b1.
    brotensor::linear_backward_batched(W1_, X_cache_, dHpre, dX, dW1_, dB1_);
}

void FeedForward::to(brotensor::Device d) {
    if (d == device_) return;
    W1_ = W1_.to(d); b1_ = b1_.to(d); W2_ = W2_.to(d); b2_ = b2_.to(d);
    dW1_ = dW1_.to(d); dB1_ = dB1_.to(d); dW2_ = dW2_.to(d); dB2_ = dB2_.to(d);
    vW1_ = vW1_.to(d); vB1_ = vB1_.to(d); vW2_ = vW2_.to(d); vB2_ = vB2_.to(d);
    mW1_ = mW1_.to(d); mB1_ = mB1_.to(d); mW2_ = mW2_.to(d); mB2_ = mB2_.to(d);
    vAW1_ = vAW1_.to(d); vAB1_ = vAB1_.to(d); vAW2_ = vAW2_.to(d); vAB2_ = vAB2_.to(d);
    X_cache_ = X_cache_.to(d);
    H_pre_ = H_pre_.to(d);
    H_post_ = H_post_.to(d);
    device_ = d;
}

void FeedForward::zero_grad() {
    dW1_.zero(); dB1_.zero(); dW2_.zero(); dB2_.zero();
}

void FeedForward::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(W1_, dW1_, vW1_, lr, momentum);
    brotensor::sgd_step(b1_, dB1_, vB1_, lr, momentum);
    brotensor::sgd_step(W2_, dW2_, vW2_, lr, momentum);
    brotensor::sgd_step(b2_, dB2_, vB2_, lr, momentum);
}

void FeedForward::adam_step(float lr, float beta1, float beta2, float eps, int step) {
    brotensor::adam_step(W1_, dW1_, mW1_, vAW1_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(b1_, dB1_, mB1_, vAB1_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(W2_, dW2_, mW2_, vAW2_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(b2_, dB2_, mB2_, vAB2_, lr, beta1, beta2, eps, step);
}

void FeedForward::save_to(std::vector<uint8_t>& out) const {
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
    // Reset optimizer state and grad buffers to match loaded shapes. resize()
    // leaves contents undefined, so each buffer is explicitly zeroed.
    dW1_.resize(df_, d_); dB1_.resize(df_, 1);
    dW2_.resize(d_, df_); dB2_.resize(d_, 1);
    vW1_.resize(df_, d_); vB1_.resize(df_, 1);
    vW2_.resize(d_, df_); vB2_.resize(d_, 1);
    mW1_.resize(df_, d_); mB1_.resize(df_, 1);
    mW2_.resize(d_, df_); mB2_.resize(d_, 1);
    vAW1_.resize(df_, d_); vAB1_.resize(df_, 1);
    vAW2_.resize(d_, df_); vAB2_.resize(d_, 1);
    dW1_.zero(); dB1_.zero(); dW2_.zero(); dB2_.zero();
    vW1_.zero(); vB1_.zero(); vW2_.zero(); vB2_.zero();
    mW1_.zero(); mB1_.zero(); mW2_.zero(); mB2_.zero();
    vAW1_.zero(); vAB1_.zero(); vAW2_.zero(); vAB2_.zero();
}

} // namespace brogameagent::nn
