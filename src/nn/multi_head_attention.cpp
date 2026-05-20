#include "brogameagent/nn/multi_head_attention.h"

#include <brotensor/ops.h>

#include <cassert>

namespace brogameagent::nn {

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
    // resize() leaves contents undefined — zero every accumulator explicitly.
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
    vWq_.zero(); vWk_.zero(); vWv_.zero(); vWo_.zero();
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();

    brotensor::xavier_init(Wq_, rng_state);
    brotensor::xavier_init(Wk_, rng_state);
    brotensor::xavier_init(Wv_, rng_state);
    brotensor::xavier_init(Wo_, rng_state);

    X_cache_.resize(n_, d_);
    Qh_.resize(h_ * n_, dh_);
    Kh_.resize(h_ * n_, dh_);
    Vh_.resize(h_ * n_, dh_);
    Attnh_.resize(h_ * n_, n_);
    Yconcat_.resize(n_, d_);
    mask_ptr_ = nullptr;
}

void MultiHeadAttention::forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& O) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == d_);
    n_ = K;

    X_cache_ = X;
    // The mask is consumed verbatim by the device-dispatched mha op — it must
    // already live on X's device. We stash the pointer (no host copy) for the
    // matching backward; the caller owns the buffer's lifetime.
    mask_ptr_ = mask;

    brotensor::mha_forward(X, Wq_, Wk_, Wv_, Wo_,
                           mask_ptr_, h_,
                           Qh_, Kh_, Vh_, Attnh_, Yconcat_, O);
}

void MultiHeadAttention::backward(const brotensor::Tensor& dO, brotensor::Tensor& dX) {
    const int K = X_cache_.rows;
    const int D = X_cache_.cols;
    assert(dO.rows == K && dO.cols == D);
    (void)D;
    brotensor::mha_backward(dO, X_cache_, Qh_, Kh_, Vh_, Attnh_, Yconcat_,
                            Wq_, Wk_, Wv_, Wo_,
                            mask_ptr_, h_,
                            dX, dWq_, dWk_, dWv_, dWo_);
}

void MultiHeadAttention::to(brotensor::Device d) {
    if (d == device_) return;
    Wq_   = Wq_.to(d);   Wk_   = Wk_.to(d);   Wv_   = Wv_.to(d);   Wo_   = Wo_.to(d);
    dWq_  = dWq_.to(d);  dWk_  = dWk_.to(d);  dWv_  = dWv_.to(d);  dWo_  = dWo_.to(d);
    vWq_  = vWq_.to(d);  vWk_  = vWk_.to(d);  vWv_  = vWv_.to(d);  vWo_  = vWo_.to(d);
    mWq_  = mWq_.to(d);  mWk_  = mWk_.to(d);  mWv_  = mWv_.to(d);  mWo_  = mWo_.to(d);
    vAWq_ = vAWq_.to(d); vAWk_ = vAWk_.to(d); vAWv_ = vAWv_.to(d); vAWo_ = vAWo_.to(d);
    X_cache_ = X_cache_.to(d);
    Qh_ = Qh_.to(d); Kh_ = Kh_.to(d); Vh_ = Vh_.to(d);
    Attnh_ = Attnh_.to(d);
    Yconcat_ = Yconcat_.to(d);
    device_ = d;
}

void MultiHeadAttention::zero_grad() {
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
}

void MultiHeadAttention::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(Wq_, dWq_, vWq_, lr, momentum);
    brotensor::sgd_step(Wk_, dWk_, vWk_, lr, momentum);
    brotensor::sgd_step(Wv_, dWv_, vWv_, lr, momentum);
    brotensor::sgd_step(Wo_, dWo_, vWo_, lr, momentum);
}

void MultiHeadAttention::adam_step(float lr, float beta1, float beta2,
                                   float eps, int step) {
    brotensor::adam_step(Wq_, dWq_, mWq_, vAWq_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wk_, dWk_, mWk_, vAWk_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wv_, dWv_, mWv_, vAWv_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wo_, dWo_, mWo_, vAWo_, lr, beta1, beta2, eps, step);
}

void MultiHeadAttention::save_to(std::vector<uint8_t>& out) const {
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
    // Reset optimizer state and grad buffers to match loaded shapes. resize()
    // leaves contents undefined, so each buffer is explicitly zeroed.
    dWq_.resize(d_, d_); dWk_.resize(d_, d_); dWv_.resize(d_, d_); dWo_.resize(d_, d_);
    vWq_.resize(d_, d_); vWk_.resize(d_, d_); vWv_.resize(d_, d_); vWo_.resize(d_, d_);
    mWq_.resize(d_, d_); mWk_.resize(d_, d_); mWv_.resize(d_, d_); mWo_.resize(d_, d_);
    vAWq_.resize(d_, d_); vAWk_.resize(d_, d_); vAWv_.resize(d_, d_); vAWo_.resize(d_, d_);
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
    vWq_.zero(); vWk_.zero(); vWv_.zero(); vWo_.zero();
    mWq_.zero(); mWk_.zero(); mWv_.zero(); mWo_.zero();
    vAWq_.zero(); vAWk_.zero(); vAWv_.zero(); vAWo_.zero();
    Qh_.resize(h_ * n_, dh_);
    Kh_.resize(h_ * n_, dh_);
    Vh_.resize(h_ * n_, dh_);
    Attnh_.resize(h_ * n_, n_);
    Yconcat_.resize(n_, d_);
}

} // namespace brogameagent::nn
