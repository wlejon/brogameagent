#include "brogameagent/nn/attention.h"

#include <brotensor/ops.h>

#include <cassert>

namespace brogameagent::nn {

void ScaledDotProductAttention::init(int n_slots, int dim, uint64_t& rng_state, int /*num_heads*/) {
    n_ = n_slots;
    d_ = dim;
    Wq_.resize(d_, d_); Wk_.resize(d_, d_); Wv_.resize(d_, d_); Wo_.resize(d_, d_);
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
    Q_.resize(n_, d_); K_.resize(n_, d_); V_.resize(n_, d_);
    Attn_.resize(n_, n_);
    Y_.resize(n_, d_);
    mask_ptr_ = nullptr;
}

void ScaledDotProductAttention::forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& O) {
    assert(X.rows == n_ && X.cols == d_);
    X_cache_ = X;
    // The mask feeds the device-dispatched attention op verbatim — it must
    // live on X's device. Stash the pointer for backward; caller owns it.
    mask_ptr_ = mask;
    brotensor::attention_forward(X, Wq_, Wk_, Wv_, Wo_,
                                 mask_ptr_,
                                 Q_, K_, V_, Attn_, Y_, O);
}

void ScaledDotProductAttention::backward(const brotensor::Tensor& dO, brotensor::Tensor& dX) {
    assert(dO.rows == n_ && dO.cols == d_);
    brotensor::attention_backward(dO, X_cache_, Q_, K_, V_, Attn_, Y_,
                                  Wq_, Wk_, Wv_, Wo_,
                                  mask_ptr_,
                                  dX,
                                  dWq_, dWk_, dWv_, dWo_);
}

void ScaledDotProductAttention::to(brotensor::Device d) {
    if (d == device_) return;
    Wq_   = Wq_.to(d);   Wk_   = Wk_.to(d);   Wv_   = Wv_.to(d);   Wo_   = Wo_.to(d);
    dWq_  = dWq_.to(d);  dWk_  = dWk_.to(d);  dWv_  = dWv_.to(d);  dWo_  = dWo_.to(d);
    vWq_  = vWq_.to(d);  vWk_  = vWk_.to(d);  vWv_  = vWv_.to(d);  vWo_  = vWo_.to(d);
    mWq_  = mWq_.to(d);  mWk_  = mWk_.to(d);  mWv_  = mWv_.to(d);  mWo_  = mWo_.to(d);
    vAWq_ = vAWq_.to(d); vAWk_ = vAWk_.to(d); vAWv_ = vAWv_.to(d); vAWo_ = vAWo_.to(d);
    X_cache_ = X_cache_.to(d);
    Q_ = Q_.to(d); K_ = K_.to(d); V_ = V_.to(d);
    Attn_ = Attn_.to(d);
    Y_ = Y_.to(d);
    device_ = d;
}

void ScaledDotProductAttention::zero_grad() {
    dWq_.zero(); dWk_.zero(); dWv_.zero(); dWo_.zero();
}

void ScaledDotProductAttention::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(Wq_, dWq_, vWq_, lr, momentum);
    brotensor::sgd_step(Wk_, dWk_, vWk_, lr, momentum);
    brotensor::sgd_step(Wv_, dWv_, vWv_, lr, momentum);
    brotensor::sgd_step(Wo_, dWo_, vWo_, lr, momentum);
}

void ScaledDotProductAttention::adam_step(float lr, float beta1, float beta2,
                                          float eps, int step) {
    brotensor::adam_step(Wq_, dWq_, mWq_, vAWq_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wk_, dWk_, mWk_, vAWk_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wv_, dWv_, mWv_, vAWv_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(Wo_, dWo_, mWo_, vAWo_, lr, beta1, beta2, eps, step);
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
}

} // namespace brogameagent::nn
