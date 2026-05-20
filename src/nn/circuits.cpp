#include "brogameagent/nn/circuits.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

// ─── adam_step_cpu ─────────────────────────────────────────────────────────
//
// Thin wrapper over the device-dispatched brotensor::adam_step. Kept so the
// many existing call sites need not change; the update runs on whatever
// device the tensors live on.

void adam_step_cpu(brotensor::Tensor& param, const brotensor::Tensor& grad,
                   brotensor::Tensor& m, brotensor::Tensor& v,
                   float lr, float beta1, float beta2, float eps, int step) {
    brotensor::adam_step(param, grad, m, v, lr, beta1, beta2, eps, step);
}

// ─── tensor_write / tensor_read ────────────────────────────────────────────
//
// Serialization is host-side: a GPU-resident tensor is staged through a host
// copy. tensor_read restores onto whatever device the destination currently
// lives on, so loading into an already-migrated layer keeps it on its device.

void tensor_write(const brotensor::Tensor& t, std::vector<uint8_t>& out) {
    const brotensor::Tensor h = t.to(brotensor::Device::CPU);
    const int32_t r = h.rows;
    const int32_t c = h.cols;
    const size_t header = sizeof(int32_t) * 2;
    const size_t bytes  = static_cast<size_t>(h.size()) * sizeof(float);
    const size_t start  = out.size();
    out.resize(start + header + bytes);
    std::memcpy(out.data() + start, &r, sizeof(int32_t));
    std::memcpy(out.data() + start + sizeof(int32_t), &c, sizeof(int32_t));
    if (bytes) std::memcpy(out.data() + start + header, h.ptr(), bytes);
}

void tensor_read(brotensor::Tensor& t, const uint8_t* data, size_t& offset, size_t size) {
    assert(offset + sizeof(int32_t) * 2 <= size);
    int32_t r = 0, c = 0;
    std::memcpy(&r, data + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    std::memcpy(&c, data + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    const size_t bytes = static_cast<size_t>(r) * c * sizeof(float);
    assert(offset + bytes <= size);
    brotensor::Tensor h = brotensor::Tensor::mat(r, c);
    if (bytes) std::memcpy(h.ptr(), data + offset, bytes);
    offset += bytes;
    t = h.to(t.device);
}

// ─── Linear ────────────────────────────────────────────────────────────────

Linear::Linear(int in_dim, int out_dim, uint64_t& rng_state) {
    init(in_dim, out_dim, rng_state);
}

void Linear::init(int in_dim, int out_dim, uint64_t& rng_state) {
    W_.resize(out_dim, in_dim);
    b_.resize(out_dim, 1);
    dW_.resize(out_dim, in_dim);
    dB_.resize(out_dim, 1);
    vW_.resize(out_dim, in_dim);
    vB_.resize(out_dim, 1);
    mW_.resize(out_dim, in_dim);
    mB_.resize(out_dim, 1);
    vAW_.resize(out_dim, in_dim);
    vAB_.resize(out_dim, 1);
    x_cache_.resize(in_dim, 1);
    brotensor::xavier_init(W_, rng_state);
    b_.zero();
    dW_.zero(); dB_.zero();
    vW_.zero(); vB_.zero();
    mW_.zero(); mB_.zero();
    vAW_.zero(); vAB_.zero();
}

void Linear::forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
    x_cache_ = x;
    brotensor::linear_forward(W_, b_, x, y);
}

void Linear::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    brotensor::linear_backward(W_, x_cache_, dY, dX, dW_, dB_);
}

void Linear::backward(const brotensor::Tensor& x_input, const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    brotensor::linear_backward(W_, x_input, dY, dX, dW_, dB_);
}

void Linear::forward_batched_train(const brotensor::Tensor& X_BD, brotensor::Tensor& Y_BD) {
    x_cache_btr_ = X_BD;
    brotensor::linear_forward_batched(W_, b_, X_BD, Y_BD);
}

void Linear::backward_batched(const brotensor::Tensor& dY_BD, brotensor::Tensor& dX_BD) {
    brotensor::linear_backward_batched(W_, x_cache_btr_, dY_BD, dX_BD, dW_, dB_);
}

void Linear::to(brotensor::Device d) {
    if (d == device_) return;
    W_   = W_.to(d);   b_   = b_.to(d);
    dW_  = dW_.to(d);  dB_  = dB_.to(d);
    vW_  = vW_.to(d);  vB_  = vB_.to(d);
    mW_  = mW_.to(d);  mB_  = mB_.to(d);
    vAW_ = vAW_.to(d); vAB_ = vAB_.to(d);
    device_ = d;
}

void Linear::zero_grad() {
    dW_.zero();
    dB_.zero();
}

void Linear::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(W_, dW_, vW_, lr, momentum);
    brotensor::sgd_step(b_, dB_, vB_, lr, momentum);
}

void Linear::adam_step(float lr, float beta1, float beta2, float eps, int step) {
    brotensor::adam_step(W_, dW_, mW_, vAW_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(b_, dB_, mB_, vAB_, lr, beta1, beta2, eps, step);
}

void Linear::save_to(std::vector<uint8_t>& out) const {
    tensor_write(W_, out);
    tensor_write(b_, out);
}

void Linear::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(W_, data, offset, size);
    tensor_read(b_, data, offset, size);
    // Reset optimizer state and grad buffers to match loaded shapes. resize()
    // leaves contents undefined, so each buffer is explicitly zeroed.
    dW_.resize(W_.rows, W_.cols);   dW_.zero();
    dB_.resize(b_.size(), 1);       dB_.zero();
    vW_.resize(W_.rows, W_.cols);   vW_.zero();
    vB_.resize(b_.size(), 1);       vB_.zero();
    mW_.resize(W_.rows, W_.cols);   mW_.zero();
    mB_.resize(b_.size(), 1);       mB_.zero();
    vAW_.resize(W_.rows, W_.cols);  vAW_.zero();
    vAB_.resize(b_.size(), 1);      vAB_.zero();
    x_cache_.resize(W_.cols, 1);
}

} // namespace brogameagent::nn
