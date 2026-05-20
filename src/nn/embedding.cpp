#include "brogameagent/nn/embedding.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void Embedding::init(int vocab, int dim, uint64_t& rng_state) {
    W_.resize(vocab, dim);
    dW_.resize(vocab, dim);
    vW_.resize(vocab, dim);
    mW_.resize(vocab, dim);
    vAW_.resize(vocab, dim);
    // resize() leaves contents undefined — zero every accumulator explicitly.
    dW_.zero(); vW_.zero(); mW_.zero(); vAW_.zero();
    brotensor::xavier_init(W_, rng_state);
}

void Embedding::forward(int idx, brotensor::Tensor& out) {
    assert(idx >= 0 && idx < W_.rows);
    assert(out.size() == W_.cols);
    const int d = W_.cols;
    std::memcpy(out.ptr(), W_.ptr() + static_cast<size_t>(idx) * d, d * sizeof(float));
}

void Embedding::backward(int idx, const brotensor::Tensor& dY) {
    assert(idx >= 0 && idx < W_.rows);
    assert(dY.size() == W_.cols);
    const int d = W_.cols;
    float* row = dW_.ptr() + static_cast<size_t>(idx) * d;
    const float* g = dY.ptr();
    for (int j = 0; j < d; ++j) row[j] += g[j];
}

void Embedding::to(brotensor::Device d) {
    if (d == device_) return;
    W_   = W_.to(d);
    dW_  = dW_.to(d);
    vW_  = vW_.to(d);
    mW_  = mW_.to(d);
    vAW_ = vAW_.to(d);
    device_ = d;
}

void Embedding::zero_grad() { dW_.zero(); }

void Embedding::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(W_, dW_, vW_, lr, momentum);
}

void Embedding::adam_step(float lr, float beta1, float beta2, float eps, int step) {
    brotensor::adam_step(W_, dW_, mW_, vAW_, lr, beta1, beta2, eps, step);
}

void Embedding::save_to(std::vector<uint8_t>& out) const {
    tensor_write(W_, out);
}

void Embedding::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(W_, data, offset, size);
    // Reset optimizer state and grad buffers to match loaded shapes. resize()
    // leaves contents undefined, so each buffer is explicitly zeroed.
    dW_.resize(W_.rows, W_.cols);
    vW_.resize(W_.rows, W_.cols);
    mW_.resize(W_.rows, W_.cols);
    vAW_.resize(W_.rows, W_.cols);
    dW_.zero(); vW_.zero(); mW_.zero(); vAW_.zero();
}

} // namespace brogameagent::nn
