#include "brogameagent/nn/circuits.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

// ─── tensor_write / tensor_read ────────────────────────────────────────────

void tensor_write(const Tensor& t, std::vector<uint8_t>& out) {
    const int32_t r = t.rows;
    const int32_t c = t.cols;
    const size_t header = sizeof(int32_t) * 2;
    const size_t bytes  = static_cast<size_t>(t.size()) * sizeof(float);
    const size_t start  = out.size();
    out.resize(start + header + bytes);
    std::memcpy(out.data() + start, &r, sizeof(int32_t));
    std::memcpy(out.data() + start + sizeof(int32_t), &c, sizeof(int32_t));
    std::memcpy(out.data() + start + header, t.ptr(), bytes);
}

void tensor_read(Tensor& t, const uint8_t* data, size_t& offset, size_t size) {
    assert(offset + sizeof(int32_t) * 2 <= size);
    int32_t r = 0, c = 0;
    std::memcpy(&r, data + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    std::memcpy(&c, data + offset, sizeof(int32_t)); offset += sizeof(int32_t);
    t.resize(r, c);
    const size_t bytes = static_cast<size_t>(r) * c * sizeof(float);
    assert(offset + bytes <= size);
    std::memcpy(t.ptr(), data + offset, bytes);
    offset += bytes;
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
    x_cache_.resize(in_dim, 1);
    xavier_init(W_, rng_state);
    b_.zero();
    dW_.zero(); dB_.zero();
    vW_.zero(); vB_.zero();
}

void Linear::forward(const Tensor& x, Tensor& y) {
    x_cache_ = x;
    linear_forward(W_, b_, x, y);
}

void Linear::backward(const Tensor& dY, Tensor& dX) {
    linear_backward(W_, x_cache_, dY, dX, dW_, dB_);
}

void Linear::zero_grad() {
    dW_.zero();
    dB_.zero();
}

void Linear::sgd_step(float lr, float momentum) {
    const int nw = W_.size();
    float* w = W_.ptr();  float* vw = vW_.ptr(); const float* gw = dW_.ptr();
    for (int i = 0; i < nw; ++i) {
        vw[i] = momentum * vw[i] + gw[i];
        w[i] -= lr * vw[i];
    }
    const int nb = b_.size();
    float* bb = b_.ptr(); float* vb = vB_.ptr(); const float* gb = dB_.ptr();
    for (int i = 0; i < nb; ++i) {
        vb[i] = momentum * vb[i] + gb[i];
        bb[i] -= lr * vb[i];
    }
}

void Linear::save_to(std::vector<uint8_t>& out) const {
    tensor_write(W_, out);
    tensor_write(b_, out);
}

void Linear::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(W_, data, offset, size);
    tensor_read(b_, data, offset, size);
    // Reset optimizer state and grad buffers to match loaded shapes.
    dW_.resize(W_.rows, W_.cols);
    dB_.resize(b_.size(), 1);
    vW_.resize(W_.rows, W_.cols);
    vB_.resize(b_.size(), 1);
    x_cache_.resize(W_.cols, 1);
}

} // namespace brogameagent::nn
