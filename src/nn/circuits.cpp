#include "brogameagent/nn/circuits.h"

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brogameagent/nn/gpu_glue.h>
#endif

#include <cassert>
#include <cmath>
#include <cstring>

namespace brogameagent::nn {

// ─── adam_step_cpu ─────────────────────────────────────────────────────────

void adam_step_cpu(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                   float lr, float beta1, float beta2, float eps, int step) {
    assert(param.size() == grad.size());
    assert(param.size() == m.size());
    assert(param.size() == v.size());
    assert(step >= 1);
    const int n = param.size();
    float* p  = param.ptr();
    const float* g = grad.ptr();
    float* mp = m.ptr();
    float* vp = v.ptr();
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    for (int i = 0; i < n; ++i) {
        const float gi = g[i];
        const float mi = beta1 * mp[i] + (1.0f - beta1) * gi;
        const float vi = beta2 * vp[i] + (1.0f - beta2) * gi * gi;
        mp[i] = mi;
        vp[i] = vi;
        const float m_hat = mi * inv_bc1;
        const float v_hat = vi * inv_bc2;
        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

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
    mW_.resize(out_dim, in_dim);
    mB_.resize(out_dim, 1);
    vAW_.resize(out_dim, in_dim);
    vAB_.resize(out_dim, 1);
    x_cache_.resize(in_dim, 1);
    xavier_init(W_, rng_state);
    b_.zero();
    dW_.zero(); dB_.zero();
    vW_.zero(); vB_.zero();
    mW_.zero(); mB_.zero();
    vAW_.zero(); vAB_.zero();
}

void Linear::forward(const Tensor& x, Tensor& y) {
    x_cache_ = x;
    linear_forward(W_, b_, x, y);
}

void Linear::backward(const Tensor& dY, Tensor& dX) {
    linear_backward(W_, x_cache_, dY, dX, dW_, dB_);
}

void Linear::backward(const Tensor& x_input, const Tensor& dY, Tensor& dX) {
    linear_backward(W_, x_input, dY, dX, dW_, dB_);
}

#ifdef BROTENSOR_HAS_GPU
void Linear::forward(const brotensor::GpuTensor& x, brotensor::GpuTensor& y) {
    assert(device_ == Device::GPU);
    // Cache a non-owning view of x so backward can read the same input. The
    // caller must keep x alive between forward and backward (matches the CPU
    // x_cache_ semantics where we stash a value copy).
    x_cache_g_ = brotensor::GpuTensor::view(x.data, x.rows, x.cols);
    brotensor::linear_forward_gpu(W_g_, b_g_, x, y);
}

void Linear::backward(const brotensor::GpuTensor& dY, brotensor::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    brotensor::linear_backward_gpu(W_g_, x_cache_g_, dY, dX, dW_g_, dB_g_);
}

void Linear::forward_batched_train(const brotensor::GpuTensor& X_BD,
                                   brotensor::GpuTensor& Y_BD) {
    assert(device_ == Device::GPU);
    x_cache_btr_g_ = brotensor::GpuTensor::view(X_BD.data, X_BD.rows, X_BD.cols);
    brotensor::linear_forward_batched_gpu(W_g_, b_g_, X_BD, Y_BD);
}

void Linear::backward_batched(const brotensor::GpuTensor& dY_BD,
                              brotensor::GpuTensor& dX_BD) {
    assert(device_ == Device::GPU);
    brotensor::linear_backward_batched_gpu(W_g_, x_cache_btr_g_, dY_BD,
                                     dX_BD, dW_g_, dB_g_);
}
#endif

void Linear::to(Device d) {
    if (d == device_) return;
    device_require_cuda("Linear");
#ifdef BROTENSOR_HAS_GPU
    if (d == Device::GPU) {
        upload_to(W_, W_g_);
        upload_to(b_, b_g_);
        upload_to(dW_, dW_g_);
        upload_to(dB_, dB_g_);
        upload_to(vW_, vW_g_);
        upload_to(vB_, vB_g_);
        upload_to(mW_, mW_g_);
        upload_to(mB_, mB_g_);
        upload_to(vAW_, vAW_g_);
        upload_to(vAB_, vAB_g_);
        device_ = Device::GPU;
    } else {
        download_to(W_g_, W_);
        download_to(b_g_, b_);
        download_to(dW_g_, dW_);
        download_to(dB_g_, dB_);
        download_to(vW_g_, vW_);
        download_to(vB_g_, vB_);
        download_to(mW_g_, mW_);
        download_to(mB_g_, mB_);
        download_to(vAW_g_, vAW_);
        download_to(vAB_g_, vAB_);
        brotensor::cuda_sync();
        device_ = Device::CPU;
    }
#endif
}

void Linear::zero_grad() {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        dW_g_.zero();
        dB_g_.zero();
        return;
    }
#endif
    dW_.zero();
    dB_.zero();
}

void Linear::sgd_step(float lr, float momentum) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        brotensor::sgd_step_gpu(W_g_, dW_g_, vW_g_, lr, momentum);
        brotensor::sgd_step_gpu(b_g_, dB_g_, vB_g_, lr, momentum);
        return;
    }
#endif
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

void Linear::adam_step(float lr, float beta1, float beta2, float eps, int step) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        brotensor::adam_step_gpu(W_g_, dW_g_, mW_g_, vAW_g_, lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(b_g_, dB_g_, mB_g_, vAB_g_, lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(W_, dW_, mW_, vAW_, lr, beta1, beta2, eps, step);
    adam_step_cpu(b_, dB_, mB_, vAB_, lr, beta1, beta2, eps, step);
}

void Linear::save_to(std::vector<uint8_t>& out) const {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        // Sync host shadow before serializing.
        auto* self = const_cast<Linear*>(this);
        download_to(W_g_, self->W_);
        download_to(b_g_, self->b_);
        brotensor::cuda_sync();
    }
#endif
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
    mW_.resize(W_.rows, W_.cols);
    mB_.resize(b_.size(), 1);
    vAW_.resize(W_.rows, W_.cols);
    vAB_.resize(b_.size(), 1);
    mW_.zero(); mB_.zero(); vAW_.zero(); vAB_.zero();
    x_cache_.resize(W_.cols, 1);
#ifdef BROTENSOR_HAS_GPU
    if (device_ == Device::GPU) {
        upload_to(W_, W_g_);
        upload_to(b_, b_g_);
        upload_to(dW_, dW_g_);
        upload_to(dB_, dB_g_);
        upload_to(vW_, vW_g_);
        upload_to(vB_, vB_g_);
        upload_to(mW_, mW_g_);
        upload_to(mB_, mB_g_);
        upload_to(vAW_, vAW_g_);
        upload_to(vAB_, vAB_g_);
    }
#endif
}

} // namespace brogameagent::nn
