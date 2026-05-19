#include "brogameagent/nn/layernorm.h"

#ifdef BROTENSOR_HAS_GPU
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#endif

#include <cassert>
#include <cmath>

namespace brogameagent::nn {

void LayerNorm::init(int n, float eps) {
    eps_ = eps;
    gamma_.resize(n, 1);
    beta_.resize(n, 1);
    dGamma_.resize(n, 1);
    dBeta_.resize(n, 1);
    vGamma_.resize(n, 1);
    vBeta_.resize(n, 1);
    mGamma_.resize(n, 1); mBeta_.resize(n, 1);
    vAGamma_.resize(n, 1); vABeta_.resize(n, 1);
    mGamma_.zero(); mBeta_.zero();
    vAGamma_.zero(); vABeta_.zero();
    xhat_.resize(n, 1);
    for (int i = 0; i < n; ++i) { gamma_[i] = 1.0f; beta_[i] = 0.0f; }
}

void LayerNorm::forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
    const int n = x.size();
    assert(gamma_.size() == n && y.size() == n);
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) { sum += x[i]; sum_sq += x[i] * x[i]; }
    const float nf = static_cast<float>(n);
    const float m = sum / nf;
    float v = sum_sq / nf - m * m;
    if (v < 0.0f) v = 0.0f;  // guard against tiny negative from FP error
    const float rstd = 1.0f / std::sqrt(v + eps_);
    mean_ = m;
    rstd_ = rstd;
    for (int i = 0; i < n; ++i) {
        xhat_[i] = (x[i] - m) * rstd;
        y[i] = gamma_[i] * xhat_[i] + beta_[i];
    }
}

void LayerNorm::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    const int n = dY.size();
    const float nf = static_cast<float>(n);

    // Accumulate param grads.
    for (int i = 0; i < n; ++i) {
        dGamma_[i] += dY[i] * xhat_[i];
        dBeta_[i]  += dY[i];
    }

    // dX_hat_i = dY_i * gamma_i
    // dX_i = rstd/N * (N*dX_hat_i - sum(dX_hat) - xhat_i * sum(dX_hat * xhat))
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float dxh = dY[i] * gamma_[i];
        sum_dxhat      += dxh;
        sum_dxhat_xhat += dxh * xhat_[i];
    }
    const float scale = rstd_ / nf;
    for (int i = 0; i < n; ++i) {
        const float dxh = dY[i] * gamma_[i];
        dX[i] = scale * (nf * dxh - sum_dxhat - xhat_[i] * sum_dxhat_xhat);
    }
}

#ifdef BROTENSOR_HAS_GPU
void LayerNorm::forward(const brotensor::GpuTensor& x, brotensor::GpuTensor& y) {
    assert(device_ == brotensor::Device::GPU);
    const int n = gamma_.size();
    if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
    brotensor::layernorm_forward_gpu(x, gamma_g_, beta_g_, y, xhat_g_,
                               mean_, rstd_, eps_);
}

void LayerNorm::backward(const brotensor::GpuTensor& dY, brotensor::GpuTensor& dX) {
    assert(device_ == brotensor::Device::GPU);
    brotensor::layernorm_backward_gpu(dY, xhat_g_, gamma_g_, rstd_,
                                dX, dGamma_g_, dBeta_g_);
}
#endif

void LayerNorm::to(brotensor::Device d) {
    if (d == device_) return;
    brotensor::device_require_gpu("LayerNorm");
#ifdef BROTENSOR_HAS_GPU
    if (d == brotensor::Device::GPU) {
        // Upload params/grads/velocities; allocate xhat mirror.
        brotensor::upload(gamma_, gamma_g_);
        brotensor::upload(beta_,  beta_g_);
        brotensor::upload(dGamma_, dGamma_g_);
        brotensor::upload(dBeta_,  dBeta_g_);
        brotensor::upload(vGamma_, vGamma_g_);
        brotensor::upload(vBeta_,  vBeta_g_);
        brotensor::upload(mGamma_, mGamma_g_);
        brotensor::upload(mBeta_,  mBeta_g_);
        brotensor::upload(vAGamma_, vAGamma_g_);
        brotensor::upload(vABeta_,  vABeta_g_);
        const int n = gamma_.size();
        if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
        device_ = brotensor::Device::GPU;
    } else {
        // Download back to host.
        brotensor::download(gamma_g_, gamma_);
        brotensor::download(beta_g_,  beta_);
        brotensor::download(dGamma_g_, dGamma_);
        brotensor::download(dBeta_g_,  dBeta_);
        brotensor::download(vGamma_g_, vGamma_);
        brotensor::download(vBeta_g_,  vBeta_);
        brotensor::download(mGamma_g_, mGamma_);
        brotensor::download(mBeta_g_,  mBeta_);
        brotensor::download(vAGamma_g_, vAGamma_);
        brotensor::download(vABeta_g_,  vABeta_);
        brotensor::cuda_sync();
        device_ = brotensor::Device::CPU;
    }
#endif
}

void LayerNorm::zero_grad() {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        dGamma_g_.zero();
        dBeta_g_.zero();
        return;
    }
#endif
    dGamma_.zero();
    dBeta_.zero();
}

void LayerNorm::sgd_step(float lr, float momentum) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        brotensor::sgd_step_gpu(gamma_g_, dGamma_g_, vGamma_g_, lr, momentum);
        brotensor::sgd_step_gpu(beta_g_,  dBeta_g_,  vBeta_g_,  lr, momentum);
        return;
    }
#endif
    const int n = gamma_.size();
    for (int i = 0; i < n; ++i) {
        vGamma_[i] = momentum * vGamma_[i] + dGamma_[i];
        gamma_[i] -= lr * vGamma_[i];
        vBeta_[i]  = momentum * vBeta_[i] + dBeta_[i];
        beta_[i]  -= lr * vBeta_[i];
    }
}

void LayerNorm::adam_step(float lr, float beta1, float beta2, float eps, int step) {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        brotensor::adam_step_gpu(gamma_g_, dGamma_g_, mGamma_g_, vAGamma_g_,
                           lr, beta1, beta2, eps, step);
        brotensor::adam_step_gpu(beta_g_,  dBeta_g_,  mBeta_g_,  vABeta_g_,
                           lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(gamma_, dGamma_, mGamma_, vAGamma_, lr, beta1, beta2, eps, step);
    adam_step_cpu(beta_,  dBeta_,  mBeta_,  vABeta_,  lr, beta1, beta2, eps, step);
}

void LayerNorm::save_to(std::vector<uint8_t>& out) const {
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        // Sync host shadow before serializing. const_cast is local to here:
        // download writes into the (logically cached) host brotensor::Tensor.
        auto* self = const_cast<LayerNorm*>(this);
        brotensor::download(gamma_g_, self->gamma_);
        brotensor::download(beta_g_,  self->beta_);
        brotensor::cuda_sync();
    }
#endif
    tensor_write(gamma_, out);
    tensor_write(beta_, out);
}

void LayerNorm::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(gamma_, data, offset, size);
    tensor_read(beta_, data, offset, size);
    const int n = gamma_.size();
    dGamma_.resize(n, 1); dBeta_.resize(n, 1);
    vGamma_.resize(n, 1); vBeta_.resize(n, 1);
    mGamma_.resize(n, 1); mBeta_.resize(n, 1);
    vAGamma_.resize(n, 1); vABeta_.resize(n, 1);
    mGamma_.zero(); mBeta_.zero();
    vAGamma_.zero(); vABeta_.zero();
    xhat_.resize(n, 1);
#ifdef BROTENSOR_HAS_GPU
    if (device_ == brotensor::Device::GPU) {
        // Re-upload after deserialization.
        brotensor::upload(gamma_, gamma_g_);
        brotensor::upload(beta_,  beta_g_);
        brotensor::upload(dGamma_, dGamma_g_);
        brotensor::upload(dBeta_,  dBeta_g_);
        brotensor::upload(vGamma_, vGamma_g_);
        brotensor::upload(vBeta_,  vBeta_g_);
        brotensor::upload(mGamma_, mGamma_g_);
        brotensor::upload(mBeta_,  mBeta_g_);
        brotensor::upload(vAGamma_, vAGamma_g_);
        brotensor::upload(vABeta_,  vABeta_g_);
        if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
    }
#endif
}

} // namespace brogameagent::nn
