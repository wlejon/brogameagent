#include "brogameagent/nn/layernorm.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
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

void LayerNorm::forward(const Tensor& x, Tensor& y) {
    const int n = x.size();
    assert(gamma_.size() == n && y.size() == n);
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m += x[i];
    m /= static_cast<float>(n);
    float v = 0.0f;
    for (int i = 0; i < n; ++i) { const float d = x[i] - m; v += d * d; }
    v /= static_cast<float>(n);
    const float rstd = 1.0f / std::sqrt(v + eps_);
    mean_ = m;
    rstd_ = rstd;
    for (int i = 0; i < n; ++i) {
        xhat_[i] = (x[i] - m) * rstd;
        y[i] = gamma_[i] * xhat_[i] + beta_[i];
    }
}

void LayerNorm::backward(const Tensor& dY, Tensor& dX) {
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

#ifdef BGA_HAS_CUDA
void LayerNorm::forward(const gpu::GpuTensor& x, gpu::GpuTensor& y) {
    assert(device_ == Device::GPU);
    const int n = gamma_.size();
    if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
    gpu::layernorm_forward_gpu(x, gamma_g_, beta_g_, y, xhat_g_,
                               mean_, rstd_, eps_);
}

void LayerNorm::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    gpu::layernorm_backward_gpu(dY, xhat_g_, gamma_g_, rstd_,
                                dX, dGamma_g_, dBeta_g_);
}
#endif

void LayerNorm::to(Device d) {
    if (d == device_) return;
    device_require_cuda("LayerNorm");
#ifdef BGA_HAS_CUDA
    if (d == Device::GPU) {
        // Upload params/grads/velocities; allocate xhat mirror.
        gpu::upload(gamma_, gamma_g_);
        gpu::upload(beta_,  beta_g_);
        gpu::upload(dGamma_, dGamma_g_);
        gpu::upload(dBeta_,  dBeta_g_);
        gpu::upload(vGamma_, vGamma_g_);
        gpu::upload(vBeta_,  vBeta_g_);
        gpu::upload(mGamma_, mGamma_g_);
        gpu::upload(mBeta_,  mBeta_g_);
        gpu::upload(vAGamma_, vAGamma_g_);
        gpu::upload(vABeta_,  vABeta_g_);
        const int n = gamma_.size();
        if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
        device_ = Device::GPU;
    } else {
        // Download back to host.
        gpu::download(gamma_g_, gamma_);
        gpu::download(beta_g_,  beta_);
        gpu::download(dGamma_g_, dGamma_);
        gpu::download(dBeta_g_,  dBeta_);
        gpu::download(vGamma_g_, vGamma_);
        gpu::download(vBeta_g_,  vBeta_);
        gpu::download(mGamma_g_, mGamma_);
        gpu::download(mBeta_g_,  mBeta_);
        gpu::download(vAGamma_g_, vAGamma_);
        gpu::download(vABeta_g_,  vABeta_);
        gpu::cuda_sync();
        device_ = Device::CPU;
    }
#endif
}

void LayerNorm::zero_grad() {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        dGamma_g_.zero();
        dBeta_g_.zero();
        return;
    }
#endif
    dGamma_.zero();
    dBeta_.zero();
}

void LayerNorm::sgd_step(float lr, float momentum) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::sgd_step_gpu(gamma_g_, dGamma_g_, vGamma_g_, lr, momentum);
        gpu::sgd_step_gpu(beta_g_,  dBeta_g_,  vBeta_g_,  lr, momentum);
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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::adam_step_gpu(gamma_g_, dGamma_g_, mGamma_g_, vAGamma_g_,
                           lr, beta1, beta2, eps, step);
        gpu::adam_step_gpu(beta_g_,  dBeta_g_,  mBeta_g_,  vABeta_g_,
                           lr, beta1, beta2, eps, step);
        return;
    }
#endif
    adam_step_cpu(gamma_, dGamma_, mGamma_, vAGamma_, lr, beta1, beta2, eps, step);
    adam_step_cpu(beta_,  dBeta_,  mBeta_,  vABeta_,  lr, beta1, beta2, eps, step);
}

void LayerNorm::save_to(std::vector<uint8_t>& out) const {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        // Sync host shadow before serializing. const_cast is local to here:
        // download writes into the (logically cached) host Tensor.
        auto* self = const_cast<LayerNorm*>(this);
        gpu::download(gamma_g_, self->gamma_);
        gpu::download(beta_g_,  self->beta_);
        gpu::cuda_sync();
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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        // Re-upload after deserialization.
        gpu::upload(gamma_, gamma_g_);
        gpu::upload(beta_,  beta_g_);
        gpu::upload(dGamma_, dGamma_g_);
        gpu::upload(dBeta_,  dBeta_g_);
        gpu::upload(vGamma_, vGamma_g_);
        gpu::upload(vBeta_,  vBeta_g_);
        gpu::upload(mGamma_, mGamma_g_);
        gpu::upload(mBeta_,  mBeta_g_);
        gpu::upload(vAGamma_, vAGamma_g_);
        gpu::upload(vABeta_,  vABeta_g_);
        if (xhat_g_.rows != n || xhat_g_.cols != 1) xhat_g_.resize(n, 1);
    }
#endif
}

} // namespace brogameagent::nn
