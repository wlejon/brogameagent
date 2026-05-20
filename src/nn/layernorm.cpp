#include "brogameagent/nn/layernorm.h"

#include <brotensor/ops.h>

namespace brogameagent::nn {

void LayerNorm::init(int n, float eps) {
    eps_ = eps;
    gamma_.resize(n, 1);
    beta_.resize(n, 1);
    dGamma_.resize(n, 1); dBeta_.resize(n, 1);
    vGamma_.resize(n, 1); vBeta_.resize(n, 1);
    mGamma_.resize(n, 1); mBeta_.resize(n, 1);
    vAGamma_.resize(n, 1); vABeta_.resize(n, 1);
    // resize() leaves contents undefined — zero every accumulator explicitly.
    dGamma_.zero(); dBeta_.zero();
    vGamma_.zero(); vBeta_.zero();
    mGamma_.zero(); mBeta_.zero();
    vAGamma_.zero(); vABeta_.zero();
    xhat_.resize(n, 1);
    for (int i = 0; i < n; ++i) { gamma_[i] = 1.0f; beta_[i] = 0.0f; }
}

void LayerNorm::forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
    brotensor::layernorm_forward(x, gamma_, beta_, y, xhat_, mean_, rstd_, eps_);
}

void LayerNorm::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    brotensor::layernorm_backward(dY, xhat_, gamma_, rstd_, dX, dGamma_, dBeta_);
}

void LayerNorm::to(brotensor::Device d) {
    if (d == device_) return;
    gamma_   = gamma_.to(d);   beta_   = beta_.to(d);
    dGamma_  = dGamma_.to(d);  dBeta_  = dBeta_.to(d);
    vGamma_  = vGamma_.to(d);  vBeta_  = vBeta_.to(d);
    mGamma_  = mGamma_.to(d);  mBeta_  = mBeta_.to(d);
    vAGamma_ = vAGamma_.to(d); vABeta_ = vABeta_.to(d);
    xhat_    = xhat_.to(d);
    device_  = d;
}

void LayerNorm::zero_grad() {
    dGamma_.zero();
    dBeta_.zero();
}

void LayerNorm::sgd_step(float lr, float momentum) {
    brotensor::sgd_step(gamma_, dGamma_, vGamma_, lr, momentum);
    brotensor::sgd_step(beta_,  dBeta_,  vBeta_,  lr, momentum);
}

void LayerNorm::adam_step(float lr, float beta1, float beta2, float eps, int step) {
    brotensor::adam_step(gamma_, dGamma_, mGamma_, vAGamma_, lr, beta1, beta2, eps, step);
    brotensor::adam_step(beta_,  dBeta_,  mBeta_,  vABeta_,  lr, beta1, beta2, eps, step);
}

void LayerNorm::save_to(std::vector<uint8_t>& out) const {
    tensor_write(gamma_, out);
    tensor_write(beta_, out);
}

void LayerNorm::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(gamma_, data, offset, size);
    tensor_read(beta_, data, offset, size);
    const int n = gamma_.size();
    dGamma_.resize(n, 1);  dBeta_.resize(n, 1);
    vGamma_.resize(n, 1);  vBeta_.resize(n, 1);
    mGamma_.resize(n, 1);  mBeta_.resize(n, 1);
    vAGamma_.resize(n, 1); vABeta_.resize(n, 1);
    dGamma_.zero(); dBeta_.zero();
    vGamma_.zero(); vBeta_.zero();
    mGamma_.zero(); mBeta_.zero();
    vAGamma_.zero(); vABeta_.zero();
    xhat_.resize(n, 1);
}

} // namespace brogameagent::nn
