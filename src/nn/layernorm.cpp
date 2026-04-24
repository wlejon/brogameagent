#include "brogameagent/nn/layernorm.h"

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

void LayerNorm::zero_grad() {
    dGamma_.zero();
    dBeta_.zero();
}

void LayerNorm::sgd_step(float lr, float momentum) {
    const int n = gamma_.size();
    for (int i = 0; i < n; ++i) {
        vGamma_[i] = momentum * vGamma_[i] + dGamma_[i];
        gamma_[i] -= lr * vGamma_[i];
        vBeta_[i]  = momentum * vBeta_[i] + dBeta_[i];
        beta_[i]  -= lr * vBeta_[i];
    }
}

void LayerNorm::save_to(std::vector<uint8_t>& out) const {
    tensor_write(gamma_, out);
    tensor_write(beta_, out);
}

void LayerNorm::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(gamma_, data, offset, size);
    tensor_read(beta_, data, offset, size);
    const int n = gamma_.size();
    dGamma_.resize(n, 1); dBeta_.resize(n, 1);
    vGamma_.resize(n, 1); vBeta_.resize(n, 1);
    xhat_.resize(n, 1);
}

} // namespace brogameagent::nn
