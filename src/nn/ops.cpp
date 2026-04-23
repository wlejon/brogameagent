#include "brogameagent/nn/ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace brogameagent::nn {

void linear_forward(const Tensor& W, const Tensor& b, const Tensor& x, Tensor& y) {
    assert(W.cols == x.size());
    assert(W.rows == b.size());
    assert(y.size() == W.rows);

    const int out = W.rows;
    const int in  = W.cols;
    const float* Wp = W.ptr();
    const float* xp = x.ptr();
    const float* bp = b.ptr();
    float* yp = y.ptr();

    for (int i = 0; i < out; ++i) {
        float acc = bp[i];
        const float* row = Wp + static_cast<size_t>(i) * in;
        for (int j = 0; j < in; ++j) acc += row[j] * xp[j];
        yp[i] = acc;
    }
}

void linear_backward(const Tensor& W, const Tensor& x, const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB) {
    const int out = W.rows;
    const int in  = W.cols;
    assert(x.size() == in);
    assert(dY.size() == out);
    assert(dX.size() == in);
    assert(dW.rows == out && dW.cols == in);
    assert(dB.size() == out);

    const float* Wp  = W.ptr();
    const float* xp  = x.ptr();
    const float* dYp = dY.ptr();
    float* dXp = dX.ptr();
    float* dWp = dW.ptr();
    float* dBp = dB.ptr();

    // dX = W^T * dY
    for (int j = 0; j < in; ++j) dXp[j] = 0.0f;
    for (int i = 0; i < out; ++i) {
        const float g = dYp[i];
        const float* row = Wp + static_cast<size_t>(i) * in;
        for (int j = 0; j < in; ++j) dXp[j] += row[j] * g;
    }
    // dW += dY * x^T  (outer product)
    for (int i = 0; i < out; ++i) {
        const float g = dYp[i];
        float* row = dWp + static_cast<size_t>(i) * in;
        for (int j = 0; j < in; ++j) row[j] += g * xp[j];
    }
    // dB += dY
    for (int i = 0; i < out; ++i) dBp[i] += dYp[i];
}

void relu_forward(const Tensor& x, Tensor& y) {
    const int n = x.size();
    const float* xp = x.ptr();
    float* yp = y.ptr();
    for (int i = 0; i < n; ++i) yp[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
}

void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const int n = x.size();
    const float* xp  = x.ptr();
    const float* dYp = dY.ptr();
    float* dXp = dX.ptr();
    for (int i = 0; i < n; ++i) dXp[i] = xp[i] > 0.0f ? dYp[i] : 0.0f;
}

void tanh_forward(const Tensor& x, Tensor& y) {
    const int n = x.size();
    const float* xp = x.ptr();
    float* yp = y.ptr();
    for (int i = 0; i < n; ++i) yp[i] = std::tanh(xp[i]);
}

void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    const int n = y.size();
    const float* yp  = y.ptr();
    const float* dYp = dY.ptr();
    float* dXp = dX.ptr();
    for (int i = 0; i < n; ++i) dXp[i] = dYp[i] * (1.0f - yp[i] * yp[i]);
}

void softmax_forward(const Tensor& logits, Tensor& probs, const float* mask) {
    const int n = logits.size();
    const float* lp = logits.ptr();
    float* pp = probs.ptr();

    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0.0f) continue;
        if (lp[i] > m) m = lp[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0.0f) { pp[i] = 0.0f; continue; }
        pp[i] = std::exp(lp[i] - m);
        s += pp[i];
    }
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < n; ++i) pp[i] *= inv;
}

void softmax_backward(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits) {
    // dL/dz_i = p_i * (dL/dp_i - sum_j dL/dp_j * p_j)
    const int n = probs.size();
    const float* pp = probs.ptr();
    const float* dp = dProbs.ptr();
    float* dz = dLogits.ptr();
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) dot += dp[i] * pp[i];
    for (int i = 0; i < n; ++i) dz[i] = pp[i] * (dp[i] - dot);
}

float softmax_xent(const Tensor& logits, const Tensor& target,
                   Tensor& probs, Tensor& dLogits,
                   const float* mask) {
    softmax_forward(logits, probs, mask);
    const int n = logits.size();
    const float* tp = target.ptr();
    const float* pp = probs.ptr();
    float* dz = dLogits.ptr();
    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0.0f) { dz[i] = 0.0f; continue; }
        if (tp[i] > 0.0f) {
            const float p = pp[i] > 1e-12f ? pp[i] : 1e-12f;
            loss -= tp[i] * std::log(p);
        }
        dz[i] = pp[i] - tp[i];
    }
    return loss;
}

float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

void add_inplace(Tensor& y, const Tensor& x) {
    const int n = y.size();
    assert(x.size() == n);
    const float* xp = x.ptr();
    float* yp = y.ptr();
    for (int i = 0; i < n; ++i) yp[i] += xp[i];
}

void add_scalar_inplace(Tensor& y, float s) {
    const int n = y.size();
    float* yp = y.ptr();
    for (int i = 0; i < n; ++i) yp[i] += s;
}

// splitmix64 advanced by reference; deterministic, no external dep.
static inline uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline float u01(uint64_t& s) {
    // top 24 bits → [0, 1)
    return static_cast<float>((splitmix(s) >> 40)) / 16777216.0f;
}

void xavier_init(Tensor& W, uint64_t& rng_state) {
    // fan-in = cols, fan-out = rows ; limit = sqrt(6 / (in + out))
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    const int n = W.size();
    float* wp = W.ptr();
    for (int i = 0; i < n; ++i) {
        const float u = u01(rng_state);       // [0,1)
        wp[i] = (u * 2.0f - 1.0f) * limit;    // [-limit, +limit]
    }
}

} // namespace brogameagent::nn
