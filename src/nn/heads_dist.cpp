#include "brogameagent/nn/heads_dist.h"

#include <cassert>
#include <cmath>

namespace brogameagent::nn {

void DistributionalValueHead::init(int embed_dim, int hidden, int K, uint64_t& rng_state) {
    K_ = K;
    fc1_.init(embed_dim, hidden, rng_state);
    fc2_.init(hidden, K, rng_state);
    h_raw_.resize(hidden, 1);
    h_act_.resize(hidden, 1);
    logits_.resize(K, 1);
}

void DistributionalValueHead::forward(const Tensor& embed, Tensor& probs, float& value) {
    assert(probs.size() == K_);
    fc1_.forward(embed, h_raw_);
    relu_forward(h_raw_, h_act_);
    fc2_.forward(h_act_, logits_);
    softmax_forward(logits_, probs, nullptr);
    float v = 0.0f;
    for (int i = 0; i < K_; ++i) v += probs[i] * support(i);
    value = v;
}

float DistributionalValueHead::xent_backward(const Tensor& probs, const Tensor& p_target, Tensor& dEmbed) {
    assert(probs.size() == K_ && p_target.size() == K_);
    float loss = 0.0f;
    Tensor dLogits = Tensor::vec(K_);
    for (int i = 0; i < K_; ++i) {
        if (p_target[i] > 0.0f) {
            const float p = probs[i] > 1e-12f ? probs[i] : 1e-12f;
            loss -= p_target[i] * std::log(p);
        }
        dLogits[i] = probs[i] - p_target[i];
    }
    Tensor dHact = Tensor::vec(h_act_.size());
    fc2_.backward(dLogits, dHact);
    Tensor dHraw = Tensor::vec(h_raw_.size());
    relu_backward(h_raw_, dHact, dHraw);
    fc1_.backward(dHraw, dEmbed);
    return loss;
}

void DistributionalValueHead::project_target(float z, Tensor& out) const {
    assert(out.size() == K_);
    out.zero();
    // Clamp to [-1, 1].
    if (z < -1.0f) z = -1.0f;
    if (z >  1.0f) z =  1.0f;
    // Continuous bin index in [0, K-1].
    const float b = (z + 1.0f) * 0.5f * static_cast<float>(K_ - 1);
    const int lo = static_cast<int>(std::floor(b));
    const int hi = static_cast<int>(std::ceil(b));
    if (lo == hi) {
        out[lo] = 1.0f;
    } else {
        const float w_hi = b - static_cast<float>(lo);
        const float w_lo = 1.0f - w_hi;
        out[lo] = w_lo;
        out[hi] = w_hi;
    }
}

} // namespace brogameagent::nn
