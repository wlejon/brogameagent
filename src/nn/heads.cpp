#include "brogameagent/nn/heads.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cmath>
#include <cstring>

namespace brogameagent::nn {

// ─── ValueHead ────────────────────────────────────────────────────────────

void ValueHead::init(int embed_dim, int hidden, uint64_t& rng_state) {
    fc1_.init(embed_dim, hidden, rng_state);
    fc2_.init(hidden, 1, rng_state);
    h_raw_.resize(hidden, 1);
    h_act_.resize(hidden, 1);
    out_raw_.resize(1, 1);
}

void ValueHead::forward(const brotensor::Tensor& embed, float& value) {
    fc1_.forward(embed, h_raw_);
    brotensor::relu_forward(h_raw_, h_act_);
    fc2_.forward(h_act_, out_raw_);
    // out_raw_ is a 1-element tensor on the parameter device; stage it to host
    // to read the scalar (host accessors throw on a device tensor).
    const float pre = (device_ == brotensor::Device::CPU)
                          ? out_raw_[0]
                          : out_raw_.to(brotensor::Device::CPU)[0];
    const float y = std::tanh(pre);
    y_cache_ = y;
    value = y;
}

void ValueHead::to(brotensor::Device d) {
    if (d == device_) return;
    fc1_.to(d);
    fc2_.to(d);
    h_raw_   = h_raw_.to(d);
    h_act_   = h_act_.to(d);
    out_raw_ = out_raw_.to(d);
    device_  = d;
}

void ValueHead::backward(float dValue, brotensor::Tensor& dEmbed) {
    // d/dx tanh(x) = 1 - tanh(x)^2. dOut is a host-staged 1-vector (the grad
    // is a host scalar) migrated to the parameter device so the downstream
    // Linear backward ops dispatch consistently.
    const float d_out_raw = dValue * (1.0f - y_cache_ * y_cache_);
    brotensor::Tensor dOut = brotensor::Tensor::vec(1);
    dOut[0] = d_out_raw;
    dOut = dOut.to(device_);
    brotensor::Tensor dHact = brotensor::Tensor::zeros_on(device_, h_act_.size(), 1);
    fc2_.backward(dOut, dHact);
    brotensor::Tensor dHraw = brotensor::Tensor::zeros_on(device_, h_raw_.size(), 1);
    brotensor::relu_backward(h_raw_, dHact, dHraw);
    fc1_.backward(dHraw, dEmbed);
}

// ─── FactoredPolicyHead ───────────────────────────────────────────────────

void FactoredPolicyHead::init(int embed_dim, uint64_t& rng_state) {
    move_.init(embed_dim, N_MOVE,    rng_state);
    atk_.init(embed_dim,  N_ATTACK,  rng_state);
    abil_.init(embed_dim, N_ABILITY, rng_state);
}

void FactoredPolicyHead::to(brotensor::Device d) {
    if (d == device_) return;
    move_.to(d); atk_.to(d); abil_.to(d);
    device_ = d;
}

void FactoredPolicyHead::forward(const brotensor::Tensor& embed, brotensor::Tensor& logits) {
    assert(logits.size() == total_logits());
    brotensor::Tensor lm = brotensor::Tensor::vec(N_MOVE).to(device_);
    brotensor::Tensor la = brotensor::Tensor::vec(N_ATTACK).to(device_);
    brotensor::Tensor lb = brotensor::Tensor::vec(N_ABILITY).to(device_);
    move_.forward(embed, lm);
    atk_.forward(embed, la);
    abil_.forward(embed, lb);
    std::vector<const brotensor::Tensor*> parts{&lm, &la, &lb};
    brotensor::concat_rows(parts, logits);
}

void FactoredPolicyHead::backward(const brotensor::Tensor& dLogits, brotensor::Tensor& dEmbed) {
    assert(dLogits.size() == total_logits());
    dEmbed.zero();

    brotensor::Tensor dLm = brotensor::Tensor::vec(N_MOVE).to(device_);
    brotensor::Tensor dLa = brotensor::Tensor::vec(N_ATTACK).to(device_);
    brotensor::Tensor dLb = brotensor::Tensor::vec(N_ABILITY).to(device_);
    std::vector<brotensor::Tensor*> parts{&dLm, &dLa, &dLb};
    brotensor::split_rows(dLogits, parts);

    // dEmbed = move.backward + atk.backward + abil.backward
    brotensor::Tensor de = brotensor::Tensor::vec(dEmbed.size()).to(device_);
    move_.backward(dLm, dEmbed);
    atk_.backward(dLa, de);
    brotensor::add_inplace(dEmbed, de);
    abil_.backward(dLb, de);
    brotensor::add_inplace(dEmbed, de);
}

static void softmax_slice(const float* logits, int n, float* probs, const float* mask) {
    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        if (logits[i] > m) m = logits[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) { probs[i] = 0.0f; continue; }
        probs[i] = std::exp(logits[i] - m);
        s += probs[i];
    }
    const float inv = s > 0 ? 1.0f / s : 0.0f;
    for (int i = 0; i < n; ++i) probs[i] *= inv;
}

void factored_softmax(const brotensor::Tensor& logits, brotensor::Tensor& probs,
                      const float* attack_mask, const float* ability_mask) {
    const int N_MOVE = FactoredPolicyHead::N_MOVE;
    const int N_ATK  = FactoredPolicyHead::N_ATTACK;
    const int N_AB   = FactoredPolicyHead::N_ABILITY;
    softmax_slice(logits.ptr() + 0, N_MOVE, probs.ptr() + 0, nullptr);

    // Compose attack mask: first N_ATK-1 entries from input mask; last entry (no-op) always 1.
    float amask[FactoredPolicyHead::N_ATTACK];
    for (int i = 0; i < N_ATK - 1; ++i) amask[i] = attack_mask ? attack_mask[i] : 1.0f;
    amask[N_ATK - 1] = 1.0f;
    softmax_slice(logits.ptr() + N_MOVE, N_ATK, probs.ptr() + N_MOVE, amask);

    float bmask[FactoredPolicyHead::N_ABILITY];
    for (int i = 0; i < N_AB - 1; ++i) bmask[i] = ability_mask ? ability_mask[i] : 1.0f;
    bmask[N_AB - 1] = 1.0f;
    softmax_slice(logits.ptr() + N_MOVE + N_ATK, N_AB, probs.ptr() + N_MOVE + N_ATK, bmask);
}

static float xent_slice(const float* logits, int n, const float* target,
                        float* probs, float* dLogits, const float* mask) {
    softmax_slice(logits, n, probs, mask);
    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) { dLogits[i] = 0.0f; continue; }
        if (target[i] > 0.0f) {
            const float p = probs[i] > 1e-12f ? probs[i] : 1e-12f;
            loss -= target[i] * std::log(p);
        }
        dLogits[i] = probs[i] - target[i];
    }
    return loss;
}

float factored_xent(const brotensor::Tensor& logits,
                    const brotensor::Tensor& move_target,
                    const brotensor::Tensor& attack_target,
                    const brotensor::Tensor& ability_target,
                    brotensor::Tensor& probs, brotensor::Tensor& dLogits,
                    const float* attack_mask, const float* ability_mask) {
    const int N_MOVE = FactoredPolicyHead::N_MOVE;
    const int N_ATK  = FactoredPolicyHead::N_ATTACK;
    const int N_AB   = FactoredPolicyHead::N_ABILITY;

    float loss = 0.0f;
    loss += xent_slice(logits.ptr() + 0, N_MOVE, move_target.ptr(),
                       probs.ptr() + 0, dLogits.ptr() + 0, nullptr);

    float amask[FactoredPolicyHead::N_ATTACK];
    for (int i = 0; i < N_ATK - 1; ++i) amask[i] = attack_mask ? attack_mask[i] : 1.0f;
    amask[N_ATK - 1] = 1.0f;
    loss += xent_slice(logits.ptr() + N_MOVE, N_ATK, attack_target.ptr(),
                       probs.ptr() + N_MOVE, dLogits.ptr() + N_MOVE, amask);

    float bmask[FactoredPolicyHead::N_ABILITY];
    for (int i = 0; i < N_AB - 1; ++i) bmask[i] = ability_mask ? ability_mask[i] : 1.0f;
    bmask[N_AB - 1] = 1.0f;
    loss += xent_slice(logits.ptr() + N_MOVE + N_ATK, N_AB, ability_target.ptr(),
                       probs.ptr() + N_MOVE + N_ATK, dLogits.ptr() + N_MOVE + N_ATK, bmask);

    return loss;
}

} // namespace brogameagent::nn
