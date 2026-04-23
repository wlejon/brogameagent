#include "brogameagent/nn/heads.h"

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

void ValueHead::forward(const Tensor& embed, float& value) {
    fc1_.forward(embed, h_raw_);
    relu_forward(h_raw_, h_act_);
    fc2_.forward(h_act_, out_raw_);
    const float y = std::tanh(out_raw_[0]);
    y_cache_ = y;
    value = y;
}

void ValueHead::backward(float dValue, Tensor& dEmbed) {
    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_out_raw = dValue * (1.0f - y_cache_ * y_cache_);
    Tensor dOut = Tensor::vec(1);
    dOut[0] = d_out_raw;
    Tensor dHact = Tensor::vec(h_act_.size());
    fc2_.backward(dOut, dHact);
    Tensor dHraw = Tensor::vec(h_raw_.size());
    relu_backward(h_raw_, dHact, dHraw);
    fc1_.backward(dHraw, dEmbed);
}

// ─── FactoredPolicyHead ───────────────────────────────────────────────────

void FactoredPolicyHead::init(int embed_dim, uint64_t& rng_state) {
    move_.init(embed_dim, N_MOVE,    rng_state);
    atk_.init(embed_dim,  N_ATTACK,  rng_state);
    abil_.init(embed_dim, N_ABILITY, rng_state);
}

void FactoredPolicyHead::forward(const Tensor& embed, Tensor& logits) {
    assert(logits.size() == total_logits());
    Tensor lm = Tensor::vec(N_MOVE);
    Tensor la = Tensor::vec(N_ATTACK);
    Tensor lb = Tensor::vec(N_ABILITY);
    move_.forward(embed, lm);
    atk_.forward(embed, la);
    abil_.forward(embed, lb);
    std::memcpy(logits.ptr() + 0,                 lm.ptr(), N_MOVE    * sizeof(float));
    std::memcpy(logits.ptr() + N_MOVE,            la.ptr(), N_ATTACK  * sizeof(float));
    std::memcpy(logits.ptr() + N_MOVE + N_ATTACK, lb.ptr(), N_ABILITY * sizeof(float));
}

void FactoredPolicyHead::backward(const Tensor& dLogits, Tensor& dEmbed) {
    assert(dLogits.size() == total_logits());
    dEmbed.zero();

    Tensor dLm = Tensor::vec(N_MOVE);
    Tensor dLa = Tensor::vec(N_ATTACK);
    Tensor dLb = Tensor::vec(N_ABILITY);
    std::memcpy(dLm.ptr(), dLogits.ptr() + 0,                 N_MOVE    * sizeof(float));
    std::memcpy(dLa.ptr(), dLogits.ptr() + N_MOVE,            N_ATTACK  * sizeof(float));
    std::memcpy(dLb.ptr(), dLogits.ptr() + N_MOVE + N_ATTACK, N_ABILITY * sizeof(float));

    Tensor de = Tensor::vec(dEmbed.size());
    move_.backward(dLm, de);
    for (int i = 0; i < de.size(); ++i) dEmbed[i] += de[i];
    atk_.backward(dLa, de);
    for (int i = 0; i < de.size(); ++i) dEmbed[i] += de[i];
    abil_.backward(dLb, de);
    for (int i = 0; i < de.size(); ++i) dEmbed[i] += de[i];
}

static void softmax_slice(const float* logits, int n, float* probs, const float* mask) {
    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0.0f) continue;
        if (logits[i] > m) m = logits[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0.0f) { probs[i] = 0.0f; continue; }
        probs[i] = std::exp(logits[i] - m);
        s += probs[i];
    }
    const float inv = s > 0 ? 1.0f / s : 0.0f;
    for (int i = 0; i < n; ++i) probs[i] *= inv;
}

void factored_softmax(const Tensor& logits, Tensor& probs,
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
        if (mask && mask[i] == 0.0f) { dLogits[i] = 0.0f; continue; }
        if (target[i] > 0.0f) {
            const float p = probs[i] > 1e-12f ? probs[i] : 1e-12f;
            loss -= target[i] * std::log(p);
        }
        dLogits[i] = probs[i] - target[i];
    }
    return loss;
}

float factored_xent(const Tensor& logits,
                    const Tensor& move_target,
                    const Tensor& attack_target,
                    const Tensor& ability_target,
                    Tensor& probs, Tensor& dLogits,
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
