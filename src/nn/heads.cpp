#include "brogameagent/nn/heads.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

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

void ValueHead::to(Device d) {
    if (d == device_) return;
    device_require_cuda("ValueHead");
#ifdef BGA_HAS_CUDA
    fc1_.to(d);
    fc2_.to(d);
    if (d == Device::GPU) {
        h_raw_g_.resize(h_raw_.rows, 1);
        h_act_g_.resize(h_act_.rows, 1);
        pre_tanh_g_.resize(1, 1);
        post_tanh_g_.resize(1, 1);
        dValue_g_.resize(1, 1);
        dPre_g_.resize(1, 1);
        dHact_g_.resize(h_act_.rows, 1);
        dHraw_g_.resize(h_raw_.rows, 1);
    }
#endif
    device_ = d;
}

#ifdef BGA_HAS_CUDA
void ValueHead::forward(const gpu::GpuTensor& embed) {
    assert(device_ == Device::GPU);
    fc1_.forward(embed, h_raw_g_);
    gpu::relu_forward_gpu(h_raw_g_, h_act_g_);
    fc2_.forward(h_act_g_, pre_tanh_g_);
    gpu::tanh_forward_gpu(pre_tanh_g_, post_tanh_g_);
}

void ValueHead::backward(gpu::GpuTensor& dEmbed) {
    assert(device_ == Device::GPU);
    gpu::tanh_backward_gpu(post_tanh_g_, dValue_g_, dPre_g_);
    fc2_.backward(dPre_g_, dHact_g_);
    gpu::relu_backward_gpu(h_raw_g_, dHact_g_, dHraw_g_);
    fc1_.backward(dHraw_g_, dEmbed);
}
#endif

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

void FactoredPolicyHead::to(Device d) {
    if (d == device_) return;
    device_require_cuda("FactoredPolicyHead");
#ifdef BGA_HAS_CUDA
    move_.to(d); atk_.to(d); abil_.to(d);
    if (d == Device::GPU) {
        lm_g_.resize(N_MOVE,    1);
        la_g_.resize(N_ATTACK,  1);
        lb_g_.resize(N_ABILITY, 1);
        dLm_g_.resize(N_MOVE,    1);
        dLa_g_.resize(N_ATTACK,  1);
        dLb_g_.resize(N_ABILITY, 1);
        dEmbedTmp_g_.resize(move_.in_dim(), 1);
    }
#endif
    device_ = d;
}

#ifdef BGA_HAS_CUDA
void FactoredPolicyHead::forward(const gpu::GpuTensor& embed, gpu::GpuTensor& logits) {
    assert(device_ == Device::GPU);
    move_.forward(embed, lm_g_);
    atk_.forward(embed,  la_g_);
    abil_.forward(embed, lb_g_);
    std::vector<const gpu::GpuTensor*> parts{&lm_g_, &la_g_, &lb_g_};
    gpu::concat_rows_gpu(parts, logits);
}

void FactoredPolicyHead::backward(const gpu::GpuTensor& dLogits, gpu::GpuTensor& dEmbed) {
    assert(device_ == Device::GPU);
    std::vector<gpu::GpuTensor*> parts{&dLm_g_, &dLa_g_, &dLb_g_};
    gpu::split_rows_gpu(dLogits, parts);

    // dEmbed = move.backward + atk.backward + abil.backward
    move_.backward(dLm_g_, dEmbed);
    atk_.backward(dLa_g_, dEmbedTmp_g_);
    gpu::add_inplace_gpu(dEmbed, dEmbedTmp_g_);
    abil_.backward(dLb_g_, dEmbedTmp_g_);
    gpu::add_inplace_gpu(dEmbed, dEmbedTmp_g_);
}
#endif

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
