#include "brogameagent/nn/forward_model.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void ForwardModelHead::init(int embed_dim, int hidden, uint64_t& rng_state) {
    embed_dim_ = embed_dim;
    const int in_dim = embed_dim + ACTION_DIM;
    fc1_.init(in_dim, hidden, rng_state);
    fc2_.init(hidden, embed_dim, rng_state);
    input_cat_.resize(in_dim, 1);
    h_raw_.resize(hidden, 1);
    h_act_.resize(hidden, 1);
}

void ForwardModelHead::forward(const Tensor& embed, const Tensor& action, Tensor& pred_next) {
    assert(embed.size() == embed_dim_);
    assert(action.size() == ACTION_DIM);
    assert(pred_next.size() == embed_dim_);
    std::memcpy(input_cat_.ptr(),                embed.ptr(),  embed_dim_ * sizeof(float));
    std::memcpy(input_cat_.ptr() + embed_dim_,   action.ptr(), ACTION_DIM * sizeof(float));
    fc1_.forward(input_cat_, h_raw_);
    relu_forward(h_raw_, h_act_);
    fc2_.forward(h_act_, pred_next);
}

void ForwardModelHead::backward(const Tensor& dPred, Tensor& dEmbed) {
    assert(dPred.size() == embed_dim_);
    assert(dEmbed.size() == embed_dim_);
    Tensor dHact = Tensor::vec(h_act_.size());
    fc2_.backward(dPred, dHact);
    Tensor dHraw = Tensor::vec(h_raw_.size());
    relu_backward(h_raw_, dHact, dHraw);
    Tensor dInput = Tensor::vec(input_cat_.size());
    fc1_.backward(dHraw, dInput);
    // Split: first embed_dim_ to dEmbed, remainder (action) is discarded.
    std::memcpy(dEmbed.ptr(), dInput.ptr(), embed_dim_ * sizeof(float));
}

void build_action_onehot(int move_idx, int attack_idx, int ability_idx, Tensor& out) {
    assert(out.size() == ForwardModelHead::ACTION_DIM);
    out.zero();
    if (move_idx    >= 0 && move_idx    < ForwardModelHead::N_MOVE)    out[move_idx] = 1.0f;
    if (attack_idx  >= 0 && attack_idx  < ForwardModelHead::N_ATTACK)  out[ForwardModelHead::N_MOVE + attack_idx] = 1.0f;
    if (ability_idx >= 0 && ability_idx < ForwardModelHead::N_ABILITY) out[ForwardModelHead::N_MOVE + ForwardModelHead::N_ATTACK + ability_idx] = 1.0f;
}

float spr_loss(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    assert(pred.size() == target.size() && dPred.size() == pred.size());
    float l = 0.0f;
    for (int i = 0; i < pred.size(); ++i) {
        const float d = pred[i] - target[i];
        l += 0.5f * d * d;
        dPred[i] = d;
    }
    return l;
}

} // namespace brogameagent::nn
