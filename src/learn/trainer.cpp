#include "brogameagent/learn/trainer.h"
#include "brogameagent/nn/heads.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cstring>

namespace brogameagent::learn {

static void fill_from_array(nn::Tensor& t, const float* src, int n) {
    assert(t.size() == n);
    std::memcpy(t.ptr(), src, n * sizeof(float));
}

TrainStep ExItTrainer::step() {
    TrainStep s;
    if (!net_ || !buf_ || buf_->size() == 0) return s;

    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    net_->zero_grad();

    nn::Tensor obs    = nn::Tensor::vec(observation::TOTAL);
    nn::Tensor logits = nn::Tensor::vec(net_->policy_logits());
    nn::Tensor probs  = nn::Tensor::vec(net_->policy_logits());
    nn::Tensor dLog   = nn::Tensor::vec(net_->policy_logits());

    nn::Tensor tgt_move   = nn::Tensor::vec(nn::FactoredPolicyHead::N_MOVE);
    nn::Tensor tgt_atk    = nn::Tensor::vec(nn::FactoredPolicyHead::N_ATTACK);
    nn::Tensor tgt_abil   = nn::Tensor::vec(nn::FactoredPolicyHead::N_ABILITY);

    float tot_loss_v = 0.0f, tot_loss_p = 0.0f;

    for (const auto& sit : batch) {
        fill_from_array(obs, sit.obs.data(), observation::TOTAL);
        fill_from_array(tgt_move, sit.target_move.data(),    nn::FactoredPolicyHead::N_MOVE);
        fill_from_array(tgt_atk,  sit.target_attack.data(),  nn::FactoredPolicyHead::N_ATTACK);
        fill_from_array(tgt_abil, sit.target_ability.data(), nn::FactoredPolicyHead::N_ABILITY);

        float v_pred = 0.0f;
        net_->forward(obs, v_pred, logits);

        // Value gradient.
        float dv = 0.0f;
        const float lv = nn::mse_scalar(v_pred, sit.value_target, dv);
        tot_loss_v += lv;

        // Policy gradient: combined factored softmax-xent.
        const float lp = nn::factored_xent(logits, tgt_move, tgt_atk, tgt_abil,
                                            probs, dLog,
                                            sit.atk_mask.data(),
                                            sit.abil_mask.data());
        tot_loss_p += lp;

        // Scale grads by batch-normalization + loss weights.
        const float scale_v = cfg_.value_weight  / static_cast<float>(batch.size());
        const float scale_p = cfg_.policy_weight / static_cast<float>(batch.size());
        for (int i = 0; i < dLog.size(); ++i) dLog[i] *= scale_p;

        net_->backward(dv * scale_v, dLog);
    }

    net_->sgd_step(cfg_.lr, cfg_.momentum);

    s.loss_value  = tot_loss_v / static_cast<float>(batch.size());
    s.loss_policy = tot_loss_p / static_cast<float>(batch.size());
    s.loss_total  = s.loss_value * cfg_.value_weight + s.loss_policy * cfg_.policy_weight;
    s.samples     = static_cast<int>(batch.size());

    ++steps_;
    if (cfg_.publish_every > 0 && (steps_ % cfg_.publish_every) == 0) maybe_publish();
    return s;
}

TrainStep ExItTrainer::step_n(int n) {
    TrainStep last;
    for (int i = 0; i < n; ++i) last = step();
    return last;
}

void ExItTrainer::maybe_publish() {
    if (!handle_ || !net_) return;
    auto blob = net_->save();
    handle_->publish(std::move(blob), static_cast<uint64_t>(steps_));
    ++publishes_;
}

} // namespace brogameagent::learn
