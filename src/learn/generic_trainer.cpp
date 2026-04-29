#include "brogameagent/learn/generic_trainer.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cstring>

namespace brogameagent::learn {

GenericTrainStep GenericExItTrainer::step() {
    GenericTrainStep s;
    if (!net_ || !buf_ || buf_->size() == 0) return s;

    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    const int in_dim   = net_->in_dim();
    const int n_act    = net_->num_actions();

    net_->zero_grad();

    nn::Tensor obs    = nn::Tensor::vec(in_dim);
    nn::Tensor logits = nn::Tensor::vec(n_act);
    nn::Tensor probs  = nn::Tensor::vec(n_act);
    nn::Tensor dLog   = nn::Tensor::vec(n_act);
    nn::Tensor target = nn::Tensor::vec(n_act);

    float tot_lv = 0.0f, tot_lp = 0.0f;

    for (const auto& sit : batch) {
        // Defensive: situations whose shape doesn't match the net are skipped.
        if (static_cast<int>(sit.obs.size()) != in_dim) continue;
        if (static_cast<int>(sit.policy_target.size()) != n_act) continue;

        std::memcpy(obs.ptr(),    sit.obs.data(),           in_dim * sizeof(float));
        std::memcpy(target.ptr(), sit.policy_target.data(), n_act  * sizeof(float));

        float v_pred = 0.0f;
        net_->forward(obs, v_pred, logits);

        // Value loss.
        float dv = 0.0f;
        const float lv = nn::mse_scalar(v_pred, sit.value_target, dv);
        tot_lv += lv;

        // Policy loss with optional mask. For multi-head nets, run an
        // independent softmax-xent per head segment and mean-reduce across
        // heads so loss magnitude is invariant to head count. For single-
        // head nets the loop runs once and is equivalent to the old call.
        const float* mask_ptr = nullptr;
        if (!sit.action_mask.empty() &&
            static_cast<int>(sit.action_mask.size()) == n_act) {
            mask_ptr = sit.action_mask.data();
        }
        const auto& offsets = net_->head_offsets();
        const int n_heads = net_->num_heads();
        float lp = 0.0f;
        for (int h = 0; h < n_heads; ++h) {
            const int off = offsets[h];
            const int len = offsets[h + 1] - off;
            const float* head_mask = mask_ptr ? mask_ptr + off : nullptr;
            const float lh = nn::softmax_xent_segment(
                logits.ptr() + off, target.ptr() + off,
                probs.ptr()  + off, dLog.ptr()   + off,
                len, head_mask);
            lp += lh;
        }
        lp /= static_cast<float>(n_heads);
        // Mean-reduce gradient too so per-head and overall scaling line up.
        if (n_heads > 1) {
            const float inv = 1.0f / static_cast<float>(n_heads);
            for (int i = 0; i < dLog.size(); ++i) dLog[i] *= inv;
        }
        tot_lp += lp;

        // Scale by 1/batch and the configured loss weights, then accumulate
        // into the net's grads via a single backward.
        const float scale_v = cfg_.value_weight  / static_cast<float>(batch.size());
        const float scale_p = cfg_.policy_weight / static_cast<float>(batch.size());
        for (int i = 0; i < dLog.size(); ++i) dLog[i] *= scale_p;

        net_->backward(dv * scale_v, dLog);
    }

    net_->sgd_step(cfg_.lr, cfg_.momentum);

    s.loss_value  = tot_lv / static_cast<float>(batch.size());
    s.loss_policy = tot_lp / static_cast<float>(batch.size());
    s.loss_total  = s.loss_value * cfg_.value_weight + s.loss_policy * cfg_.policy_weight;
    s.samples     = static_cast<int>(batch.size());

    ++steps_;
    if (cfg_.publish_every > 0 && (steps_ % cfg_.publish_every) == 0) maybe_publish();
    return s;
}

GenericTrainStep GenericExItTrainer::step_n(int n) {
    GenericTrainStep last;
    for (int i = 0; i < n; ++i) last = step();
    return last;
}

void GenericExItTrainer::maybe_publish() {
    if (!handle_ || !net_) return;
    auto blob = net_->save();
    handle_->publish(std::move(blob), static_cast<uint64_t>(steps_));
    ++publishes_;
}

} // namespace brogameagent::learn
