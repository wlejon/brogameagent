#include "brogameagent/learn/generic_trainer.h"
#include "brogameagent/nn/ops.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

#include <cassert>
#include <cstring>

namespace brogameagent::learn {

GenericTrainStep GenericExItTrainer::step() {
    if (!net_ || !buf_ || buf_->size() == 0) return {};
#ifdef BGA_HAS_CUDA
    if (cfg_.device == nn::Device::GPU) {
        return step_gpu_();
    }
#endif
    return step_cpu_();
}

GenericTrainStep GenericExItTrainer::step_cpu_() {
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

#ifdef BGA_HAS_CUDA
void GenericExItTrainer::ensure_gpu_staging_() {
    if (gpu_ready_) return;
    const int in_dim = net_->in_dim();
    const int n_act  = net_->num_actions();
    obs_g_.resize(in_dim, 1);
    logits_g_.resize(n_act, 1);
    probs_g_.resize(n_act, 1);
    dLog_g_.resize(n_act, 1);
    dLog_acc_g_.resize(n_act, 1);
    target_g_.resize(n_act, 1);
    mask_g_.resize(n_act, 1);
    v_tgt_g_.resize(1, 1);
    dV_acc_g_.resize(1, 1);
    gpu_ready_ = true;
}

GenericTrainStep GenericExItTrainer::step_gpu_() {
    GenericTrainStep s;
    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    const int in_dim   = net_->in_dim();
    const int n_act    = net_->num_actions();
    ensure_gpu_staging_();

    net_->zero_grad();

    // Per-sample staging on host so we can upload contiguous floats. We
    // re-use the host vectors across samples to avoid alloc churn.
    nn::Tensor obs_h    = nn::Tensor::vec(in_dim);
    nn::Tensor target_h = nn::Tensor::vec(n_act);
    nn::Tensor mask_h   = nn::Tensor::vec(n_act);
    nn::Tensor v_tgt_h  = nn::Tensor::vec(1);
    nn::Tensor v_pred_h = nn::Tensor::vec(1);

    float tot_lv = 0.0f, tot_lp = 0.0f;
    int samples = 0;

    const auto& offsets = net_->head_offsets();
    const int n_heads = net_->num_heads();

    for (const auto& sit : batch) {
        if (static_cast<int>(sit.obs.size()) != in_dim) continue;
        if (static_cast<int>(sit.policy_target.size()) != n_act) continue;

        std::memcpy(obs_h.ptr(),    sit.obs.data(),           in_dim * sizeof(float));
        std::memcpy(target_h.ptr(), sit.policy_target.data(), n_act  * sizeof(float));
        nn::gpu::upload(obs_h,    obs_g_);
        nn::gpu::upload(target_h, target_g_);

        const bool has_mask = !sit.action_mask.empty() &&
                              static_cast<int>(sit.action_mask.size()) == n_act;
        const float* mask_dev = nullptr;
        if (has_mask) {
            std::memcpy(mask_h.ptr(), sit.action_mask.data(), n_act * sizeof(float));
            nn::gpu::upload(mask_h, mask_g_);
            mask_dev = mask_g_.data;
        }

        // Forward.
        net_->forward(obs_g_, logits_g_);

        // Value loss / grad — host roundtrip to keep CPU-identical 0.5*d^2
        // semantics. Sync once to read v_pred.
        nn::gpu::download(net_->value_gpu(), v_pred_h);
        nn::gpu::cuda_sync();
        const float v_pred = v_pred_h[0];
        float dv = 0.0f;
        const float lv = nn::mse_scalar(v_pred, sit.value_target, dv);
        tot_lv += lv;
        const float scale_v = cfg_.value_weight  / static_cast<float>(batch.size());
        v_pred_h[0] = dv * scale_v;
        nn::gpu::upload(v_pred_h, net_->dValue_gpu());

        // Policy loss / grad — per-head softmax-xent, mean-reduced over heads.
        // softmax_xent_fused_gpu writes (probs - target) into dLog at the
        // segment slice; we accumulate into dLog_g_ across heads via a
        // segment-wise call. Since each head writes to its own non-overlapping
        // segment, we can write directly to dLog_g_ by copying logit/target
        // segments out into temporary slices — but that requires extra ops.
        // Simpler and within spec: call the fused kernel per head against
        // FULL-WIDTH logits / target / probs / dLog tensors, but with each
        // head's slice masked in. We achieve that by constructing a per-head
        // mask: zero-everywhere except the [off, off+len) range, optionally
        // AND-ed with the user mask.
        //
        // To avoid that complexity (and an extra mask upload per head), use
        // a single non-segmented call when n_heads == 1, and a segment-wise
        // call walking GpuTensor::view() for each slice.
        float lp = 0.0f;
        // dLog_g_ is the full-width gradient; per-head writes cover the
        // entire [0, n_act) range since head segments are contiguous, so
        // no pre-zeroing is required.
        for (int h = 0; h < n_heads; ++h) {
            const int off = offsets[h];
            const int len = offsets[h + 1] - off;
            // Slice device pointers into per-head views.
            nn::gpu::GpuTensor logits_slice =
                nn::gpu::GpuTensor::view(logits_g_.data + off, len, 1);
            nn::gpu::GpuTensor target_slice =
                nn::gpu::GpuTensor::view(target_g_.data + off, len, 1);
            nn::gpu::GpuTensor probs_slice =
                nn::gpu::GpuTensor::view(probs_g_.data + off, len, 1);
            nn::gpu::GpuTensor dLog_slice =
                nn::gpu::GpuTensor::view(dLog_g_.data + off, len, 1);
            const float* head_mask = mask_dev ? (mask_dev + off) : nullptr;
            const float lh = nn::gpu::softmax_xent_fused_gpu(
                logits_slice, target_slice, head_mask,
                probs_slice, dLog_slice);
            lp += lh;
        }
        lp /= static_cast<float>(n_heads);
        tot_lp += lp;

        // Scale dLog by (1/B) * policy_weight and the per-head mean factor.
        const float inv_n_heads = (n_heads > 1)
            ? (1.0f / static_cast<float>(n_heads)) : 1.0f;
        const float scale_p = cfg_.policy_weight / static_cast<float>(batch.size())
                              * inv_n_heads;
        // Multiply dLog_g_ in-place by scale_p. We reuse sgd_step_gpu? No —
        // need a generic scale. Use a tiny add_scalar trick? No, that adds.
        // Simpler: do the scaling on host: download, scale, upload. Cheap
        // for n_act-sized vectors; matches the value-loss host-roundtrip
        // pattern.
        nn::Tensor dLog_h = nn::Tensor::vec(n_act);
        nn::gpu::download(dLog_g_, dLog_h);
        nn::gpu::cuda_sync();
        for (int i = 0; i < n_act; ++i) dLog_h[i] *= scale_p;
        nn::gpu::upload(dLog_h, dLog_g_);

        // Backward: PolicyValueNet reads dValue from net->dValue_gpu() (we
        // already wrote to it above) and dLogits from dLog_g_.
        net_->backward(dLog_g_);

        ++samples;
    }

    net_->sgd_step(cfg_.lr, cfg_.momentum);

    s.loss_value  = tot_lv / static_cast<float>(batch.size());
    s.loss_policy = tot_lp / static_cast<float>(batch.size());
    s.loss_total  = s.loss_value * cfg_.value_weight + s.loss_policy * cfg_.policy_weight;
    s.samples     = samples;

    ++steps_;
    if (cfg_.publish_every > 0 && (steps_ % cfg_.publish_every) == 0) maybe_publish();
    return s;
}
#endif

} // namespace brogameagent::learn
