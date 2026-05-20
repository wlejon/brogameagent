#include "brogameagent/learn/generic_trainer.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cassert>
#include <cstring>

namespace brogameagent::learn {

// ─── INetForExIt adapters ──────────────────────────────────────────────────
//
// Thin per-net wrappers so the step paths can call into either net via a
// single virtual interface without templating the trainer.
namespace {

// Row-base pointer into a (B, W) row-major FP32 tensor.
inline void* row_ptr(const brotensor::Tensor& t, int row, int width) {
    return static_cast<char*>(t.data) +
           static_cast<size_t>(row) * width * sizeof(float);
}

class PolicyValueNetAdapter final : public INetForExIt {
public:
    explicit PolicyValueNetAdapter(nn::PolicyValueNet* n) : n_(n) {}
    int in_dim()      const override { return n_->in_dim(); }
    int num_actions() const override { return n_->num_actions(); }
    int num_heads()   const override { return n_->num_heads(); }
    const std::vector<int>& head_offsets() const override { return n_->head_offsets(); }
    void zero_grad() override { n_->zero_grad(); }
    void sgd_step(float lr, float m) override { n_->sgd_step(lr, m); }
    std::vector<uint8_t> save() const override { return n_->save(); }
    void forward(const brotensor::Tensor& x, float& v, brotensor::Tensor& l) override { n_->forward(x, v, l); }
    void backward(float dV, const brotensor::Tensor& dL) override { n_->backward(dV, dL); }
    void forward_batched_train(const brotensor::Tensor& X, brotensor::Tensor& L,
                               brotensor::Tensor& V) override {
        n_->forward_batched_train(X, L, V);
    }
    void backward_batched(const brotensor::Tensor& dL,
                          const brotensor::Tensor& dV) override {
        n_->backward_batched(dL, dV);
    }
private:
    nn::PolicyValueNet* n_;
};

// SingleHeroNetTX has no native batched kernels — its batched-train API is a
// per-sample loop over the single-sample forward/backward, holding row views
// of the (B, *) staging tensors. forward()/backward() dispatch by device, so
// the loop is device-neutral.
class SingleHeroNetTXAdapter final : public INetForExIt {
public:
    explicit SingleHeroNetTXAdapter(nn::SingleHeroNetTX* n) : n_(n) {}
    int in_dim()      const override { return n_->in_dim(); }
    int num_actions() const override { return n_->num_actions(); }
    int num_heads()   const override { return n_->num_heads(); }
    const std::vector<int>& head_offsets() const override { return n_->head_offsets(); }
    void zero_grad() override { n_->zero_grad(); }
    void sgd_step(float lr, float m) override { n_->sgd_step(lr, m); }
    std::vector<uint8_t> save() const override { return n_->save(); }
    void forward(const brotensor::Tensor& x, float& v, brotensor::Tensor& l) override { n_->forward(x, v, l); }
    void backward(float dV, const brotensor::Tensor& dL) override { n_->backward(dV, dL); }

    void forward_batched_train(const brotensor::Tensor& X, brotensor::Tensor& L,
                               brotensor::Tensor& V) override {
        const int B    = X.rows;
        const int din  = X.cols;
        const int lw   = n_->num_actions();
        if (L.rows != B || L.cols != lw) L.resize(B, lw);
        brotensor::Tensor Vh = brotensor::Tensor::mat(B, 1);
        last_X_ = &X;
        for (int b = 0; b < B; ++b) {
            brotensor::Tensor xv = brotensor::Tensor::view(X.device, row_ptr(X, b, din), din, 1);
            brotensor::Tensor lv = brotensor::Tensor::view(L.device, row_ptr(L, b, lw), lw, 1);
            float v = 0.0f;
            n_->forward(xv, v, lv);
            Vh.ptr()[b] = v;
        }
        V = Vh.to(V.device);
    }

    void backward_batched(const brotensor::Tensor& dL,
                          const brotensor::Tensor& dV) override {
        assert(last_X_ != nullptr &&
               "forward_batched_train must precede backward_batched");
        const brotensor::Tensor& X = *last_X_;
        const int B   = X.rows;
        const int din = X.cols;
        const int lw  = n_->num_actions();
        const brotensor::Tensor dVh = dV.to(brotensor::Device::CPU);
        brotensor::Tensor scratch;  // re-forward logits output, unused
        for (int b = 0; b < B; ++b) {
            brotensor::Tensor xv = brotensor::Tensor::view(X.device, row_ptr(X, b, din), din, 1);
            float v = 0.0f;
            n_->forward(xv, v, scratch);   // re-prime per-sample caches
            brotensor::Tensor dlv = brotensor::Tensor::view(dL.device, row_ptr(dL, b, lw), lw, 1);
            n_->backward(dVh.ptr()[b], dlv);
        }
    }
private:
    nn::SingleHeroNetTX*     n_;
    const brotensor::Tensor* last_X_ = nullptr;
};

} // namespace

void GenericExItTrainer::set_net(nn::PolicyValueNet* net) {
    if (!net) { net_ = nullptr; net_owned_.reset(); return; }
    net_owned_ = std::make_unique<PolicyValueNetAdapter>(net);
    net_ = net_owned_.get();
}

void GenericExItTrainer::set_net(nn::SingleHeroNetTX* net) {
    if (!net) { net_ = nullptr; net_owned_.reset(); return; }
    net_owned_ = std::make_unique<SingleHeroNetTXAdapter>(net);
    net_ = net_owned_.get();
}

GenericTrainStep GenericExItTrainer::step() {
    if (!net_ || !buf_ || buf_->size() == 0) return {};
    if (cfg_.device == brotensor::Device::CPU) return step_cpu_();
    return step_batched_();
}

GenericTrainStep GenericExItTrainer::step_cpu_() {
    GenericTrainStep s;
    if (!net_ || !buf_ || buf_->size() == 0) return s;

    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    const int in_dim   = net_->in_dim();
    const int n_act    = net_->num_actions();

    net_->zero_grad();

    brotensor::Tensor obs    = brotensor::Tensor::vec(in_dim);
    brotensor::Tensor logits = brotensor::Tensor::vec(n_act);
    brotensor::Tensor probs  = brotensor::Tensor::vec(n_act);
    brotensor::Tensor dLog   = brotensor::Tensor::vec(n_act);
    brotensor::Tensor target = brotensor::Tensor::vec(n_act);

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
        const float lv = brotensor::mse_scalar(v_pred, sit.value_target, dv);
        tot_lv += lv;

        // Policy loss with optional mask. For multi-head nets, run an
        // independent softmax-xent per head segment and mean-reduce across
        // heads so loss magnitude is invariant to head count.
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
            const float lh = brotensor::softmax_xent_segment(
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

void GenericExItTrainer::ensure_staging_() {
    if (staging_ready_) return;
    const int B      = cfg_.batch;
    const int in_dim = net_->in_dim();
    const int n_act  = net_->num_actions();
    const brotensor::Device d = cfg_.device;
    X_BD_         = brotensor::Tensor::zeros_on(d, B, in_dim);
    T_BL_         = brotensor::Tensor::zeros_on(d, B, n_act);
    M_BL_         = brotensor::Tensor::zeros_on(d, B, n_act);
    V_B1_         = brotensor::Tensor::zeros_on(d, B, 1);
    logits_BL_    = brotensor::Tensor::zeros_on(d, B, n_act);
    values_B1_    = brotensor::Tensor::zeros_on(d, B, 1);
    probs_BL_     = brotensor::Tensor::zeros_on(d, B, n_act);
    dLog_BL_      = brotensor::Tensor::zeros_on(d, B, n_act);
    dV_B1_        = brotensor::Tensor::zeros_on(d, B, 1);
    lp_per_sample_ = brotensor::Tensor::zeros_on(d, B, 1);
    lv_per_sample_ = brotensor::Tensor::zeros_on(d, B, 1);

    // Upload head_offsets into a small device int buffer (INT32 carrier).
    const auto& offsets = net_->head_offsets();
    const int n = static_cast<int>(offsets.size());
    brotensor::Tensor offs_h =
        brotensor::Tensor::zeros_on(brotensor::Device::CPU, n, 1, brotensor::Dtype::INT32);
    int* op = static_cast<int*>(offs_h.host_raw_mut());
    for (int i = 0; i < n; ++i) op[i] = offsets[i];
    head_offsets_dev_ = offs_h.to(d);

    staging_ready_ = true;
}

GenericTrainStep GenericExItTrainer::step_batched_() {
    GenericTrainStep s;
    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    const int B       = cfg_.batch;
    const int in_dim  = net_->in_dim();
    const int n_act   = net_->num_actions();
    const int n_heads = net_->num_heads();
    ensure_staging_();

    // Stage minibatch on host into contiguous (B, *) buffers. Defensive: any
    // shape-mismatched situation contributes zeros (target = uniform, mask
    // = 1s, value = 0) so we always have exactly B rows.
    brotensor::Tensor X_h = brotensor::Tensor::mat(B, in_dim);
    brotensor::Tensor T_h = brotensor::Tensor::mat(B, n_act);
    brotensor::Tensor M_h = brotensor::Tensor::mat(B, n_act);
    brotensor::Tensor V_h = brotensor::Tensor::mat(B, 1);

    bool any_mask = false;
    int valid = 0;
    for (int b = 0; b < B; ++b) {
        const auto& sit = batch[b];
        if (static_cast<int>(sit.obs.size()) != in_dim) continue;
        if (static_cast<int>(sit.policy_target.size()) != n_act) continue;
        std::memcpy(X_h.ptr() + static_cast<size_t>(b) * in_dim,
                    sit.obs.data(), in_dim * sizeof(float));
        std::memcpy(T_h.ptr() + static_cast<size_t>(b) * n_act,
                    sit.policy_target.data(), n_act * sizeof(float));
        V_h.ptr()[b] = sit.value_target;
        if (!sit.action_mask.empty() &&
            static_cast<int>(sit.action_mask.size()) == n_act) {
            std::memcpy(M_h.ptr() + static_cast<size_t>(b) * n_act,
                        sit.action_mask.data(), n_act * sizeof(float));
            any_mask = true;
        } else {
            // No-mask sentinel: 1s so the kernel treats every entry as valid.
            float* row = M_h.ptr() + static_cast<size_t>(b) * n_act;
            for (int i = 0; i < n_act; ++i) row[i] = 1.0f;
        }
        ++valid;
    }
    if (valid == 0) return s;

    X_BD_ = X_h.to(cfg_.device);
    T_BL_ = T_h.to(cfg_.device);
    V_B1_ = V_h.to(cfg_.device);
    const float* mask_dev = nullptr;
    if (any_mask) {
        M_BL_ = M_h.to(cfg_.device);
        mask_dev = static_cast<const float*>(M_BL_.data);
    }

    net_->zero_grad();

    // Forward (training) — caches every layer's batched activation.
    net_->forward_batched_train(X_BD_, logits_BL_, values_B1_);

    // Per-sample value MSE: dV_B1_ = (pred - target), lv_per_sample_ = 0.5*d^2.
    brotensor::mse_vec_per_sample(values_B1_, V_B1_, dV_B1_, lv_per_sample_);

    // Per-(sample, head) softmax-xent: probs, dLog = (p - t) on valid,
    // lp_per_sample = sum-over-heads of head xent.
    brotensor::softmax_xent_fused_batched(
        logits_BL_, T_BL_, mask_dev,
        static_cast<const int*>(head_offsets_dev_.data), n_heads,
        probs_BL_, dLog_BL_, lp_per_sample_);

    // Scale gradients. Match the CPU path exactly:
    //   dV   *= value_weight  / B
    //   dLog *= policy_weight / B / n_heads
    const float B_f = static_cast<float>(B);
    brotensor::scale_inplace(dV_B1_, cfg_.value_weight / B_f);
    brotensor::scale_inplace(dLog_BL_,
                             cfg_.policy_weight / B_f / static_cast<float>(n_heads));

    // Backward + optimizer.
    net_->backward_batched(dLog_BL_, dV_B1_);
    net_->sgd_step(cfg_.lr, cfg_.momentum);

    // Drain the device queue, then read the two per-sample loss vectors back.
    brotensor::sync(cfg_.device);
    const brotensor::Tensor lv_h = lv_per_sample_.to(brotensor::Device::CPU);
    const brotensor::Tensor lp_h = lp_per_sample_.to(brotensor::Device::CPU);

    float tot_lv = 0.0f, tot_lp = 0.0f;
    for (int b = 0; b < B; ++b) tot_lv += lv_h[b];
    for (int b = 0; b < B; ++b) tot_lp += lp_h[b];
    // CPU mean-reduces per-sample policy loss across heads.
    tot_lp /= static_cast<float>(n_heads);

    s.loss_value  = tot_lv / B_f;
    s.loss_policy = tot_lp / B_f;
    s.loss_total  = s.loss_value * cfg_.value_weight + s.loss_policy * cfg_.policy_weight;
    s.samples     = valid;

    ++steps_;
    if (cfg_.publish_every > 0 && (steps_ % cfg_.publish_every) == 0) maybe_publish();
    return s;
}

} // namespace brogameagent::learn
