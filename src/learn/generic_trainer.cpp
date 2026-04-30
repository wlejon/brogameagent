#include "brogameagent/learn/generic_trainer.h"
#include "brogameagent/nn/ops.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#include <cuda_runtime.h>
#endif

#include <cassert>
#include <cstring>

namespace brogameagent::learn {

// ─── INetForExIt adapters ──────────────────────────────────────────────────
//
// Thin per-net wrappers so step_cpu_/step_gpu_ can call into either net via
// a single virtual interface without templating the trainer.
namespace {

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
    void forward(const nn::Tensor& x, float& v, nn::Tensor& l) override { n_->forward(x, v, l); }
    void backward(float dV, const nn::Tensor& dL) override { n_->backward(dV, dL); }
#ifdef BGA_HAS_CUDA
    void forward_batched_train(const nn::gpu::GpuTensor& X, nn::gpu::GpuTensor& L,
                               nn::gpu::GpuTensor& V) override {
        n_->forward_batched_train(X, L, V);
    }
    void backward_batched(const nn::gpu::GpuTensor& dL,
                          const nn::gpu::GpuTensor& dV) override {
        n_->backward_batched(dL, dV);
    }
#endif
private:
    nn::PolicyValueNet* n_;
};

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
    void forward(const nn::Tensor& x, float& v, nn::Tensor& l) override { n_->forward(x, v, l); }
    void backward(float dV, const nn::Tensor& dL) override { n_->backward(dV, dL); }
#ifdef BGA_HAS_CUDA
    void forward_batched_train(const nn::gpu::GpuTensor& X, nn::gpu::GpuTensor& L,
                               nn::gpu::GpuTensor& V) override {
        n_->forward_batched_train(X, L, V);
    }
    void backward_batched(const nn::gpu::GpuTensor& dL,
                          const nn::gpu::GpuTensor& dV) override {
        n_->backward_batched(dL, dV);
    }
#endif
private:
    nn::SingleHeroNetTX* n_;
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

#ifdef BGA_HAS_CUDA
GenericExItTrainer::~GenericExItTrainer() {
    if (head_offsets_dev_) {
        cudaFree(head_offsets_dev_);
        head_offsets_dev_ = nullptr;
    }
}
#endif

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
    const int B      = cfg_.batch;
    const int in_dim = net_->in_dim();
    const int n_act  = net_->num_actions();
    X_BD_g_.resize(B, in_dim);
    T_BL_g_.resize(B, n_act);
    M_BL_g_.resize(B, n_act);
    V_B1_g_.resize(B, 1);
    logits_BL_g_.resize(B, n_act);
    values_B1_g_.resize(B, 1);
    probs_BL_g_.resize(B, n_act);
    dLog_BL_g_.resize(B, n_act);
    dV_B1_g_.resize(B, 1);
    lp_per_sample_g_.resize(B, 1);
    lv_per_sample_g_.resize(B, 1);

    // Upload head_offsets into a small device int buffer (owned by the
    // trainer; freed in the destructor).
    const auto& offsets = net_->head_offsets();
    const int n = static_cast<int>(offsets.size());
    if (head_offsets_dev_ && head_offsets_dev_n_ != n) {
        cudaFree(head_offsets_dev_);
        head_offsets_dev_   = nullptr;
        head_offsets_dev_n_ = 0;
    }
    if (!head_offsets_dev_) {
        BGA_CUDA_CHECK(cudaMalloc(&head_offsets_dev_, sizeof(int) * n));
        head_offsets_dev_n_ = n;
    }
    BGA_CUDA_CHECK(cudaMemcpy(head_offsets_dev_, offsets.data(),
                              sizeof(int) * n, cudaMemcpyHostToDevice));

    gpu_ready_ = true;
}

GenericTrainStep GenericExItTrainer::step_gpu_() {
    GenericTrainStep s;
    auto batch = buf_->sample(cfg_.batch, rng_);
    if (batch.empty()) return s;

    const int B      = cfg_.batch;
    const int in_dim = net_->in_dim();
    const int n_act  = net_->num_actions();
    const int n_heads = net_->num_heads();
    ensure_gpu_staging_();

    // Stage minibatch on host into contiguous (B, *) buffers. Defensive: any
    // shape-mismatched situation contributes zeros (target = uniform, mask
    // = 1s, value = 0) so we always have exactly B rows. CPU path skips
    // such samples; we accept a small parity deviation in that edge case.
    nn::Tensor X_h = nn::Tensor::mat(B, in_dim);
    nn::Tensor T_h = nn::Tensor::mat(B, n_act);
    nn::Tensor M_h = nn::Tensor::mat(B, n_act);
    nn::Tensor V_h = nn::Tensor::mat(B, 1);

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

    nn::gpu::upload(X_h, X_BD_g_);
    nn::gpu::upload(T_h, T_BL_g_);
    nn::gpu::upload(V_h, V_B1_g_);
    const float* mask_dev = nullptr;
    if (any_mask) {
        nn::gpu::upload(M_h, M_BL_g_);
        mask_dev = M_BL_g_.data;
    }

    net_->zero_grad();

    // Forward (training) — caches every layer's batched activation.
    net_->forward_batched_train(X_BD_g_, logits_BL_g_, values_B1_g_);

    // Per-sample value MSE: writes dV_B1_g_ = (pred - target),
    // lv_per_sample_g_ = 0.5 * d^2.
    nn::gpu::mse_vec_per_sample_gpu(values_B1_g_, V_B1_g_,
                                    dV_B1_g_, lv_per_sample_g_);

    // Per-(sample, head) softmax-xent: writes probs, dLog = (p - t) on valid,
    // lp_per_sample = sum-over-heads of head xent.
    nn::gpu::softmax_xent_fused_batched_gpu(
        logits_BL_g_, T_BL_g_, mask_dev,
        head_offsets_dev_, n_heads,
        probs_BL_g_, dLog_BL_g_, lp_per_sample_g_);

    // Scale gradients on device. Match CPU exactly:
    //   dV  *= value_weight / B
    //   dLog *= policy_weight / B / n_heads
    const float B_f = static_cast<float>(B);
    nn::gpu::scale_inplace_gpu(dV_B1_g_, cfg_.value_weight / B_f);
    nn::gpu::scale_inplace_gpu(dLog_BL_g_,
                               cfg_.policy_weight / B_f / static_cast<float>(n_heads));

    // Backward + optimizer (all on device).
    net_->backward_batched(dLog_BL_g_, dV_B1_g_);
    net_->sgd_step(cfg_.lr, cfg_.momentum);

    // Single sync + downloads of the two per-sample loss vectors.
    nn::Tensor lv_h, lp_h;
    nn::gpu::download(lv_per_sample_g_, lv_h);
    nn::gpu::download(lp_per_sample_g_, lp_h);
    nn::gpu::cuda_sync();

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
#endif

} // namespace brogameagent::learn
