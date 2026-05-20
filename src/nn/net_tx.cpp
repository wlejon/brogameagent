#include "brogameagent/nn/net_tx.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void SingleHeroNetTX::init(const Config& cfg) {
    cfg_ = cfg;
    uint64_t rng = cfg.seed;

    // Self stream.
    self_fc1_.init(observation::SELF_FEATURES, cfg.self_hidden, rng);
    self_fc2_.init(cfg.self_hidden, cfg.d_model, rng);
    self_in_.resize(observation::SELF_FEATURES, 1);
    self_h_raw_.resize(cfg.self_hidden, 1);
    self_h_act_.resize(cfg.self_hidden, 1);
    self_z_.resize(cfg.d_model, 1);

    // Enemy stream.
    enemy_proj_.init(observation::ENEMY_FEATURES, cfg.d_model, rng);
    {
        TransformerEncoder::Config ec{};
        ec.n_layers  = cfg.num_blocks;
        ec.dim       = cfg.d_model;
        ec.num_heads = cfg.num_heads;
        ec.d_ff      = cfg.d_ff;
        ec.n_slots   = observation::K_ENEMIES;
        ec.ln_eps    = cfg.ln_eps;
        ec.norm      = cfg.norm;
        enemy_enc_.init(ec, rng);
    }
    enemy_slotin_.resize(observation::K_ENEMIES, observation::ENEMY_FEATURES);
    enemy_in_.resize(observation::K_ENEMIES, cfg.d_model);
    enemy_out_.resize(observation::K_ENEMIES, cfg.d_model);
    e_mask_.resize(observation::K_ENEMIES, 1);
    enemy_pooled_.resize(cfg.d_model, 1);

    // Ally stream.
    ally_proj_.init(observation::ALLY_FEATURES, cfg.d_model, rng);
    {
        TransformerEncoder::Config ec{};
        ec.n_layers  = cfg.num_blocks;
        ec.dim       = cfg.d_model;
        ec.num_heads = cfg.num_heads;
        ec.d_ff      = cfg.d_ff;
        ec.n_slots   = observation::K_ALLIES;
        ec.ln_eps    = cfg.ln_eps;
        ec.norm      = cfg.norm;
        ally_enc_.init(ec, rng);
    }
    ally_slotin_.resize(observation::K_ALLIES, observation::ALLY_FEATURES);
    ally_in_.resize(observation::K_ALLIES, cfg.d_model);
    ally_out_.resize(observation::K_ALLIES, cfg.d_model);
    a_mask_.resize(observation::K_ALLIES, 1);
    ally_pooled_.resize(cfg.d_model, 1);

    // Trunk + heads.
    trunk_.init(3 * cfg.d_model, cfg.trunk_hidden, rng);
    concat_.resize(3 * cfg.d_model, 1);
    trunk_raw_.resize(cfg.trunk_hidden, 1);
    trunk_act_out_.resize(cfg.trunk_hidden, 1);
    value_head_.init(cfg.trunk_hidden, cfg.value_hidden, rng);
    head_.init(cfg.trunk_hidden, rng);

    x_cache_.resize(observation::TOTAL, 1);

    // Resolve head_sizes_ / head_offsets_ to mirror PolicyValueNet's contract.
    // FactoredPolicyHead is fixed at three segments: [N_MOVE, N_ATTACK, N_ABILITY].
    head_sizes_  = { FactoredPolicyHead::N_MOVE,
                     FactoredPolicyHead::N_ATTACK,
                     FactoredPolicyHead::N_ABILITY };
    head_offsets_.clear();
    head_offsets_.reserve(head_sizes_.size() + 1);
    int off = 0;
    for (int h : head_sizes_) { head_offsets_.push_back(off); off += h; }
    head_offsets_.push_back(off);     // trailing total == total_logits()
}

int SingleHeroNetTX::num_params() const {
    int n = 0;
    n += self_fc1_.num_params() + self_fc2_.num_params();
    n += enemy_proj_.num_params() + enemy_enc_.num_params();
    n += ally_proj_.num_params()  + ally_enc_.num_params();
    n += trunk_.num_params();
    n += value_head_.num_params() + head_.num_params();
    return n;
}

void SingleHeroNetTX::zero_grad() {
    self_fc1_.zero_grad(); self_fc2_.zero_grad();
    enemy_proj_.zero_grad(); enemy_enc_.zero_grad();
    ally_proj_.zero_grad();  ally_enc_.zero_grad();
    trunk_.zero_grad();
    value_head_.zero_grad();
    head_.zero_grad();
}

void SingleHeroNetTX::sgd_step(float lr, float m) {
    self_fc1_.sgd_step(lr, m); self_fc2_.sgd_step(lr, m);
    enemy_proj_.sgd_step(lr, m); enemy_enc_.sgd_step(lr, m);
    ally_proj_.sgd_step(lr, m);  ally_enc_.sgd_step(lr, m);
    trunk_.sgd_step(lr, m);
    value_head_.sgd_step(lr, m);
    head_.sgd_step(lr, m);
}

void SingleHeroNetTX::adam_step(float lr, float b1, float b2, float eps, int step) {
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_proj_.adam_step(lr, b1, b2, eps, step); enemy_enc_.adam_step(lr, b1, b2, eps, step);
    ally_proj_.adam_step(lr, b1, b2, eps, step);  ally_enc_.adam_step(lr, b1, b2, eps, step);
    trunk_.adam_step(lr, b1, b2, eps, step);
    value_head_.adam_step(lr, b1, b2, eps, step);
    head_.adam_step(lr, b1, b2, eps, step);
}

void SingleHeroNetTX::to(brotensor::Device d) {
    if (d == device_) return;
    self_fc1_.to(d);
    self_fc2_.to(d);
    enemy_proj_.to(d);
    enemy_enc_.to(d);
    ally_proj_.to(d);
    ally_enc_.to(d);
    trunk_.to(d);
    value_head_.to(d);
    head_.to(d);
    // Directly-owned activation caches.
    self_in_      = self_in_.to(d);
    self_h_raw_   = self_h_raw_.to(d);
    self_h_act_   = self_h_act_.to(d);
    self_z_       = self_z_.to(d);
    enemy_slotin_ = enemy_slotin_.to(d);
    enemy_in_     = enemy_in_.to(d);
    enemy_out_    = enemy_out_.to(d);
    e_mask_       = e_mask_.to(d);
    enemy_pooled_ = enemy_pooled_.to(d);
    ally_slotin_  = ally_slotin_.to(d);
    ally_in_      = ally_in_.to(d);
    ally_out_     = ally_out_.to(d);
    a_mask_       = a_mask_.to(d);
    ally_pooled_  = ally_pooled_.to(d);
    concat_       = concat_.to(d);
    trunk_raw_    = trunk_raw_.to(d);
    trunk_act_out_ = trunk_act_out_.to(d);
    x_cache_      = x_cache_.to(d);
    device_ = d;
}

void SingleHeroNetTX::save_to(std::vector<uint8_t>& out) const {
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_proj_.save_to(out); enemy_enc_.save_to(out);
    ally_proj_.save_to(out);  ally_enc_.save_to(out);
    trunk_.save_to(out);
    value_head_.save_to(out);
    head_.save_to(out);
}

void SingleHeroNetTX::load_from(const uint8_t* d, size_t& o, size_t s) {
    self_fc1_.load_from(d, o, s); self_fc2_.load_from(d, o, s);
    enemy_proj_.load_from(d, o, s); enemy_enc_.load_from(d, o, s);
    ally_proj_.load_from(d, o, s);  ally_enc_.load_from(d, o, s);
    trunk_.load_from(d, o, s);
    value_head_.load_from(d, o, s);
    head_.load_from(d, o, s);
}

static constexpr uint32_t kMagicTX   = 0x58544742; // "BGTX" LE
static constexpr uint32_t kVersionTX = 1;

std::vector<uint8_t> SingleHeroNetTX::save() const {
    std::vector<uint8_t> out;
    out.resize(2 * sizeof(uint32_t));
    std::memcpy(out.data(),                     &kMagicTX,   sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t),  &kVersionTX, sizeof(uint32_t));
    save_to(out);
    return out;
}

void SingleHeroNetTX::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= 2 * sizeof(uint32_t));
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic,   blob.data(),                    sizeof(uint32_t));
    std::memcpy(&version, blob.data() + sizeof(uint32_t), sizeof(uint32_t));
    assert(magic == kMagicTX);
    assert(version == kVersionTX);
    size_t off = 2 * sizeof(uint32_t);
    load_from(blob.data(), off, blob.size());
}

// ─── Forward / backward ───────────────────────────────────────────────────

void SingleHeroNetTX::forward(const brotensor::Tensor& x_in, float& value, brotensor::Tensor& logits) {
    assert(x_in.size() == observation::TOTAL);
    const int D = cfg_.d_model;

    // Stage the input to the parameter device so every op below dispatches
    // consistently — callers may hand us a host observation even when the net
    // has been migrated to a GPU backend.
    const brotensor::Device want = logits.device;  // caller's output device
    const brotensor::Tensor x = (x_in.device == device_) ? x_in : x_in.to(device_);
    x_cache_ = x;

    // ── Self stream ──
    brotensor::copy_d2d(x, 0, self_in_, 0, observation::SELF_FEATURES);
    self_fc1_.forward(self_in_, self_h_raw_);
    brotensor::relu_forward(self_h_raw_, self_h_act_);
    self_fc2_.forward(self_h_act_, self_z_);

    // ── Enemy stream: per-slot proj into (K, D), transformer, masked pool ──
    // Device-neutral: slot slicing via copy_d2d, validity mask via
    // build_slot_mask (device-resident), shared-weight per-slot proj via the
    // batched Linear, masked mean-pool via masked_mean_pool_forward.
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        brotensor::copy_d2d(x, off_e + k * observation::ENEMY_FEATURES,
                            enemy_slotin_, k * observation::ENEMY_FEATURES,
                            observation::ENEMY_FEATURES);
    }
    brotensor::build_slot_mask(x, off_e, observation::K_ENEMIES,
                               observation::ENEMY_FEATURES, e_mask_);
    enemy_proj_.forward_batched_train(enemy_slotin_, enemy_in_);
    enemy_enc_.forward(enemy_in_, static_cast<const float*>(e_mask_.data), enemy_out_);
    brotensor::masked_mean_pool_forward(
        enemy_out_, static_cast<const float*>(e_mask_.data), enemy_pooled_);

    // ── Ally stream ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        brotensor::copy_d2d(x, off_a + k * observation::ALLY_FEATURES,
                            ally_slotin_, k * observation::ALLY_FEATURES,
                            observation::ALLY_FEATURES);
    }
    brotensor::build_slot_mask(x, off_a, observation::K_ALLIES,
                               observation::ALLY_FEATURES, a_mask_);
    ally_proj_.forward_batched_train(ally_slotin_, ally_in_);
    ally_enc_.forward(ally_in_, static_cast<const float*>(a_mask_.data), ally_out_);
    brotensor::masked_mean_pool_forward(
        ally_out_, static_cast<const float*>(a_mask_.data), ally_pooled_);

    // ── Concat → trunk → heads ──
    brotensor::copy_d2d(self_z_,       0, concat_, 0 * D, D);
    brotensor::copy_d2d(enemy_pooled_, 0, concat_, 1 * D, D);
    brotensor::copy_d2d(ally_pooled_,  0, concat_, 2 * D, D);

    trunk_.forward(concat_, trunk_raw_);
    trunk_act_.forward(trunk_raw_, trunk_act_out_);
    value_head_.forward(trunk_act_out_, value);

    // Policy logits. When the caller's `logits` already lives on the parameter
    // device we write into it in place (preserving any non-owning view into a
    // larger batched buffer). Otherwise we compute on-device and copy back —
    // logits is a fresh standalone tensor in that case, so reassigning is safe.
    if (want == device_) {
        head_.forward(trunk_act_out_, logits);
    } else {
        brotensor::Tensor logits_dev =
            brotensor::Tensor::zeros_on(device_, head_.total_logits(), 1);
        head_.forward(trunk_act_out_, logits_dev);
        logits = logits_dev.to(want);
    }
}

void SingleHeroNetTX::backward(float dValue, const brotensor::Tensor& dLogits_in) {
    const int D = cfg_.d_model;
    const int TH = cfg_.trunk_hidden;
    const brotensor::Device dev = device_;
    // Stage the upstream gradient to the parameter device — callers may pass a
    // host dLogits even when the net lives on a GPU backend.
    const brotensor::Tensor dLogits =
        (dLogits_in.device == dev) ? dLogits_in : dLogits_in.to(dev);

    // Heads → trunk_act_out_.
    brotensor::Tensor dTrunkV = brotensor::Tensor::zeros_on(dev, TH, 1);
    value_head_.backward(dValue, dTrunkV);
    brotensor::Tensor dTrunkP = brotensor::Tensor::zeros_on(dev, TH, 1);
    head_.backward(dLogits, dTrunkP);

    brotensor::Tensor dTrunkAct = dTrunkV;
    brotensor::add_inplace(dTrunkAct, dTrunkP);

    brotensor::Tensor dTrunkRaw = brotensor::Tensor::zeros_on(dev, TH, 1);
    trunk_act_.backward(dTrunkAct, dTrunkRaw);

    // Trunk Linear.
    brotensor::Tensor dConcat = brotensor::Tensor::zeros_on(dev, 3 * D, 1);
    trunk_.backward(dTrunkRaw, dConcat);

    // ── Self backward ──
    brotensor::Tensor dSelfZ = brotensor::Tensor::zeros_on(dev, D, 1);
    brotensor::copy_d2d(dConcat, 0 * D, dSelfZ, 0, D);
    brotensor::Tensor dSelfHact = brotensor::Tensor::zeros_on(dev, cfg_.self_hidden, 1);
    self_fc2_.backward(dSelfZ, dSelfHact);
    brotensor::Tensor dSelfHraw = brotensor::Tensor::zeros_on(dev, cfg_.self_hidden, 1);
    brotensor::relu_backward(self_h_raw_, dSelfHact, dSelfHraw);
    brotensor::Tensor dSelfIn = brotensor::Tensor::zeros_on(
        dev, observation::SELF_FEATURES, 1);
    self_fc1_.backward(dSelfHraw, dSelfIn);
    (void)dSelfIn;

    // ── Enemy backward ──
    // Pooled-embedding grad is broadcast across valid slot rows by
    // masked_mean_pool_backward (1/n_valid each, 0 on invalid).
    brotensor::Tensor dPoolE = brotensor::Tensor::zeros_on(dev, D, 1);
    brotensor::copy_d2d(dConcat, 1 * D, dPoolE, 0, D);
    brotensor::Tensor dEnemyOut = brotensor::Tensor::zeros_on(
        dev, observation::K_ENEMIES, D);
    brotensor::masked_mean_pool_backward(
        dPoolE, static_cast<const float*>(e_mask_.data),
        observation::K_ENEMIES, dEnemyOut);
    brotensor::Tensor dEnemyIn = brotensor::Tensor::zeros_on(
        dev, observation::K_ENEMIES, D);
    enemy_enc_.backward(dEnemyOut, dEnemyIn);
    brotensor::Tensor dEnemySlot = brotensor::Tensor::zeros_on(
        dev, observation::K_ENEMIES, observation::ENEMY_FEATURES);
    enemy_proj_.backward_batched(dEnemyIn, dEnemySlot);
    (void)dEnemySlot;

    // ── Ally backward ──
    brotensor::Tensor dPoolA = brotensor::Tensor::zeros_on(dev, D, 1);
    brotensor::copy_d2d(dConcat, 2 * D, dPoolA, 0, D);
    brotensor::Tensor dAllyOut = brotensor::Tensor::zeros_on(
        dev, observation::K_ALLIES, D);
    brotensor::masked_mean_pool_backward(
        dPoolA, static_cast<const float*>(a_mask_.data),
        observation::K_ALLIES, dAllyOut);
    brotensor::Tensor dAllyIn = brotensor::Tensor::zeros_on(
        dev, observation::K_ALLIES, D);
    ally_enc_.backward(dAllyOut, dAllyIn);
    brotensor::Tensor dAllySlot = brotensor::Tensor::zeros_on(
        dev, observation::K_ALLIES, observation::ALLY_FEATURES);
    ally_proj_.backward_batched(dAllyIn, dAllySlot);
    (void)dAllySlot;
}

// ─── BatchedNet: inference-only batched forward ────────────────────────────
//
// No true batched kernels — loop the single-sample forward over row views of
// the (B, *) staging tensors. forward() dispatches by device, so this is
// device-neutral. Outputs are (re)allocated on X_BD's device so the row views
// and the net's compute share a backend.
void SingleHeroNetTX::forward_batched(const brotensor::Tensor& X_BD,
                                      brotensor::Tensor& logits_BL,
                                      brotensor::Tensor& values_B1) {
    const int B    = X_BD.rows;
    const int din  = X_BD.cols;
    const int L    = logits_dim();
    if (logits_BL.rows != B || logits_BL.cols != L ||
        logits_BL.device != X_BD.device) {
        logits_BL = brotensor::Tensor::zeros_on(X_BD.device, B, L);
    }
    brotensor::Tensor Vh = brotensor::Tensor::mat(B, 1);
    for (int b = 0; b < B; ++b) {
        brotensor::Tensor xv = brotensor::Tensor::view(
            X_BD.device,
            static_cast<char*>(X_BD.data) + static_cast<size_t>(b) * din * sizeof(float),
            din, 1);
        brotensor::Tensor lv = brotensor::Tensor::view(
            logits_BL.device,
            static_cast<char*>(logits_BL.data) + static_cast<size_t>(b) * L * sizeof(float),
            L, 1);
        float v = 0.0f;
        forward(xv, v, lv);
        Vh.ptr()[b] = v;
    }
    values_B1 = Vh.to(X_BD.device);
}


} // namespace brogameagent::nn
