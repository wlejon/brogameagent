#include "brogameagent/nn/net_tx.h"

#ifdef BGA_HAS_GPU
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

namespace {

inline void copy_slice(const Tensor& src, int off, int n, Tensor& dst) {
    std::memcpy(dst.ptr(), src.ptr() + off, n * sizeof(float));
}

} // namespace

void SingleHeroNetTX::init(const Config& cfg) {
    cfg_ = cfg;
    uint64_t rng = cfg.seed;

    // Self stream.
    self_fc1_.init(observation::SELF_FEATURES, cfg.self_hidden, rng);
    self_fc2_.init(cfg.self_hidden, cfg.d_model, rng);
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
    enemy_in_.resize(observation::K_ENEMIES, cfg.d_model);
    enemy_out_.resize(observation::K_ENEMIES, cfg.d_model);
    e_mask_.assign(observation::K_ENEMIES, 0.0f);
    e_valid_.assign(observation::K_ENEMIES, 0);
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
    ally_in_.resize(observation::K_ALLIES, cfg.d_model);
    ally_out_.resize(observation::K_ALLIES, cfg.d_model);
    a_mask_.assign(observation::K_ALLIES, 0.0f);
    a_valid_.assign(observation::K_ALLIES, 0);
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

void SingleHeroNetTX::to(Device d) {
    if (d == device_) return;
    device_require_cuda("SingleHeroNetTX");
#ifdef BGA_HAS_GPU
    self_fc1_.to(d);
    self_fc2_.to(d);
    enemy_proj_.to(d);
    enemy_enc_.to(d);
    ally_proj_.to(d);
    ally_enc_.to(d);
    trunk_.to(d);
    value_head_.to(d);
    head_.to(d);
    if (d == Device::GPU) {
        const int D  = cfg_.d_model;
        const int TH = cfg_.trunk_hidden;
        x_g_.resize(observation::TOTAL, 1);
        self_in_g_.resize(observation::SELF_FEATURES, 1);
        self_h_raw_g_.resize(cfg_.self_hidden, 1);
        self_h_act_g_.resize(cfg_.self_hidden, 1);
        self_z_g_.resize(D, 1);
        enemy_in_g_.resize(observation::K_ENEMIES, D);
        enemy_out_g_.resize(observation::K_ENEMIES, D);
        ally_in_g_.resize(observation::K_ALLIES, D);
        ally_out_g_.resize(observation::K_ALLIES, D);
        e_mask_g_.resize(observation::K_ENEMIES, 1);
        a_mask_g_.resize(observation::K_ALLIES, 1);
        enemy_pooled_g_.resize(D, 1);
        ally_pooled_g_.resize(D, 1);
        concat_g_.resize(3 * D, 1);
        trunk_raw_g_.resize(TH, 1);
        trunk_act_g_.resize(TH, 1);
        slot_in_e_g_.resize(observation::ENEMY_FEATURES, 1);
        slot_in_a_g_.resize(observation::ALLY_FEATURES, 1);
        slot_proj_g_.resize(D, 1);
        dTrunkAct_g_.resize(TH, 1);
        dTrunkRaw_g_.resize(TH, 1);
        dTrunkFromV_g_.resize(TH, 1);
        dTrunkFromP_g_.resize(TH, 1);
        dConcat_g_.resize(3 * D, 1);
        dSelfZ_g_.resize(D, 1);
        dSelfHact_g_.resize(cfg_.self_hidden, 1);
        dSelfHraw_g_.resize(cfg_.self_hidden, 1);
        dSelfIn_g_.resize(observation::SELF_FEATURES, 1);
        dEnemyOut_g_.resize(observation::K_ENEMIES, D);
        dEnemyIn_g_.resize(observation::K_ENEMIES, D);
        dAllyOut_g_.resize(observation::K_ALLIES, D);
        dAllyIn_g_.resize(observation::K_ALLIES, D);
        dSlotProj_g_.resize(D, 1);
        dSlotInE_g_.resize(observation::ENEMY_FEATURES, 1);
        dSlotInA_g_.resize(observation::ALLY_FEATURES, 1);
    }
#endif
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

// ─── CPU forward / backward ───────────────────────────────────────────────

void SingleHeroNetTX::forward(const Tensor& x, float& value, Tensor& logits) {
    assert(x.size() == observation::TOTAL);
    x_cache_ = x;
    const int D = cfg_.d_model;

#ifdef BGA_HAS_GPU
    if (device_ == Device::GPU) {
        // Upload x once, route through GPU-native forward, download value+logits.
        gpu::upload(x_cache_, x_g_);
        gpu::GpuTensor logits_g;
        logits_g.resize(logits.size(), 1);
        forward(x_g_, logits_g);
        // Download logits + scalar value.
        gpu::download(logits_g, logits);
        Tensor v_h(1, 1);
        gpu::download(value_head_.value_gpu(), v_h);
        gpu::cuda_sync();
        value = v_h[0];
        return;
    }
#endif

    // ── Self stream (CPU) ──
    Tensor self_in = Tensor::vec(observation::SELF_FEATURES);
    copy_slice(x, 0, observation::SELF_FEATURES, self_in);
    self_fc1_.forward(self_in, self_h_raw_);
    relu_forward(self_h_raw_, self_h_act_);
    self_fc2_.forward(self_h_act_, self_z_);

    // ── Enemy stream: per-slot proj into (K, D) ──
    const int off_e = observation::SELF_FEATURES;
    e_n_valid_ = 0;
    enemy_in_.zero();
    Tensor slot_in = Tensor::vec(observation::ENEMY_FEATURES);
    Tensor proj_out = Tensor::vec(D);
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        const bool valid = x[base] > 0.5f;
        e_valid_[k] = valid ? 1 : 0;
        e_mask_[k]  = valid ? 1.0f : 0.0f;
        if (!valid) continue;
        copy_slice(x, base, observation::ENEMY_FEATURES, slot_in);
        enemy_proj_.forward(slot_in, proj_out);
        for (int j = 0; j < D; ++j) enemy_in_(k, j) = proj_out[j];
        ++e_n_valid_;
    }
    enemy_enc_.forward(enemy_in_, e_mask_.data(), enemy_out_);
    enemy_pooled_.zero();
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) enemy_pooled_[j] += enemy_out_(k, j);
        }
        for (int j = 0; j < D; ++j) enemy_pooled_[j] *= inv;
    }

    // ── Ally stream ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    a_n_valid_ = 0;
    ally_in_.zero();
    Tensor slot_in_a = Tensor::vec(observation::ALLY_FEATURES);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        const bool valid = x[base] > 0.5f;
        a_valid_[k] = valid ? 1 : 0;
        a_mask_[k]  = valid ? 1.0f : 0.0f;
        if (!valid) continue;
        copy_slice(x, base, observation::ALLY_FEATURES, slot_in_a);
        ally_proj_.forward(slot_in_a, proj_out);
        for (int j = 0; j < D; ++j) ally_in_(k, j) = proj_out[j];
        ++a_n_valid_;
    }
    ally_enc_.forward(ally_in_, a_mask_.data(), ally_out_);
    ally_pooled_.zero();
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) ally_pooled_[j] += ally_out_(k, j);
        }
        for (int j = 0; j < D; ++j) ally_pooled_[j] *= inv;
    }

    // ── Concat → trunk → heads ──
    for (int j = 0; j < D; ++j) concat_[0*D + j] = self_z_[j];
    for (int j = 0; j < D; ++j) concat_[1*D + j] = enemy_pooled_[j];
    for (int j = 0; j < D; ++j) concat_[2*D + j] = ally_pooled_[j];

    trunk_.forward(concat_, trunk_raw_);
    trunk_act_.forward(trunk_raw_, trunk_act_out_);
    value_head_.forward(trunk_act_out_, value);
    head_.forward(trunk_act_out_, logits);
}

void SingleHeroNetTX::backward(float dValue, const Tensor& dLogits) {
    const int D = cfg_.d_model;
    const int TH = cfg_.trunk_hidden;

#ifdef BGA_HAS_GPU
    if (device_ == Device::GPU) {
        // Upload dLogits, write dValue scalar, run GPU backward.
        gpu::GpuTensor dLogits_g;
        gpu::upload(dLogits, dLogits_g);
        Tensor dv_h(1, 1); dv_h[0] = dValue;
        gpu::upload(dv_h, value_head_.dValue_gpu());
        backward(dLogits_g);
        gpu::cuda_sync();
        return;
    }
#endif

    // Heads → trunk_act_out_.
    Tensor dTrunkV = Tensor::vec(TH);
    value_head_.backward(dValue, dTrunkV);
    Tensor dTrunkP = Tensor::vec(TH);
    head_.backward(dLogits, dTrunkP);

    Tensor dTrunkAct = Tensor::vec(TH);
    for (int i = 0; i < TH; ++i) dTrunkAct[i] = dTrunkV[i] + dTrunkP[i];

    Tensor dTrunkRaw = Tensor::vec(TH);
    trunk_act_.backward(dTrunkAct, dTrunkRaw);

    // Trunk Linear.
    Tensor dConcat = Tensor::vec(3 * D);
    trunk_.backward(dTrunkRaw, dConcat);

    // ── Self backward ──
    Tensor dSelfZ = Tensor::vec(D);
    for (int j = 0; j < D; ++j) dSelfZ[j] = dConcat[0*D + j];
    Tensor dSelfHact = Tensor::vec(cfg_.self_hidden);
    self_fc2_.backward(dSelfZ, dSelfHact);
    Tensor dSelfHraw = Tensor::vec(cfg_.self_hidden);
    relu_backward(self_h_raw_, dSelfHact, dSelfHraw);
    Tensor dSelfIn = Tensor::vec(observation::SELF_FEATURES);
    self_fc1_.backward(dSelfHraw, dSelfIn);
    (void)dSelfIn;

    // ── Enemy backward ──
    Tensor dEnemyOut = Tensor::mat(observation::K_ENEMIES, D);
    dEnemyOut.zero();
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dEnemyOut(k, j) = dConcat[1*D + j] * inv;
        }
    }
    Tensor dEnemyIn(observation::K_ENEMIES, D);
    enemy_enc_.backward(dEnemyOut, dEnemyIn);
    {
        Tensor slot_in = Tensor::vec(observation::ENEMY_FEATURES);
        Tensor dRow = Tensor::vec(D);
        Tensor dSlot = Tensor::vec(observation::ENEMY_FEATURES);
        Tensor dummy = Tensor::vec(D);
        const int off_e = observation::SELF_FEATURES;
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dRow[j] = dEnemyIn(k, j);
            copy_slice(x_cache_, off_e + k * observation::ENEMY_FEATURES,
                       observation::ENEMY_FEATURES, slot_in);
            enemy_proj_.forward(slot_in, dummy);
            enemy_proj_.backward(dRow, dSlot);
            (void)dSlot;
        }
    }

    // ── Ally backward ──
    Tensor dAllyOut = Tensor::mat(observation::K_ALLIES, D);
    dAllyOut.zero();
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dAllyOut(k, j) = dConcat[2*D + j] * inv;
        }
    }
    Tensor dAllyIn(observation::K_ALLIES, D);
    ally_enc_.backward(dAllyOut, dAllyIn);
    {
        Tensor slot_in = Tensor::vec(observation::ALLY_FEATURES);
        Tensor dRow = Tensor::vec(D);
        Tensor dSlot = Tensor::vec(observation::ALLY_FEATURES);
        Tensor dummy = Tensor::vec(D);
        const int off_a = observation::SELF_FEATURES
                        + observation::K_ENEMIES * observation::ENEMY_FEATURES;
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dRow[j] = dAllyIn(k, j);
            copy_slice(x_cache_, off_a + k * observation::ALLY_FEATURES,
                       observation::ALLY_FEATURES, slot_in);
            ally_proj_.forward(slot_in, dummy);
            ally_proj_.backward(dRow, dSlot);
            (void)dSlot;
        }
    }
}

#ifdef BGA_HAS_GPU

// Helper: build per-slot mask and slot validity vectors from host x_cache_.
// Returns the count of valid slots; writes mask/valid arrays.
namespace {
int build_slot_mask(const Tensor& x, int off, int n_slots, int feat_per,
                    std::vector<float>& mask, std::vector<uint8_t>& valid) {
    int n_valid = 0;
    mask.assign(n_slots, 0.0f);
    valid.assign(n_slots, 0);
    for (int k = 0; k < n_slots; ++k) {
        const int base = off + k * feat_per;
        if (x[base] > 0.5f) { mask[k] = 1.0f; valid[k] = 1; ++n_valid; }
    }
    return n_valid;
}

// Slice a (K, D) device matrix row (of size D) into a flat (D, 1) device
// tensor. Implemented as a tiny cudaMemcpy. We avoid pulling cuda_runtime.h
// into this header by doing it via the GpuTensor copy primitive: clone is
// owning, so we use upload/download is heavy. Use a small helper from
// gpu/tensor.h indirectly? We don't have device-to-device row copy in the
// public ops API. Workaround: download the slot row to a host buffer and
// upload to slot scratch — that adds shuttles. Instead use a small batched
// approach: run per-slot Linear forward over the *whole* (K, FEAT) input
// using linear_forward_batched_gpu. That gives us (K, D) in one shot.
} // namespace

void SingleHeroNetTX::forward(const gpu::GpuTensor& x, gpu::GpuTensor& logits) {
    assert(device_ == Device::GPU);
    assert(x.size() == observation::TOTAL);
    x_external_ = &x;

    // We need x on host briefly to compute slot validity mask (a tiny scalar
    // check per slot). Download once into x_cache_.
    Tensor xh(observation::TOTAL, 1);
    gpu::download(x, xh);
    gpu::cuda_sync();
    x_cache_ = xh;

    // ── Self stream ──
    // Stage self segment into self_in_g_ via a host->device upload of just
    // SELF_FEATURES (single small upload — not a per-op shuttle).
    {
        Tensor sh = Tensor::vec(observation::SELF_FEATURES);
        copy_slice(x_cache_, 0, observation::SELF_FEATURES, sh);
        gpu::upload(sh, self_in_g_);
    }
    self_fc1_.forward(self_in_g_, self_h_raw_g_);
    gpu::relu_forward_gpu(self_h_raw_g_, self_h_act_g_);
    self_fc2_.forward(self_h_act_g_, self_z_g_);

    // ── Enemy stream ──
    const int off_e = observation::SELF_FEATURES;
    e_n_valid_ = build_slot_mask(x_cache_, off_e, observation::K_ENEMIES,
                                 observation::ENEMY_FEATURES, e_mask_, e_valid_);
    {
        // Build a (K_ENEMIES, ENEMY_FEATURES) host staging tensor for batched
        // input projection. Invalid slots are zeroed.
        Tensor staging(observation::K_ENEMIES, observation::ENEMY_FEATURES);
        staging.zero();
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            const int base = off_e + k * observation::ENEMY_FEATURES;
            for (int j = 0; j < observation::ENEMY_FEATURES; ++j)
                staging(k, j) = x_cache_[base + j];
        }
        gpu::GpuTensor staging_g;
        gpu::upload(staging, staging_g);
        gpu::linear_forward_batched_gpu(enemy_proj_.W_g(), enemy_proj_.b_g(),
                                        staging_g, enemy_in_g_);
        // Cache the staging input on the layer for backward (we need it for
        // per-slot Linear backward). Stash a copy on host; upload again as a
        // single-slot input during backward as needed.
        // Upload mask vector.
        Tensor mh = Tensor::vec(observation::K_ENEMIES);
        for (int k = 0; k < observation::K_ENEMIES; ++k) mh[k] = e_mask_[k];
        gpu::upload(mh, e_mask_g_);
    }
    // Encoder forward.
    enemy_enc_.forward(enemy_in_g_, e_mask_g_.data, enemy_out_g_);
    // Masked mean pool.
    gpu::masked_mean_pool_forward_gpu(enemy_out_g_, e_mask_g_.data, enemy_pooled_g_);

    // ── Ally stream ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    a_n_valid_ = build_slot_mask(x_cache_, off_a, observation::K_ALLIES,
                                 observation::ALLY_FEATURES, a_mask_, a_valid_);
    {
        Tensor staging(observation::K_ALLIES, observation::ALLY_FEATURES);
        staging.zero();
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            const int base = off_a + k * observation::ALLY_FEATURES;
            for (int j = 0; j < observation::ALLY_FEATURES; ++j)
                staging(k, j) = x_cache_[base + j];
        }
        gpu::GpuTensor staging_g;
        gpu::upload(staging, staging_g);
        gpu::linear_forward_batched_gpu(ally_proj_.W_g(), ally_proj_.b_g(),
                                        staging_g, ally_in_g_);
        Tensor mh = Tensor::vec(observation::K_ALLIES);
        for (int k = 0; k < observation::K_ALLIES; ++k) mh[k] = a_mask_[k];
        gpu::upload(mh, a_mask_g_);
    }
    ally_enc_.forward(ally_in_g_, a_mask_g_.data, ally_out_g_);
    gpu::masked_mean_pool_forward_gpu(ally_out_g_, a_mask_g_.data, ally_pooled_g_);

    // ── Concat → trunk → heads ──
    {
        std::vector<const gpu::GpuTensor*> parts{&self_z_g_, &enemy_pooled_g_, &ally_pooled_g_};
        gpu::concat_rows_gpu(parts, concat_g_);
    }
    trunk_.forward(concat_g_, trunk_raw_g_);
    gpu::relu_forward_gpu(trunk_raw_g_, trunk_act_g_);
    value_head_.forward(trunk_act_g_);
    head_.forward(trunk_act_g_, logits);
}

void SingleHeroNetTX::backward(const gpu::GpuTensor& dLogits) {
    assert(device_ == Device::GPU);
    const int D = cfg_.d_model;

    // ── Heads → trunk_act ──
    value_head_.backward(dTrunkFromV_g_);
    head_.backward(dLogits, dTrunkFromP_g_);

    // dTrunkAct = dTrunkFromV + dTrunkFromP. Ping the second into the first.
    // Use an explicit add: dTrunkAct = dTrunkFromV; dTrunkAct += dTrunkFromP.
    // We'll just write into dTrunkAct_g_ via a copy-then-add via concat trick.
    // Simpler: compute dTrunkAct = dTrunkFromV first, then add dTrunkFromP.
    // Use add_inplace: dTrunkFromV += dTrunkFromP; alias as dTrunkAct.
    gpu::add_inplace_gpu(dTrunkFromV_g_, dTrunkFromP_g_);
    // ReLU backward through trunk_act_ → trunk_raw_.
    gpu::relu_backward_gpu(trunk_raw_g_, dTrunkFromV_g_, dTrunkRaw_g_);
    trunk_.backward(dTrunkRaw_g_, dConcat_g_);

    // ── Split dConcat into [dSelfZ, dEnemyPooled (placeholder), dAllyPooled (placeholder)] ──
    // We can use split_rows with 3 parts of size D each.
    gpu::GpuTensor dSelfZ_local, dEnemyPooled_local, dAllyPooled_local;
    dSelfZ_local.resize(D, 1);
    dEnemyPooled_local.resize(D, 1);
    dAllyPooled_local.resize(D, 1);
    {
        std::vector<gpu::GpuTensor*> parts{&dSelfZ_local, &dEnemyPooled_local, &dAllyPooled_local};
        gpu::split_rows_gpu(dConcat_g_, parts);
    }

    // ── Self backward ──
    self_fc2_.backward(dSelfZ_local, dSelfHact_g_);
    gpu::relu_backward_gpu(self_h_raw_g_, dSelfHact_g_, dSelfHraw_g_);
    self_fc1_.backward(dSelfHraw_g_, dSelfIn_g_);

    // ── Enemy stream backward ──
    // d(enemy_pooled) → d(enemy_out): mean-pool backward broadcasts.
    gpu::masked_mean_pool_backward_gpu(dEnemyPooled_local, e_mask_g_.data,
                                       observation::K_ENEMIES, dEnemyOut_g_);
    enemy_enc_.backward(dEnemyOut_g_, dEnemyIn_g_);
    // Per-slot Linear backward. Like the CPU path, we re-prime each slot via
    // a fresh forward(slot_in) before backward(dRow, dSlot).
    {
        Tensor slot_in_h = Tensor::vec(observation::ENEMY_FEATURES);
        Tensor dRow_h    = Tensor::vec(D);
        const int off_e = observation::SELF_FEATURES;
        // Download dEnemyIn once to host so we can iterate per row cheaply.
        Tensor dEnemyIn_h(observation::K_ENEMIES, D);
        gpu::download(dEnemyIn_g_, dEnemyIn_h);
        gpu::cuda_sync();
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dRow_h[j] = dEnemyIn_h(k, j);
            const int base = off_e + k * observation::ENEMY_FEATURES;
            for (int j = 0; j < observation::ENEMY_FEATURES; ++j)
                slot_in_h[j] = x_cache_[base + j];
            gpu::upload(slot_in_h, slot_in_e_g_);
            gpu::upload(dRow_h, dSlotProj_g_);
            // Re-prime cache and accumulate grads.
            enemy_proj_.forward(slot_in_e_g_, slot_proj_g_);
            enemy_proj_.backward(dSlotProj_g_, dSlotInE_g_);
        }
    }

    // ── Ally stream backward ──
    gpu::masked_mean_pool_backward_gpu(dAllyPooled_local, a_mask_g_.data,
                                       observation::K_ALLIES, dAllyOut_g_);
    ally_enc_.backward(dAllyOut_g_, dAllyIn_g_);
    {
        Tensor slot_in_h = Tensor::vec(observation::ALLY_FEATURES);
        Tensor dRow_h    = Tensor::vec(D);
        const int off_a = observation::SELF_FEATURES
                        + observation::K_ENEMIES * observation::ENEMY_FEATURES;
        Tensor dAllyIn_h(observation::K_ALLIES, D);
        gpu::download(dAllyIn_g_, dAllyIn_h);
        gpu::cuda_sync();
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dRow_h[j] = dAllyIn_h(k, j);
            const int base = off_a + k * observation::ALLY_FEATURES;
            for (int j = 0; j < observation::ALLY_FEATURES; ++j)
                slot_in_h[j] = x_cache_[base + j];
            gpu::upload(slot_in_h, slot_in_a_g_);
            gpu::upload(dRow_h, dSlotProj_g_);
            ally_proj_.forward(slot_in_a_g_, slot_proj_g_);
            ally_proj_.backward(dSlotProj_g_, dSlotInA_g_);
        }
    }
}

// ─── Batched inference forward ────────────────────────────────────────────
//
// Stages every batch element's input on host once, builds slot-validity
// masks for all rows in a single sweep, then drives the network end-to-end
// on device:
//
//   • self stream  — three batched ops (Linear, ReLU, Linear) over (B, ·).
//   • per-slot enemy/ally projections — one batched Linear each producing
//     (B*K, D), eliminating the K_ENEMIES + K_ALLIES individual launches
//     the prior naive loop was making per batch row.
//   • encoder + masked mean-pool — looped per batch row but with no host
//     blocks; encoder forwards queue into the default stream and pooled
//     outputs land directly in (B, D) row-views via non-owning GpuTensor
//     views.
//   • concat / trunk / ReLU — one launch each (concat is cudaMemcpy2DAsync
//     per part).
//   • heads — value/policy heads aren't yet batched, so loop B times
//     queuing kernels into the stream; per-row outputs are gathered into
//     values_B1 / logits_BL via small device-to-device chunk copies. No
//     host syncs in the loop.
//
// The remaining serialization point is the encoder (B forwards back to
// back) and the heads loop — both are fixed by adding (B, K, D) batched
// kernels to MHA / LayerNorm / FF and a batched value+policy head, which
// is a separate, much larger change.
void SingleHeroNetTX::forward_batched(const gpu::GpuTensor& X_BD,
                                      gpu::GpuTensor& logits_BL,
                                      gpu::GpuTensor& values_B1) {
    assert(device_ == Device::GPU);
    const int B    = X_BD.rows;
    const int D_in = X_BD.cols;
    assert(D_in == observation::TOTAL);
    const int D    = cfg_.d_model;
    const int TH   = cfg_.trunk_hidden;
    const int L    = head_.total_logits();
    const int K_E  = observation::K_ENEMIES;
    const int K_A  = observation::K_ALLIES;
    const int F_E  = observation::ENEMY_FEATURES;
    const int F_A  = observation::ALLY_FEATURES;
    const int F_S  = observation::SELF_FEATURES;

    if (logits_BL.rows != B || logits_BL.cols != L) logits_BL.resize(B, L);
    if (values_B1.rows != B || values_B1.cols != 1) values_B1.resize(B, 1);
    if (B == 0) return;

    // ── 1. Single download of the whole input batch. ──
    Tensor X_h(B, D_in);
    gpu::download(X_BD, X_h);
    gpu::cuda_sync();

    // ── 2. Build all host staging buffers in one pass. ──
    Tensor self_in_h (B,           F_S);
    Tensor enemy_in_h(B * K_E,     F_E);
    Tensor ally_in_h (B * K_A,     F_A);
    Tensor e_masks_h (B * K_E,     1);
    Tensor a_masks_h (B * K_A,     1);
    enemy_in_h.zero();
    ally_in_h.zero();
    e_masks_h.zero();
    a_masks_h.zero();

    const int off_e = F_S;
    const int off_a = off_e + K_E * F_E;
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < F_S; ++j)
            self_in_h(b, j) = X_h(b, j);
        for (int k = 0; k < K_E; ++k) {
            const int base = off_e + k * F_E;
            const int row  = b * K_E + k;
            if (X_h(b, base) > 0.5f) {
                e_masks_h[row] = 1.0f;
                for (int j = 0; j < F_E; ++j)
                    enemy_in_h(row, j) = X_h(b, base + j);
            }
        }
        for (int k = 0; k < K_A; ++k) {
            const int base = off_a + k * F_A;
            const int row  = b * K_A + k;
            if (X_h(b, base) > 0.5f) {
                a_masks_h[row] = 1.0f;
                for (int j = 0; j < F_A; ++j)
                    ally_in_h(row, j) = X_h(b, base + j);
            }
        }
    }

    // ── 3. Single uploads of staging buffers. ──
    gpu::GpuTensor self_in_g, enemy_in_g, ally_in_g, e_masks_g, a_masks_g;
    gpu::upload(self_in_h,  self_in_g);
    gpu::upload(enemy_in_h, enemy_in_g);
    gpu::upload(ally_in_h,  ally_in_g);
    gpu::upload(e_masks_h,  e_masks_g);
    gpu::upload(a_masks_h,  a_masks_g);

    // ── 4. Batched self stream. ──
    gpu::GpuTensor self_h_raw(B, cfg_.self_hidden);
    gpu::GpuTensor self_h_act(B, cfg_.self_hidden);
    gpu::GpuTensor self_z    (B, D);
    gpu::linear_forward_batched_gpu(self_fc1_.W_g(), self_fc1_.b_g(),
                                    self_in_g, self_h_raw);
    gpu::relu_forward_batched_gpu(self_h_raw, self_h_act);
    gpu::linear_forward_batched_gpu(self_fc2_.W_g(), self_fc2_.b_g(),
                                    self_h_act, self_z);

    // ── 5. Batched per-slot projections — one launch per stream. ──
    gpu::GpuTensor enemy_proj(B * K_E, D);
    gpu::GpuTensor ally_proj (B * K_A, D);
    gpu::linear_forward_batched_gpu(enemy_proj_.W_g(), enemy_proj_.b_g(),
                                    enemy_in_g, enemy_proj);
    gpu::linear_forward_batched_gpu(ally_proj_.W_g(),  ally_proj_.b_g(),
                                    ally_in_g,  ally_proj);

    // ── 6. Batched encoder forwards (one call each). The encoders use
    //   forward_inference_batched, which dispatches the LayerNorm and
    //   FeedForward in single kernel launches over (B*K, D) and loops MHA
    //   per batch element — no host syncs. After the encoder, the masked
    //   mean-pool is still per-batch but every kernel queues async into
    //   the default stream.                                                ──
    gpu::GpuTensor enemy_enc_out(B * K_E, D);
    gpu::GpuTensor ally_enc_out (B * K_A, D);
    enemy_enc_.forward_inference_batched(enemy_proj, e_masks_g.data,
                                          enemy_enc_out, B, K_E);
    ally_enc_ .forward_inference_batched(ally_proj,  a_masks_g.data,
                                          ally_enc_out,  B, K_A);

    gpu::GpuTensor enemy_pooled_BD(B, D);
    gpu::GpuTensor ally_pooled_BD (B, D);
    for (int b = 0; b < B; ++b) {
        gpu::GpuTensor e_view = gpu::GpuTensor::view(
            enemy_enc_out.data + static_cast<size_t>(b) * K_E * D, K_E, D);
        const float* e_mask_b = e_masks_g.data + b * K_E;
        gpu::GpuTensor pool_view_e = gpu::GpuTensor::view(
            enemy_pooled_BD.data + static_cast<size_t>(b) * D, D, 1);
        gpu::masked_mean_pool_forward_gpu(e_view, e_mask_b, pool_view_e);

        gpu::GpuTensor a_view = gpu::GpuTensor::view(
            ally_enc_out.data + static_cast<size_t>(b) * K_A * D, K_A, D);
        const float* a_mask_b = a_masks_g.data + b * K_A;
        gpu::GpuTensor pool_view_a = gpu::GpuTensor::view(
            ally_pooled_BD.data + static_cast<size_t>(b) * D, D, 1);
        gpu::masked_mean_pool_forward_gpu(a_view, a_mask_b, pool_view_a);
    }

    // ── 7. Batched concat (B, 3D). ──
    gpu::GpuTensor concat_B3D;
    {
        std::vector<const gpu::GpuTensor*> parts{
            &self_z, &enemy_pooled_BD, &ally_pooled_BD};
        gpu::concat_batched_rows_gpu(parts, concat_B3D);
    }

    // ── 8. Batched trunk. ──
    gpu::GpuTensor trunk_raw(B, TH);
    gpu::GpuTensor trunk_act(B, TH);
    gpu::linear_forward_batched_gpu(trunk_.W_g(), trunk_.b_g(),
                                    concat_B3D, trunk_raw);
    gpu::relu_forward_batched_gpu(trunk_raw, trunk_act);

    // ── 9. Per-batch heads. ValueHead writes to its single (1,1) buffer;
    //   each iteration's value is captured into values_B1[b] via a
    //   stream-ordered D2D copy before the next iteration overwrites it.
    //   Policy logits go directly into a row-view of logits_BL.            ──
    for (int b = 0; b < B; ++b) {
        gpu::GpuTensor trunk_view = gpu::GpuTensor::view(
            trunk_act.data + static_cast<size_t>(b) * TH, TH, 1);
        gpu::GpuTensor logits_view = gpu::GpuTensor::view(
            logits_BL.data + static_cast<size_t>(b) * L, L, 1);
        value_head_.forward(trunk_view);
        gpu::copy_d2d_gpu(value_head_.value_gpu(), 0, values_B1, b, 1);
        head_.forward(trunk_view, logits_view);
    }

    gpu::cuda_sync();
}

// ─── Training-time batched API (per-element loop) ─────────────────────────
//
// Both calls walk B and reuse the existing single-sample forward/backward
// against per-row views of X_BD, logits_BL, dLogits_BL. forward_batched_train
// just stages outputs (no caches preserved across the B iterations);
// backward_batched re-runs forward(x_b) to rebuild caches before calling
// backward(dLogits_b). Doubles forward FLOPs per train step — acceptable in
// v2; the proper fix is per-sample state arrays or true (B, K, D) kernels.
void SingleHeroNetTX::forward_batched_train(const gpu::GpuTensor& X_BD,
                                            gpu::GpuTensor& logits_BL,
                                            gpu::GpuTensor& values_B1) {
    assert(device_ == Device::GPU);
    const int B = X_BD.rows;
    const int D_in = X_BD.cols;
    assert(D_in == observation::TOTAL);
    const int L = head_.total_logits();

    if (logits_BL.rows != B || logits_BL.cols != L) logits_BL.resize(B, L);
    if (values_B1.rows != B || values_B1.cols != 1) values_B1.resize(B, 1);

    last_train_X_BD_ = &X_BD;

    for (int b = 0; b < B; ++b) {
        gpu::GpuTensor x_view = gpu::GpuTensor::view(
            X_BD.data + static_cast<size_t>(b) * D_in,
            observation::TOTAL, 1);
        gpu::GpuTensor logits_view = gpu::GpuTensor::view(
            logits_BL.data + static_cast<size_t>(b) * L, L, 1);
        // Single-sample GPU forward — fills layer caches for this b only.
        forward(x_view, logits_view);
        // Gather scalar value into values_B1[b].
        gpu::copy_d2d_gpu(value_head_.value_gpu(), 0, values_B1, b, 1);
    }
}

void SingleHeroNetTX::backward_batched(const gpu::GpuTensor& dLogits_BL,
                                       const gpu::GpuTensor& dValues_B1) {
    assert(device_ == Device::GPU);
    assert(last_train_X_BD_ != nullptr &&
           "forward_batched_train must be called before backward_batched");
    const gpu::GpuTensor& X_BD = *last_train_X_BD_;
    const int B = X_BD.rows;
    const int D_in = X_BD.cols;
    const int L = head_.total_logits();
    assert(dLogits_BL.rows == B && dLogits_BL.cols == L);
    assert(dValues_B1.rows == B && dValues_B1.cols == 1);

    // Scratch for per-element logits output of the re-forward (we don't read
    // the values, but forward() expects an output tensor).
    if (logits_row_g_.rows != L || logits_row_g_.cols != 1)
        logits_row_g_.resize(L, 1);

    for (int b = 0; b < B; ++b) {
        gpu::GpuTensor x_view = gpu::GpuTensor::view(
            X_BD.data + static_cast<size_t>(b) * D_in,
            observation::TOTAL, 1);
        // Re-prime per-element caches (slot masks, encoder activations, etc.)
        // by running forward again. Single-sample backward depends on these.
        forward(x_view, logits_row_g_);

        // Write dValue scalar into the value head's gradient slot.
        gpu::copy_d2d_gpu(dValues_B1, b, value_head_.dValue_gpu(), 0, 1);

        // dLogits view for this row.
        gpu::GpuTensor dLogits_view = gpu::GpuTensor::view(
            dLogits_BL.data + static_cast<size_t>(b) * L, L, 1);
        backward(dLogits_view);
    }
}

#endif // BGA_HAS_GPU

} // namespace brogameagent::nn
