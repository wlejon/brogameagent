#include "brogameagent/nn/net_tx.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/runtime.h"
#endif

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

namespace {

inline void copy_slice(const Tensor& src, int off, int n, Tensor& dst) {
    std::memcpy(dst.ptr(), src.ptr() + off, n * sizeof(float));
}
inline void accum_slice(Tensor& dst_full, int off, const Tensor& slot_grad, int n) {
    float* d = dst_full.ptr() + off;
    const float* s = slot_grad.ptr();
    for (int i = 0; i < n; ++i) d[i] += s[i];
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
    enemy_enc_.to(d);
    ally_enc_.to(d);
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

void SingleHeroNetTX::forward(const Tensor& x, float& value, Tensor& logits) {
    assert(x.size() == observation::TOTAL);
    x_cache_ = x;
    const int D = cfg_.d_model;

    // ── Self stream ──
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
    // Run encoder (CPU or GPU).
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::GpuTensor gIn, gOut, gMask;
        gpu::upload(enemy_in_, gIn);
        gOut.resize(observation::K_ENEMIES, D);
        // Upload mask via a Tensor wrapper (GpuTensor needs a Tensor source).
        Tensor mask_h = Tensor::vec(observation::K_ENEMIES);
        for (int k = 0; k < observation::K_ENEMIES; ++k) mask_h[k] = e_mask_[k];
        gpu::upload(mask_h, gMask);
        enemy_enc_.forward(gIn, gMask.data, gOut);
        gpu::download(gOut, enemy_out_);
        gpu::cuda_sync();
    } else
#endif
    {
        enemy_enc_.forward(enemy_in_, e_mask_.data(), enemy_out_);
    }
    // Masked mean-pool.
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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::GpuTensor gIn, gOut, gMask;
        gpu::upload(ally_in_, gIn);
        gOut.resize(observation::K_ALLIES, D);
        Tensor mask_h = Tensor::vec(observation::K_ALLIES);
        for (int k = 0; k < observation::K_ALLIES; ++k) mask_h[k] = a_mask_[k];
        gpu::upload(mask_h, gMask);
        ally_enc_.forward(gIn, gMask.data, gOut);
        gpu::download(gOut, ally_out_);
        gpu::cuda_sync();
    } else
#endif
    {
        ally_enc_.forward(ally_in_, a_mask_.data(), ally_out_);
    }
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
    // (We don't propagate to dX — net is the input boundary.)
    (void)dSelfIn;

    // ── Enemy backward ──
    // d(pooled_e) → d(enemy_out) per valid row.
    Tensor dEnemyOut = Tensor::mat(observation::K_ENEMIES, D);
    dEnemyOut.zero();
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dEnemyOut(k, j) = dConcat[1*D + j] * inv;
        }
    }
    // Through encoder.
    Tensor dEnemyIn(observation::K_ENEMIES, D);
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::GpuTensor gdY, gdX;
        gpu::upload(dEnemyOut, gdY);
        gdX.resize(observation::K_ENEMIES, D);
        enemy_enc_.backward(gdY, gdX);
        gpu::download(gdX, dEnemyIn);
        gpu::cuda_sync();
    } else
#endif
    {
        enemy_enc_.backward(dEnemyOut, dEnemyIn);
    }
    // Per-slot Linear backward (re-prime x_cache_ via fresh forward of slot_in).
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
            enemy_proj_.forward(slot_in, dummy);   // refresh x_cache_
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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::GpuTensor gdY, gdX;
        gpu::upload(dAllyOut, gdY);
        gdX.resize(observation::K_ALLIES, D);
        ally_enc_.backward(gdY, gdX);
        gpu::download(gdX, dAllyIn);
        gpu::cuda_sync();
    } else
#endif
    {
        ally_enc_.backward(dAllyOut, dAllyIn);
    }
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

} // namespace brogameagent::nn
