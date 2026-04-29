#include "brogameagent/nn/set_transformer.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

static inline void copy_slice(const Tensor& src, int off, int n, Tensor& dst) {
    std::memcpy(dst.ptr(), src.ptr() + off, n * sizeof(float));
}
static inline void accum_slice(Tensor& dst_full, int off, const Tensor& slot_grad, int n) {
    float* d = dst_full.ptr() + off;
    const float* s = slot_grad.ptr();
    for (int i = 0; i < n; ++i) d[i] += s[i];
}

void SetTransformerEncoder::init(const Config& cfg, uint64_t& rng_state) {
    cfg_ = cfg;

    self_fc1_.init(observation::SELF_FEATURES, cfg.hidden,   rng_state);
    self_fc2_.init(cfg.hidden,                  cfg.embed_dim, rng_state);
    self_h_.resize(cfg.hidden, 1);
    self_z_.resize(cfg.embed_dim, 1);

    enemy_proj_.init(observation::ENEMY_FEATURES, cfg.embed_dim, rng_state);
    enemy_attn_.init(observation::K_ENEMIES, cfg.embed_dim, rng_state);
    enemy_ln_.assign(observation::K_ENEMIES, LayerNorm{});
    for (auto& ln : enemy_ln_) ln.init(cfg.embed_dim);
    enemy_proj_raw_.resize(observation::K_ENEMIES, cfg.embed_dim);
    enemy_proj_act_.resize(observation::K_ENEMIES, cfg.embed_dim);
    enemy_attn_out_.resize(observation::K_ENEMIES, cfg.embed_dim);
    enemy_ln_out_.resize(observation::K_ENEMIES, cfg.embed_dim);
    e_valid_.assign(observation::K_ENEMIES, 0);

    ally_proj_.init(observation::ALLY_FEATURES, cfg.embed_dim, rng_state);
    ally_attn_.init(observation::K_ALLIES, cfg.embed_dim, rng_state);
    ally_ln_.assign(observation::K_ALLIES, LayerNorm{});
    for (auto& ln : ally_ln_) ln.init(cfg.embed_dim);
    ally_proj_raw_.resize(observation::K_ALLIES, cfg.embed_dim);
    ally_proj_act_.resize(observation::K_ALLIES, cfg.embed_dim);
    ally_attn_out_.resize(observation::K_ALLIES, cfg.embed_dim);
    ally_ln_out_.resize(observation::K_ALLIES, cfg.embed_dim);
    a_valid_.assign(observation::K_ALLIES, 0);

    x_cache_.resize(observation::TOTAL, 1);
}

int SetTransformerEncoder::num_params() const {
    int n = self_fc1_.num_params() + self_fc2_.num_params();
    n += enemy_proj_.num_params() + enemy_attn_.num_params();
    for (const auto& ln : enemy_ln_) n += ln.num_params();
    n += ally_proj_.num_params() + ally_attn_.num_params();
    for (const auto& ln : ally_ln_) n += ln.num_params();
    return n;
}

void SetTransformerEncoder::zero_grad() {
    self_fc1_.zero_grad(); self_fc2_.zero_grad();
    enemy_proj_.zero_grad(); enemy_attn_.zero_grad();
    for (auto& ln : enemy_ln_) ln.zero_grad();
    ally_proj_.zero_grad(); ally_attn_.zero_grad();
    for (auto& ln : ally_ln_) ln.zero_grad();
}

void SetTransformerEncoder::sgd_step(float lr, float m) {
    self_fc1_.sgd_step(lr, m); self_fc2_.sgd_step(lr, m);
    enemy_proj_.sgd_step(lr, m); enemy_attn_.sgd_step(lr, m);
    for (auto& ln : enemy_ln_) ln.sgd_step(lr, m);
    ally_proj_.sgd_step(lr, m); ally_attn_.sgd_step(lr, m);
    for (auto& ln : ally_ln_) ln.sgd_step(lr, m);
}

void SetTransformerEncoder::adam_step(float lr, float b1, float b2, float eps, int step) {
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_proj_.adam_step(lr, b1, b2, eps, step); enemy_attn_.adam_step(lr, b1, b2, eps, step);
    for (auto& ln : enemy_ln_) ln.adam_step(lr, b1, b2, eps, step);
    ally_proj_.adam_step(lr, b1, b2, eps, step); ally_attn_.adam_step(lr, b1, b2, eps, step);
    for (auto& ln : ally_ln_) ln.adam_step(lr, b1, b2, eps, step);
}

void SetTransformerEncoder::save_to(std::vector<uint8_t>& out) const {
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_proj_.save_to(out); enemy_attn_.save_to(out);
    for (const auto& ln : enemy_ln_) ln.save_to(out);
    ally_proj_.save_to(out); ally_attn_.save_to(out);
    for (const auto& ln : ally_ln_) ln.save_to(out);
}

void SetTransformerEncoder::load_from(const uint8_t* d, size_t& o, size_t s) {
    self_fc1_.load_from(d, o, s); self_fc2_.load_from(d, o, s);
    enemy_proj_.load_from(d, o, s); enemy_attn_.load_from(d, o, s);
    for (auto& ln : enemy_ln_) ln.load_from(d, o, s);
    ally_proj_.load_from(d, o, s); ally_attn_.load_from(d, o, s);
    for (auto& ln : ally_ln_) ln.load_from(d, o, s);
}

void SetTransformerEncoder::forward(const Tensor& x, Tensor& y) {
    assert(x.size() == observation::TOTAL);
    assert(y.size() == out_dim());
    x_cache_ = x;
    const int D = cfg_.embed_dim;

    // Self stream.
    Tensor self_in = Tensor::vec(observation::SELF_FEATURES);
    copy_slice(x, 0, observation::SELF_FEATURES, self_in);
    Tensor tmp_h = Tensor::vec(cfg_.hidden);
    self_fc1_.forward(self_in, tmp_h);
    relu_forward(tmp_h, self_h_);
    self_fc2_.forward(self_h_, self_z_);

    // Enemy stream: per-slot Linear+ReLU into (K, D) matrix; attention; per-row LN; masked mean pool.
    const int off_e = observation::SELF_FEATURES;
    std::vector<float> e_mask(observation::K_ENEMIES, 0.0f);
    e_n_valid_ = 0;
    Tensor slot_in = Tensor::vec(observation::ENEMY_FEATURES);
    Tensor proj_out = Tensor::vec(D);
    enemy_proj_raw_.zero();
    enemy_proj_act_.zero();
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        const bool valid = x[base] > 0.5f;
        e_valid_[k] = valid ? 1 : 0;
        e_mask[k] = valid ? 1.0f : 0.0f;
        if (!valid) continue;
        copy_slice(x, base, observation::ENEMY_FEATURES, slot_in);
        enemy_proj_.forward(slot_in, proj_out);
        for (int j = 0; j < D; ++j) {
            enemy_proj_raw_(k, j) = proj_out[j];
            enemy_proj_act_(k, j) = proj_out[j] > 0.0f ? proj_out[j] : 0.0f;
        }
        ++e_n_valid_;
    }
    enemy_attn_.forward(enemy_proj_act_, e_mask.data(), enemy_attn_out_);
    // Per-row LayerNorm on valid rows.
    Tensor row_in = Tensor::vec(D), row_out = Tensor::vec(D);
    enemy_ln_out_.zero();
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        if (!e_valid_[k]) continue;
        for (int j = 0; j < D; ++j) row_in[j] = enemy_attn_out_(k, j);
        enemy_ln_[k].forward(row_in, row_out);
        for (int j = 0; j < D; ++j) enemy_ln_out_(k, j) = row_out[j];
    }
    // Masked mean pool.
    Tensor pooled_e = Tensor::vec(D); pooled_e.zero();
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) pooled_e[j] += enemy_ln_out_(k, j);
        }
        for (int j = 0; j < D; ++j) pooled_e[j] *= inv;
    }

    // Ally stream (same as enemy).
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    std::vector<float> a_mask(observation::K_ALLIES, 0.0f);
    a_n_valid_ = 0;
    Tensor slot_in_a = Tensor::vec(observation::ALLY_FEATURES);
    ally_proj_raw_.zero();
    ally_proj_act_.zero();
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        const bool valid = x[base] > 0.5f;
        a_valid_[k] = valid ? 1 : 0;
        a_mask[k] = valid ? 1.0f : 0.0f;
        if (!valid) continue;
        copy_slice(x, base, observation::ALLY_FEATURES, slot_in_a);
        ally_proj_.forward(slot_in_a, proj_out);
        for (int j = 0; j < D; ++j) {
            ally_proj_raw_(k, j) = proj_out[j];
            ally_proj_act_(k, j) = proj_out[j] > 0.0f ? proj_out[j] : 0.0f;
        }
        ++a_n_valid_;
    }
    ally_attn_.forward(ally_proj_act_, a_mask.data(), ally_attn_out_);
    ally_ln_out_.zero();
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        if (!a_valid_[k]) continue;
        for (int j = 0; j < D; ++j) row_in[j] = ally_attn_out_(k, j);
        ally_ln_[k].forward(row_in, row_out);
        for (int j = 0; j < D; ++j) ally_ln_out_(k, j) = row_out[j];
    }
    Tensor pooled_a = Tensor::vec(D); pooled_a.zero();
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) pooled_a[j] += ally_ln_out_(k, j);
        }
        for (int j = 0; j < D; ++j) pooled_a[j] *= inv;
    }

    // Concat.
    for (int j = 0; j < D; ++j) y[0*D + j] = self_z_[j];
    for (int j = 0; j < D; ++j) y[1*D + j] = pooled_e[j];
    for (int j = 0; j < D; ++j) y[2*D + j] = pooled_a[j];
}

void SetTransformerEncoder::backward(const Tensor& dY, Tensor& dX) {
    assert(dY.size() == out_dim());
    assert(dX.size() == observation::TOTAL);
    dX.zero();
    const int D = cfg_.embed_dim;

    // Self backward.
    Tensor dSelfZ = Tensor::vec(D);
    for (int j = 0; j < D; ++j) dSelfZ[j] = dY[0*D + j];
    Tensor dSelfH = Tensor::vec(cfg_.hidden);
    self_fc2_.backward(dSelfZ, dSelfH);
    for (int i = 0; i < cfg_.hidden; ++i) if (self_h_[i] <= 0.0f) dSelfH[i] = 0.0f;
    Tensor dSelfIn = Tensor::vec(observation::SELF_FEATURES);
    self_fc1_.backward(dSelfH, dSelfIn);
    accum_slice(dX, 0, dSelfIn, observation::SELF_FEATURES);

    // Enemy backward.
    const int off_e = observation::SELF_FEATURES;
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        // Grad distributed to each valid row's LN output.
        Tensor dLnOut = Tensor::mat(observation::K_ENEMIES, D); dLnOut.zero();
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dLnOut(k, j) = dY[1*D + j] * inv;
        }
        // Through per-row LN.
        Tensor dAttnOut = Tensor::mat(observation::K_ENEMIES, D); dAttnOut.zero();
        Tensor row_dy = Tensor::vec(D), row_dx = Tensor::vec(D);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) row_dy[j] = dLnOut(k, j);
            enemy_ln_[k].backward(row_dy, row_dx);
            for (int j = 0; j < D; ++j) dAttnOut(k, j) = row_dx[j];
        }
        // Through attention.
        Tensor dProjAct = Tensor::mat(observation::K_ENEMIES, D);
        enemy_attn_.backward(dAttnOut, dProjAct);
        // Through per-slot Linear+ReLU projection.
        Tensor dSlot = Tensor::vec(observation::ENEMY_FEATURES);
        Tensor dRow = Tensor::vec(D);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < D; ++j) {
                dRow[j] = enemy_proj_raw_(k, j) > 0.0f ? dProjAct(k, j) : 0.0f;
            }
            // Re-prime the Linear's x_cache_ for this slot's input.
            Tensor slot_in = Tensor::vec(observation::ENEMY_FEATURES);
            copy_slice(x_cache_, off_e + k * observation::ENEMY_FEATURES,
                       observation::ENEMY_FEATURES, slot_in);
            Tensor dummy = Tensor::vec(D);
            enemy_proj_.forward(slot_in, dummy);   // refresh x_cache_
            enemy_proj_.backward(dRow, dSlot);
            accum_slice(dX, off_e + k * observation::ENEMY_FEATURES, dSlot,
                        observation::ENEMY_FEATURES);
        }
    }

    // Ally backward.
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        Tensor dLnOut = Tensor::mat(observation::K_ALLIES, D); dLnOut.zero();
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) dLnOut(k, j) = dY[2*D + j] * inv;
        }
        Tensor dAttnOut = Tensor::mat(observation::K_ALLIES, D); dAttnOut.zero();
        Tensor row_dy = Tensor::vec(D), row_dx = Tensor::vec(D);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) row_dy[j] = dLnOut(k, j);
            ally_ln_[k].backward(row_dy, row_dx);
            for (int j = 0; j < D; ++j) dAttnOut(k, j) = row_dx[j];
        }
        Tensor dProjAct = Tensor::mat(observation::K_ALLIES, D);
        ally_attn_.backward(dAttnOut, dProjAct);
        Tensor dSlot = Tensor::vec(observation::ALLY_FEATURES);
        Tensor dRow = Tensor::vec(D);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < D; ++j) {
                dRow[j] = ally_proj_raw_(k, j) > 0.0f ? dProjAct(k, j) : 0.0f;
            }
            Tensor slot_in = Tensor::vec(observation::ALLY_FEATURES);
            copy_slice(x_cache_, off_a + k * observation::ALLY_FEATURES,
                       observation::ALLY_FEATURES, slot_in);
            Tensor dummy = Tensor::vec(D);
            ally_proj_.forward(slot_in, dummy);
            ally_proj_.backward(dRow, dSlot);
            accum_slice(dX, off_a + k * observation::ALLY_FEATURES, dSlot,
                        observation::ALLY_FEATURES);
        }
    }
}

} // namespace brogameagent::nn
