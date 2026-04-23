#include "brogameagent/nn/encoder.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void DeepSetsEncoder::init(const Config& cfg, uint64_t& rng_state) {
    cfg_ = cfg;

    self_fc1_.init(observation::SELF_FEATURES, cfg.hidden,   rng_state);
    self_fc2_.init(cfg.hidden,                  cfg.embed_dim, rng_state);
    enemy_fc1_.init(observation::ENEMY_FEATURES, cfg.hidden,   rng_state);
    enemy_fc2_.init(cfg.hidden,                  cfg.embed_dim, rng_state);
    ally_fc1_.init(observation::ALLY_FEATURES,  cfg.hidden,   rng_state);
    ally_fc2_.init(cfg.hidden,                  cfg.embed_dim, rng_state);

    self_h_.resize(cfg.hidden, 1);
    self_z_.resize(cfg.embed_dim, 1);

    e_h_.assign(observation::K_ENEMIES, Tensor::vec(cfg.hidden));
    e_z_.assign(observation::K_ENEMIES, Tensor::vec(cfg.embed_dim));
    a_h_.assign(observation::K_ALLIES,  Tensor::vec(cfg.hidden));
    a_z_.assign(observation::K_ALLIES,  Tensor::vec(cfg.embed_dim));
    e_valid_.assign(observation::K_ENEMIES, 0);
    a_valid_.assign(observation::K_ALLIES,  0);

    x_cache_.resize(observation::TOTAL, 1);
    slot_grad_in_.resize(std::max({observation::SELF_FEATURES,
                                   observation::ENEMY_FEATURES,
                                   observation::ALLY_FEATURES}), 1);
}

int DeepSetsEncoder::num_params() const {
    return self_fc1_.num_params() + self_fc2_.num_params()
         + enemy_fc1_.num_params() + enemy_fc2_.num_params()
         + ally_fc1_.num_params()  + ally_fc2_.num_params();
}

void DeepSetsEncoder::zero_grad() {
    self_fc1_.zero_grad(); self_fc2_.zero_grad();
    enemy_fc1_.zero_grad(); enemy_fc2_.zero_grad();
    ally_fc1_.zero_grad();  ally_fc2_.zero_grad();
}

void DeepSetsEncoder::sgd_step(float lr, float momentum) {
    self_fc1_.sgd_step(lr, momentum); self_fc2_.sgd_step(lr, momentum);
    enemy_fc1_.sgd_step(lr, momentum); enemy_fc2_.sgd_step(lr, momentum);
    ally_fc1_.sgd_step(lr, momentum);  ally_fc2_.sgd_step(lr, momentum);
}

void DeepSetsEncoder::save_to(std::vector<uint8_t>& out) const {
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_fc1_.save_to(out); enemy_fc2_.save_to(out);
    ally_fc1_.save_to(out);  ally_fc2_.save_to(out);
}

void DeepSetsEncoder::load_from(const uint8_t* data, size_t& offset, size_t size) {
    self_fc1_.load_from(data, offset, size);  self_fc2_.load_from(data, offset, size);
    enemy_fc1_.load_from(data, offset, size); enemy_fc2_.load_from(data, offset, size);
    ally_fc1_.load_from(data, offset, size);  ally_fc2_.load_from(data, offset, size);
}

static inline void copy_slice(const Tensor& src, int off, int n, Tensor& dst) {
    std::memcpy(dst.ptr(), src.ptr() + off, n * sizeof(float));
}
static inline void accum_slice(Tensor& dst_full, int off, const Tensor& slot_grad, int n) {
    float* d = dst_full.ptr() + off;
    const float* s = slot_grad.ptr();
    for (int i = 0; i < n; ++i) d[i] += s[i];
}

void DeepSetsEncoder::forward(const Tensor& x, Tensor& y) {
    assert(x.size() == observation::TOTAL);
    assert(y.size() == out_dim());
    x_cache_ = x;

    // Self stream.
    Tensor self_in = Tensor::vec(observation::SELF_FEATURES);
    copy_slice(x, 0, observation::SELF_FEATURES, self_in);
    Tensor tmp_h = Tensor::vec(cfg_.hidden);
    self_fc1_.forward(self_in, tmp_h);
    self_act_.forward(tmp_h, self_h_);
    self_fc2_.forward(self_h_, self_z_);

    // Enemy stream (per slot, masked pool).
    const int off_e = observation::SELF_FEATURES;
    e_n_valid_ = 0;
    Tensor pooled_e = Tensor::vec(cfg_.embed_dim);
    pooled_e.zero();
    Tensor slot_in = Tensor::vec(observation::ENEMY_FEATURES);
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        const bool valid = x[base] > 0.5f;   // [0] = valid flag
        e_valid_[k] = valid ? 1 : 0;
        if (!valid) continue;
        copy_slice(x, base, observation::ENEMY_FEATURES, slot_in);
        Tensor h_raw = Tensor::vec(cfg_.hidden);
        enemy_fc1_.forward(slot_in, h_raw);
        Relu act; act.forward(h_raw, e_h_[k]);    // ephemeral Relu — ok, stateless
        enemy_fc2_.forward(e_h_[k], e_z_[k]);
        for (int j = 0; j < cfg_.embed_dim; ++j) pooled_e[j] += e_z_[k][j];
        ++e_n_valid_;
    }
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        for (int j = 0; j < cfg_.embed_dim; ++j) pooled_e[j] *= inv;
    }

    // Ally stream.
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    a_n_valid_ = 0;
    Tensor pooled_a = Tensor::vec(cfg_.embed_dim);
    pooled_a.zero();
    Tensor slot_in_a = Tensor::vec(observation::ALLY_FEATURES);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        const bool valid = x[base] > 0.5f;
        a_valid_[k] = valid ? 1 : 0;
        if (!valid) continue;
        copy_slice(x, base, observation::ALLY_FEATURES, slot_in_a);
        Tensor h_raw = Tensor::vec(cfg_.hidden);
        ally_fc1_.forward(slot_in_a, h_raw);
        Relu act; act.forward(h_raw, a_h_[k]);
        ally_fc2_.forward(a_h_[k], a_z_[k]);
        for (int j = 0; j < cfg_.embed_dim; ++j) pooled_a[j] += a_z_[k][j];
        ++a_n_valid_;
    }
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        for (int j = 0; j < cfg_.embed_dim; ++j) pooled_a[j] *= inv;
    }

    // Concat y = [self_z, pooled_e, pooled_a].
    const int E = cfg_.embed_dim;
    for (int j = 0; j < E; ++j) y[0*E + j] = self_z_[j];
    for (int j = 0; j < E; ++j) y[1*E + j] = pooled_e[j];
    for (int j = 0; j < E; ++j) y[2*E + j] = pooled_a[j];
}

// Because forward uses ephemeral Relu activations for the per-slot streams,
// backward re-derives activation mask from cached h via the per-slot z/h.
// We cached e_h_ / a_h_ (post-relu), so a feature is active iff h[i] > 0.
// This makes backward correct without stashing a separate mask.
void DeepSetsEncoder::backward(const Tensor& dY, Tensor& dX) {
    assert(dY.size() == out_dim());
    assert(dX.size() == observation::TOTAL);
    dX.zero();
    const int E = cfg_.embed_dim;

    // Self backward.
    Tensor dSelfZ = Tensor::vec(E);
    for (int j = 0; j < E; ++j) dSelfZ[j] = dY[0*E + j];
    Tensor dSelfH = Tensor::vec(cfg_.hidden);
    self_fc2_.backward(dSelfZ, dSelfH);
    // derive relu mask from self_h_ (post-relu): active iff value > 0
    for (int i = 0; i < cfg_.hidden; ++i) if (self_h_[i] <= 0.0f) dSelfH[i] = 0.0f;
    Tensor dSelfIn = Tensor::vec(observation::SELF_FEATURES);
    self_fc1_.backward(dSelfH, dSelfIn);
    accum_slice(dX, 0, dSelfIn, observation::SELF_FEATURES);

    // Enemy backward.
    const int off_e = observation::SELF_FEATURES;
    if (e_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(e_n_valid_);
        Tensor dZk = Tensor::vec(E);
        Tensor dHk = Tensor::vec(cfg_.hidden);
        Tensor dSlot = Tensor::vec(observation::ENEMY_FEATURES);
        for (int k = 0; k < observation::K_ENEMIES; ++k) {
            if (!e_valid_[k]) continue;
            for (int j = 0; j < E; ++j) dZk[j] = dY[1*E + j] * inv;
            enemy_fc2_.backward(dZk, dHk);
            for (int i = 0; i < cfg_.hidden; ++i) if (e_h_[k][i] <= 0.0f) dHk[i] = 0.0f;
            enemy_fc1_.backward(dHk, dSlot);
            accum_slice(dX, off_e + k * observation::ENEMY_FEATURES, dSlot,
                        observation::ENEMY_FEATURES);
        }
    }

    // Ally backward.
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    if (a_n_valid_ > 0) {
        const float inv = 1.0f / static_cast<float>(a_n_valid_);
        Tensor dZk = Tensor::vec(E);
        Tensor dHk = Tensor::vec(cfg_.hidden);
        Tensor dSlot = Tensor::vec(observation::ALLY_FEATURES);
        for (int k = 0; k < observation::K_ALLIES; ++k) {
            if (!a_valid_[k]) continue;
            for (int j = 0; j < E; ++j) dZk[j] = dY[2*E + j] * inv;
            ally_fc2_.backward(dZk, dHk);
            for (int i = 0; i < cfg_.hidden; ++i) if (a_h_[k][i] <= 0.0f) dHk[i] = 0.0f;
            ally_fc1_.backward(dHk, dSlot);
            accum_slice(dX, off_a + k * observation::ALLY_FEATURES, dSlot,
                        observation::ALLY_FEATURES);
        }
    }
}

} // namespace brogameagent::nn
