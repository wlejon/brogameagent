#include "brogameagent/nn/decoder.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void DeepSetsDecoder::init(const Config& cfg, uint64_t& rng_state) {
    cfg_ = cfg;

    self_fc1_.init(cfg.embed_dim, cfg.hidden,                   rng_state);
    self_fc2_.init(cfg.hidden,    observation::SELF_FEATURES,   rng_state);

    enemy_fc1_.init(cfg.embed_dim, cfg.hidden,                  rng_state);
    enemy_fc2_.init(cfg.hidden,    observation::ENEMY_FEATURES, rng_state);

    ally_fc1_.init(cfg.embed_dim, cfg.hidden,                   rng_state);
    ally_fc2_.init(cfg.hidden,    observation::ALLY_FEATURES,   rng_state);

    self_h_raw_.resize(cfg.hidden, 1);
    self_h_.resize(cfg.hidden, 1);

    e_h_raw_.assign(observation::K_ENEMIES, Tensor::vec(cfg.hidden));
    e_h_.assign(observation::K_ENEMIES, Tensor::vec(cfg.hidden));
    a_h_raw_.assign(observation::K_ALLIES, Tensor::vec(cfg.hidden));
    a_h_.assign(observation::K_ALLIES, Tensor::vec(cfg.hidden));

    self_in_.resize(cfg.embed_dim, 1);
    pooled_e_.resize(cfg.embed_dim, 1);
    pooled_a_.resize(cfg.embed_dim, 1);
}

int DeepSetsDecoder::num_params() const {
    return self_fc1_.num_params() + self_fc2_.num_params()
         + enemy_fc1_.num_params() + enemy_fc2_.num_params()
         + ally_fc1_.num_params()  + ally_fc2_.num_params();
}

void DeepSetsDecoder::zero_grad() {
    self_fc1_.zero_grad(); self_fc2_.zero_grad();
    enemy_fc1_.zero_grad(); enemy_fc2_.zero_grad();
    ally_fc1_.zero_grad();  ally_fc2_.zero_grad();
}

void DeepSetsDecoder::sgd_step(float lr, float momentum) {
    self_fc1_.sgd_step(lr, momentum); self_fc2_.sgd_step(lr, momentum);
    enemy_fc1_.sgd_step(lr, momentum); enemy_fc2_.sgd_step(lr, momentum);
    ally_fc1_.sgd_step(lr, momentum);  ally_fc2_.sgd_step(lr, momentum);
}

void DeepSetsDecoder::save_to(std::vector<uint8_t>& out) const {
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_fc1_.save_to(out); enemy_fc2_.save_to(out);
    ally_fc1_.save_to(out);  ally_fc2_.save_to(out);
}

void DeepSetsDecoder::load_from(const uint8_t* data, size_t& offset, size_t size) {
    self_fc1_.load_from(data, offset, size);  self_fc2_.load_from(data, offset, size);
    enemy_fc1_.load_from(data, offset, size); enemy_fc2_.load_from(data, offset, size);
    ally_fc1_.load_from(data, offset, size);  ally_fc2_.load_from(data, offset, size);
}

static inline void relu_inplace(const Tensor& src, Tensor& dst) {
    const int n = src.size();
    for (int i = 0; i < n; ++i) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void DeepSetsDecoder::forward(const Tensor& x, Tensor& y) {
    assert(x.size() == in_dim());
    assert(y.size() == out_dim());
    const int E = cfg_.embed_dim;

    // Split input into three embed-sized chunks; cache for backward.
    for (int j = 0; j < E; ++j) self_in_[j]  = x[0*E + j];
    for (int j = 0; j < E; ++j) pooled_e_[j] = x[1*E + j];
    for (int j = 0; j < E; ++j) pooled_a_[j] = x[2*E + j];

    // --- Self stream ---
    self_fc1_.forward(self_in_, self_h_raw_);
    relu_inplace(self_h_raw_, self_h_);
    Tensor self_out = Tensor::vec(observation::SELF_FEATURES);
    self_fc2_.forward(self_h_, self_out);
    for (int j = 0; j < observation::SELF_FEATURES; ++j) y[j] = self_out[j];

    // --- Enemy stream (broadcast pooled_e to each slot) ---
    const int off_e = observation::SELF_FEATURES;
    Tensor slot_out = Tensor::vec(observation::ENEMY_FEATURES);
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        enemy_fc1_.forward(pooled_e_, e_h_raw_[k]);
        relu_inplace(e_h_raw_[k], e_h_[k]);
        enemy_fc2_.forward(e_h_[k], slot_out);
        const int base = off_e + k * observation::ENEMY_FEATURES;
        for (int j = 0; j < observation::ENEMY_FEATURES; ++j) y[base + j] = slot_out[j];
    }

    // --- Ally stream (broadcast pooled_a to each slot) ---
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    Tensor slot_out_a = Tensor::vec(observation::ALLY_FEATURES);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        ally_fc1_.forward(pooled_a_, a_h_raw_[k]);
        relu_inplace(a_h_raw_[k], a_h_[k]);
        ally_fc2_.forward(a_h_[k], slot_out_a);
        const int base = off_a + k * observation::ALLY_FEATURES;
        for (int j = 0; j < observation::ALLY_FEATURES; ++j) y[base + j] = slot_out_a[j];
    }
}

void DeepSetsDecoder::backward(const Tensor& dY, Tensor& dX) {
    assert(dY.size() == out_dim());
    assert(dX.size() == in_dim());
    dX.zero();
    const int E = cfg_.embed_dim;

    // --- Self stream ---
    Tensor dSelfOut = Tensor::vec(observation::SELF_FEATURES);
    for (int j = 0; j < observation::SELF_FEATURES; ++j) dSelfOut[j] = dY[j];
    Tensor dSelfH = Tensor::vec(cfg_.hidden);
    self_fc2_.backward(dSelfOut, dSelfH);
    // relu mask from raw pre-activation
    for (int i = 0; i < cfg_.hidden; ++i) if (self_h_raw_[i] <= 0.0f) dSelfH[i] = 0.0f;
    Tensor dSelfIn = Tensor::vec(E);
    self_fc1_.backward(dSelfH, dSelfIn);
    for (int j = 0; j < E; ++j) dX[0*E + j] += dSelfIn[j];

    // --- Enemy stream — per-slot backward, accumulate into pooled_e gradient ---
    const int off_e = observation::SELF_FEATURES;
    Tensor dSlot = Tensor::vec(observation::ENEMY_FEATURES);
    Tensor dHk   = Tensor::vec(cfg_.hidden);
    Tensor dEnc  = Tensor::vec(E);
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        for (int j = 0; j < observation::ENEMY_FEATURES; ++j) dSlot[j] = dY[base + j];
        enemy_fc2_.backward(dSlot, dHk);
        for (int i = 0; i < cfg_.hidden; ++i) if (e_h_raw_[k][i] <= 0.0f) dHk[i] = 0.0f;
        enemy_fc1_.backward(dHk, dEnc);
        for (int j = 0; j < E; ++j) dX[1*E + j] += dEnc[j];
    }

    // --- Ally stream ---
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    Tensor dSlotA = Tensor::vec(observation::ALLY_FEATURES);
    Tensor dHkA   = Tensor::vec(cfg_.hidden);
    Tensor dEncA  = Tensor::vec(E);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        for (int j = 0; j < observation::ALLY_FEATURES; ++j) dSlotA[j] = dY[base + j];
        ally_fc2_.backward(dSlotA, dHkA);
        for (int i = 0; i < cfg_.hidden; ++i) if (a_h_raw_[k][i] <= 0.0f) dHkA[i] = 0.0f;
        ally_fc1_.backward(dHkA, dEncA);
        for (int j = 0; j < E; ++j) dX[2*E + j] += dEncA[j];
    }
}

} // namespace brogameagent::nn
