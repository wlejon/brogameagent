#include "brogameagent/nn/encoder.h"

#include <brotensor/ops.h>

#include <algorithm>
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

    self_in_.resize(observation::SELF_FEATURES, 1);
    self_h_pre_.resize(cfg.hidden, 1);
    self_h_.resize(cfg.hidden, 1);
    self_z_.resize(cfg.embed_dim, 1);

    // Batched per-slot caches: (K, FEATURES) / (K, hidden) / (K, embed).
    e_in_.resize(observation::K_ENEMIES, observation::ENEMY_FEATURES);
    e_hpre_.resize(observation::K_ENEMIES, cfg.hidden);
    e_h_.resize(observation::K_ENEMIES, cfg.hidden);
    e_z_.resize(observation::K_ENEMIES, cfg.embed_dim);
    e_mask_.resize(observation::K_ENEMIES, 1);

    a_in_.resize(observation::K_ALLIES, observation::ALLY_FEATURES);
    a_hpre_.resize(observation::K_ALLIES, cfg.hidden);
    a_h_.resize(observation::K_ALLIES, cfg.hidden);
    a_z_.resize(observation::K_ALLIES, cfg.embed_dim);
    a_mask_.resize(observation::K_ALLIES, 1);

    x_cache_.resize(observation::TOTAL, 1);
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

void DeepSetsEncoder::adam_step(float lr, float b1, float b2, float eps, int step) {
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_fc1_.adam_step(lr, b1, b2, eps, step); enemy_fc2_.adam_step(lr, b1, b2, eps, step);
    ally_fc1_.adam_step(lr, b1, b2, eps, step);  ally_fc2_.adam_step(lr, b1, b2, eps, step);
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

void DeepSetsEncoder::to(brotensor::Device d) {
    if (d == device_) return;
    self_fc1_.to(d);  self_fc2_.to(d);
    enemy_fc1_.to(d); enemy_fc2_.to(d);
    ally_fc1_.to(d);  ally_fc2_.to(d);
    self_in_     = self_in_.to(d);
    self_h_pre_  = self_h_pre_.to(d);
    self_h_      = self_h_.to(d);
    self_z_      = self_z_.to(d);
    e_in_   = e_in_.to(d);   e_hpre_ = e_hpre_.to(d);
    e_h_    = e_h_.to(d);    e_z_    = e_z_.to(d);    e_mask_ = e_mask_.to(d);
    a_in_   = a_in_.to(d);   a_hpre_ = a_hpre_.to(d);
    a_h_    = a_h_.to(d);    a_z_    = a_z_.to(d);    a_mask_ = a_mask_.to(d);
    if (x_cache_.size() > 0) x_cache_ = x_cache_.to(d);
    device_ = d;
}

// ─── forward ───────────────────────────────────────────────────────────────
//
// Device-neutral: every op below dispatches on operand device. Per-slot
// streams are processed as a single (K_slots, FEATURES) minibatch through a
// shared-weight Linear + ReLU + Linear, then masked-mean-pooled. No host
// reads of tensor values — slot slicing uses copy_d2d, mask construction uses
// build_slot_mask, both device-dispatched.
void DeepSetsEncoder::forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
    assert(x.size() == observation::TOTAL);
    assert(y.size() == out_dim());
    x_cache_ = x;
    const int E = cfg_.embed_dim;

    // ── Self stream ──
    brotensor::copy_d2d(x, 0, self_in_, 0, observation::SELF_FEATURES);
    self_fc1_.forward(self_in_, self_h_pre_);
    self_act_.forward(self_h_pre_, self_h_);
    self_fc2_.forward(self_h_, self_z_);

    // ── Enemy stream (per-slot shared MLP + masked mean-pool) ──
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        brotensor::copy_d2d(x, off_e + k * observation::ENEMY_FEATURES,
                            e_in_, k * observation::ENEMY_FEATURES,
                            observation::ENEMY_FEATURES);
    }
    brotensor::build_slot_mask(x, off_e, observation::K_ENEMIES,
                               observation::ENEMY_FEATURES, e_mask_);
    enemy_fc1_.forward_batched_train(e_in_, e_hpre_);
    brotensor::relu_forward_batched(e_hpre_, e_h_);
    enemy_fc2_.forward_batched_train(e_h_, e_z_);
    brotensor::Tensor pooled_e = brotensor::Tensor::zeros_on(x.device, E, 1);
    brotensor::masked_mean_pool_forward(
        e_z_, static_cast<const float*>(e_mask_.data), pooled_e);

    // ── Ally stream ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        brotensor::copy_d2d(x, off_a + k * observation::ALLY_FEATURES,
                            a_in_, k * observation::ALLY_FEATURES,
                            observation::ALLY_FEATURES);
    }
    brotensor::build_slot_mask(x, off_a, observation::K_ALLIES,
                               observation::ALLY_FEATURES, a_mask_);
    ally_fc1_.forward_batched_train(a_in_, a_hpre_);
    brotensor::relu_forward_batched(a_hpre_, a_h_);
    ally_fc2_.forward_batched_train(a_h_, a_z_);
    brotensor::Tensor pooled_a = brotensor::Tensor::zeros_on(x.device, E, 1);
    brotensor::masked_mean_pool_forward(
        a_z_, static_cast<const float*>(a_mask_.data), pooled_a);

    // ── Concat y = [self_z, pooled_e, pooled_a] ──
    brotensor::copy_d2d(self_z_,  0, y, 0 * E, E);
    brotensor::copy_d2d(pooled_e, 0, y, 1 * E, E);
    brotensor::copy_d2d(pooled_a, 0, y, 2 * E, E);
}

// ─── backward ──────────────────────────────────────────────────────────────
void DeepSetsEncoder::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    assert(dY.size() == out_dim());
    assert(dX.size() == observation::TOTAL);
    if (dX.rows != observation::TOTAL || dX.cols != 1 || dX.device != dY.device) {
        dX = brotensor::Tensor::zeros_on(dY.device, observation::TOTAL, 1);
    } else {
        dX.zero();
    }
    const int E = cfg_.embed_dim;

    // ── Self backward ──
    brotensor::Tensor dSelfZ = brotensor::Tensor::zeros_on(dY.device, E, 1);
    brotensor::copy_d2d(dY, 0 * E, dSelfZ, 0, E);
    brotensor::Tensor dSelfH  = brotensor::Tensor::zeros_on(dY.device, cfg_.hidden, 1);
    brotensor::Tensor dSelfHp = brotensor::Tensor::zeros_on(dY.device, cfg_.hidden, 1);
    self_fc2_.backward(self_h_, dSelfZ, dSelfH);
    self_act_.backward(dSelfH, dSelfHp);
    brotensor::Tensor dSelfIn = brotensor::Tensor::zeros_on(
        dY.device, observation::SELF_FEATURES, 1);
    self_fc1_.backward(self_in_, dSelfHp, dSelfIn);
    brotensor::copy_d2d(dSelfIn, 0, dX, 0, observation::SELF_FEATURES);

    // ── Enemy backward ──
    // dY for the enemy embedding is the pooled grad; masked_mean_pool_backward
    // broadcasts it across valid slot rows (1/n_valid each, 0 on invalid).
    const int off_e = observation::SELF_FEATURES;
    brotensor::Tensor dPoolE = brotensor::Tensor::zeros_on(dY.device, E, 1);
    brotensor::copy_d2d(dY, 1 * E, dPoolE, 0, E);
    brotensor::Tensor dEz = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, E);
    brotensor::masked_mean_pool_backward(
        dPoolE, static_cast<const float*>(e_mask_.data),
        observation::K_ENEMIES, dEz);
    brotensor::Tensor dEh  = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, cfg_.hidden);
    brotensor::Tensor dEhp = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, cfg_.hidden);
    enemy_fc2_.backward_batched(dEz, dEh);
    brotensor::relu_backward_batched(e_hpre_, dEh, dEhp);
    brotensor::Tensor dEin = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, observation::ENEMY_FEATURES);
    enemy_fc1_.backward_batched(dEhp, dEin);
    // dEin is (K, ENEMY_FEATURES) row-major == TOTAL-block contiguous layout.
    brotensor::copy_d2d(dEin, 0, dX, off_e,
                        observation::K_ENEMIES * observation::ENEMY_FEATURES);

    // ── Ally backward ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    brotensor::Tensor dPoolA = brotensor::Tensor::zeros_on(dY.device, E, 1);
    brotensor::copy_d2d(dY, 2 * E, dPoolA, 0, E);
    brotensor::Tensor dAz = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, E);
    brotensor::masked_mean_pool_backward(
        dPoolA, static_cast<const float*>(a_mask_.data),
        observation::K_ALLIES, dAz);
    brotensor::Tensor dAh  = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, cfg_.hidden);
    brotensor::Tensor dAhp = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, cfg_.hidden);
    ally_fc2_.backward_batched(dAz, dAh);
    brotensor::relu_backward_batched(a_hpre_, dAh, dAhp);
    brotensor::Tensor dAin = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, observation::ALLY_FEATURES);
    ally_fc1_.backward_batched(dAhp, dAin);
    brotensor::copy_d2d(dAin, 0, dX, off_a,
                        observation::K_ALLIES * observation::ALLY_FEATURES);
}

} // namespace brogameagent::nn
