#include "brogameagent/nn/decoder.h"

#include <brotensor/ops.h>

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
    self_out_.resize(observation::SELF_FEATURES, 1);

    e_in_.resize(observation::K_ENEMIES, cfg.embed_dim);
    e_h_raw_.resize(observation::K_ENEMIES, cfg.hidden);
    e_h_.resize(observation::K_ENEMIES, cfg.hidden);
    e_out_.resize(observation::K_ENEMIES, observation::ENEMY_FEATURES);

    a_in_.resize(observation::K_ALLIES, cfg.embed_dim);
    a_h_raw_.resize(observation::K_ALLIES, cfg.hidden);
    a_h_.resize(observation::K_ALLIES, cfg.hidden);
    a_out_.resize(observation::K_ALLIES, observation::ALLY_FEATURES);

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

void DeepSetsDecoder::adam_step(float lr, float b1, float b2, float eps, int step) {
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_fc1_.adam_step(lr, b1, b2, eps, step); enemy_fc2_.adam_step(lr, b1, b2, eps, step);
    ally_fc1_.adam_step(lr, b1, b2, eps, step);  ally_fc2_.adam_step(lr, b1, b2, eps, step);
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

void DeepSetsDecoder::to(brotensor::Device d) {
    if (d == device_) return;
    self_fc1_.to(d);  self_fc2_.to(d);
    enemy_fc1_.to(d); enemy_fc2_.to(d);
    ally_fc1_.to(d);  ally_fc2_.to(d);
    self_h_raw_ = self_h_raw_.to(d);
    self_h_     = self_h_.to(d);
    self_out_   = self_out_.to(d);
    e_in_    = e_in_.to(d);    e_h_raw_ = e_h_raw_.to(d);
    e_h_     = e_h_.to(d);     e_out_   = e_out_.to(d);
    a_in_    = a_in_.to(d);    a_h_raw_ = a_h_raw_.to(d);
    a_h_     = a_h_.to(d);     a_out_   = a_out_.to(d);
    if (self_in_.size() > 0)  self_in_  = self_in_.to(d);
    if (pooled_e_.size() > 0) pooled_e_ = pooled_e_.to(d);
    if (pooled_a_.size() > 0) pooled_a_ = pooled_a_.to(d);
    device_ = d;
}

// ─── forward ───────────────────────────────────────────────────────────────
//
// Device-neutral. Per-slot streams broadcast a single pooled embedding into a
// (K_slots, embed_dim) minibatch and run a shared-weight Linear+ReLU+Linear
// over it via the batched ops. No host reads of tensor values.
void DeepSetsDecoder::forward(const brotensor::Tensor& x, brotensor::Tensor& y) {
    assert(x.size() == in_dim());
    assert(y.size() == out_dim());
    const int E = cfg_.embed_dim;

    // Split input into three embed-sized chunks; cache for backward.
    brotensor::copy_d2d(x, 0 * E, self_in_,  0, E);
    brotensor::copy_d2d(x, 1 * E, pooled_e_, 0, E);
    brotensor::copy_d2d(x, 2 * E, pooled_a_, 0, E);

    // --- Self stream ---
    self_fc1_.forward(self_in_, self_h_raw_);
    brotensor::relu_forward(self_h_raw_, self_h_);
    self_fc2_.forward(self_h_, self_out_);
    brotensor::copy_d2d(self_out_, 0, y, 0, observation::SELF_FEATURES);

    // --- Enemy stream (broadcast pooled_e to each slot) ---
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        brotensor::copy_d2d(pooled_e_, 0, e_in_, k * E, E);
    }
    enemy_fc1_.forward_batched_train(e_in_, e_h_raw_);
    brotensor::relu_forward_batched(e_h_raw_, e_h_);
    enemy_fc2_.forward_batched_train(e_h_, e_out_);
    brotensor::copy_d2d(e_out_, 0, y, off_e,
                        observation::K_ENEMIES * observation::ENEMY_FEATURES);

    // --- Ally stream (broadcast pooled_a to each slot) ---
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        brotensor::copy_d2d(pooled_a_, 0, a_in_, k * E, E);
    }
    ally_fc1_.forward_batched_train(a_in_, a_h_raw_);
    brotensor::relu_forward_batched(a_h_raw_, a_h_);
    ally_fc2_.forward_batched_train(a_h_, a_out_);
    brotensor::copy_d2d(a_out_, 0, y, off_a,
                        observation::K_ALLIES * observation::ALLY_FEATURES);
}

// ─── backward ──────────────────────────────────────────────────────────────
void DeepSetsDecoder::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    assert(dY.size() == out_dim());
    assert(dX.size() == in_dim());
    if (dX.rows != in_dim() || dX.cols != 1 || dX.device != dY.device) {
        dX = brotensor::Tensor::zeros_on(dY.device, in_dim(), 1);
    } else {
        dX.zero();
    }
    const int E = cfg_.embed_dim;

    // --- Self stream ---
    brotensor::Tensor dSelfOut = brotensor::Tensor::zeros_on(
        dY.device, observation::SELF_FEATURES, 1);
    brotensor::copy_d2d(dY, 0, dSelfOut, 0, observation::SELF_FEATURES);
    brotensor::Tensor dSelfH  = brotensor::Tensor::zeros_on(dY.device, cfg_.hidden, 1);
    brotensor::Tensor dSelfHp = brotensor::Tensor::zeros_on(dY.device, cfg_.hidden, 1);
    self_fc2_.backward(self_h_, dSelfOut, dSelfH);
    brotensor::relu_backward(self_h_raw_, dSelfH, dSelfHp);
    brotensor::Tensor dSelfIn = brotensor::Tensor::zeros_on(dY.device, E, 1);
    self_fc1_.backward(self_in_, dSelfHp, dSelfIn);
    brotensor::copy_d2d(dSelfIn, 0, dX, 0 * E, E);

    // --- Enemy stream — per-slot backward, sum slot grads into pooled_e grad.
    // Each slot shares pooled_e_ as its input; the encoder-side grad is the
    // SUM of every slot's input gradient. masked_mean_pool with a null mask
    // gives the mean over K rows; multiply by K to recover the sum.
    const int off_e = observation::SELF_FEATURES;
    brotensor::Tensor dEout = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, observation::ENEMY_FEATURES);
    brotensor::copy_d2d(dY, off_e, dEout, 0,
                        observation::K_ENEMIES * observation::ENEMY_FEATURES);
    brotensor::Tensor dEh  = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, cfg_.hidden);
    brotensor::Tensor dEhp = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, cfg_.hidden);
    enemy_fc2_.backward_batched(dEout, dEh);
    brotensor::relu_backward_batched(e_h_raw_, dEh, dEhp);
    brotensor::Tensor dEin = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ENEMIES, E);
    enemy_fc1_.backward_batched(dEhp, dEin);
    brotensor::Tensor dEnc = brotensor::Tensor::zeros_on(dY.device, E, 1);
    brotensor::masked_mean_pool_forward(dEin, nullptr, dEnc);
    brotensor::scale_inplace(dEnc, static_cast<float>(observation::K_ENEMIES));
    brotensor::copy_d2d(dEnc, 0, dX, 1 * E, E);

    // --- Ally stream ---
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    brotensor::Tensor dAout = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, observation::ALLY_FEATURES);
    brotensor::copy_d2d(dY, off_a, dAout, 0,
                        observation::K_ALLIES * observation::ALLY_FEATURES);
    brotensor::Tensor dAh  = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, cfg_.hidden);
    brotensor::Tensor dAhp = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, cfg_.hidden);
    ally_fc2_.backward_batched(dAout, dAh);
    brotensor::relu_backward_batched(a_h_raw_, dAh, dAhp);
    brotensor::Tensor dAin = brotensor::Tensor::zeros_on(
        dY.device, observation::K_ALLIES, E);
    ally_fc1_.backward_batched(dAhp, dAin);
    brotensor::Tensor dEncA = brotensor::Tensor::zeros_on(dY.device, E, 1);
    brotensor::masked_mean_pool_forward(dAin, nullptr, dEncA);
    brotensor::scale_inplace(dEncA, static_cast<float>(observation::K_ALLIES));
    brotensor::copy_d2d(dEncA, 0, dX, 2 * E, E);
}

} // namespace brogameagent::nn
