#include "brogameagent/nn/transformer_encoder.h"

#include <brotensor/ops.h>

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void TransformerEncoder::init(const Config& cfg, uint64_t& rng_state) {
    cfg_ = cfg;
    blocks_.clear();
    blocks_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        auto blk = std::make_unique<TransformerBlock>();
        TransformerBlock::Config bcfg{};
        bcfg.dim       = cfg.dim;
        bcfg.num_heads = cfg.num_heads;
        bcfg.d_ff      = cfg.d_ff;
        bcfg.n_slots   = cfg.n_slots;
        bcfg.ln_eps    = cfg.ln_eps;
        bcfg.norm      = cfg.norm;
        blk->init(bcfg, rng_state);
        blocks_.push_back(std::move(blk));
    }
    activations_.assign(cfg.n_layers + 1, brotensor::Tensor());
    has_final_ln_ = (cfg.norm == NormPlacement::PreNorm);
    if (has_final_ln_) final_ln_.init(cfg.dim, cfg.ln_eps);
}

int TransformerEncoder::num_params() const {
    int n = 0;
    for (const auto& b : blocks_) n += b->num_params();
    if (has_final_ln_) n += final_ln_.num_params();
    return n;
}

void TransformerEncoder::forward(const brotensor::Tensor& X, const float* mask, brotensor::Tensor& Y) {
    const int K = X.rows;
    const int D = X.cols;
    assert(D == cfg_.dim);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);

    activations_[0] = X;
    for (int i = 0; i < cfg_.n_layers; ++i) {
        if (activations_[i + 1].rows != K || activations_[i + 1].cols != D) {
            activations_[i + 1].resize(K, D);
        }
        blocks_[i]->forward(activations_[i], mask, activations_[i + 1]);
    }
    const brotensor::Tensor& last = activations_[cfg_.n_layers];
    if (has_final_ln_) {
        if (pre_final_ln_.rows != K || pre_final_ln_.cols != D) pre_final_ln_.resize(K, D);
        pre_final_ln_ = last;
        final_ln_.forward(pre_final_ln_, Y);
    } else {
        // Copy last → Y (device-aware deep copy).
        Y = last;
    }
}

void TransformerEncoder::backward(const brotensor::Tensor& dY, brotensor::Tensor& dX) {
    const int K = dY.rows;
    const int D = dY.cols;
    brotensor::Tensor d_top = brotensor::Tensor::zeros_on(dY.device, K, D);
    if (has_final_ln_) {
        final_ln_.backward(dY, d_top);
    } else {
        d_top = dY;
    }

    brotensor::Tensor d_cur = std::move(d_top);
    for (int i = cfg_.n_layers - 1; i >= 0; --i) {
        brotensor::Tensor d_in = brotensor::Tensor::zeros_on(dY.device, K, D);
        blocks_[i]->backward(d_cur, d_in);
        d_cur = std::move(d_in);
    }
    dX = std::move(d_cur);
}

void TransformerEncoder::to(brotensor::Device d) {
    if (d == device_) return;
    for (auto& b : blocks_) b->to(d);
    if (has_final_ln_) final_ln_.to(d);
    for (auto& t : activations_) t = t.to(d);
    pre_final_ln_ = pre_final_ln_.to(d);
    device_ = d;
}

void TransformerEncoder::zero_grad() {
    for (auto& b : blocks_) b->zero_grad();
    if (has_final_ln_) final_ln_.zero_grad();
}

void TransformerEncoder::adam_step(float lr, float beta1, float beta2,
                                   float eps, int step) {
    for (auto& b : blocks_) b->adam_step(lr, beta1, beta2, eps, step);
    if (has_final_ln_) final_ln_.adam_step(lr, beta1, beta2, eps, step);
}

void TransformerEncoder::sgd_step(float lr, float momentum) {
    for (auto& b : blocks_) b->sgd_step(lr, momentum);
    if (has_final_ln_) final_ln_.sgd_step(lr, momentum);
}

void TransformerEncoder::save_to(std::vector<uint8_t>& out) const {
    const int32_t n = static_cast<int32_t>(blocks_.size());
    const size_t start = out.size();
    out.resize(start + sizeof(int32_t));
    std::memcpy(out.data() + start, &n, sizeof(int32_t));
    for (const auto& b : blocks_) b->save_to(out);
    if (has_final_ln_) final_ln_.save_to(out);
}

void TransformerEncoder::load_from(const uint8_t* data, size_t& offset, size_t size) {
    assert(offset + sizeof(int32_t) <= size);
    int32_t n = 0;
    std::memcpy(&n, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);
    assert(static_cast<int>(n) == cfg_.n_layers
           && "TransformerEncoder must be init()'d with matching n_layers before load");
    for (auto& b : blocks_) b->load_from(data, offset, size);
    if (has_final_ln_) final_ln_.load_from(data, offset, size);
}

} // namespace brogameagent::nn
