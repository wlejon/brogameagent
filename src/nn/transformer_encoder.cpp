#include "brogameagent/nn/transformer_encoder.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/runtime.h"
#endif

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
    activations_.assign(cfg.n_layers + 1, Tensor());
    has_final_ln_ = (cfg.norm == NormPlacement::PreNorm);
    if (has_final_ln_) final_ln_.init(cfg.dim, cfg.ln_eps);
}

int TransformerEncoder::num_params() const {
    int n = 0;
    for (const auto& b : blocks_) n += b->num_params();
    if (has_final_ln_) n += final_ln_.num_params();
    return n;
}

void TransformerEncoder::forward(const Tensor& X, const float* mask, Tensor& Y) {
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
    const Tensor& last = activations_[cfg_.n_layers];
    if (has_final_ln_) {
        if (pre_final_ln_.rows != K || pre_final_ln_.cols != D) pre_final_ln_.resize(K, D);
        pre_final_ln_ = last;
        final_ln_.forward(pre_final_ln_, Y);
    } else {
        // Copy last → Y.
        if (Y.rows != K || Y.cols != D) Y.resize(K, D);
        std::memcpy(Y.ptr(), last.ptr(), sizeof(float) * K * D);
    }
}

void TransformerEncoder::backward(const Tensor& dY, Tensor& dX) {
    const int K = dY.rows;
    const int D = dY.cols;
    Tensor d_top(K, D);
    if (has_final_ln_) {
        final_ln_.backward(dY, d_top);
    } else {
        d_top = dY;
    }

    Tensor d_cur = std::move(d_top);
    for (int i = cfg_.n_layers - 1; i >= 0; --i) {
        Tensor d_in(K, D);
        blocks_[i]->backward(d_cur, d_in);
        d_cur = std::move(d_in);
    }
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    std::memcpy(dX.ptr(), d_cur.ptr(), sizeof(float) * K * D);
}

#ifdef BGA_HAS_CUDA
void TransformerEncoder::forward(const gpu::GpuTensor& X, const float* mask_dev,
                                 gpu::GpuTensor& Y) {
    assert(device_ == Device::GPU);
    const int K = X.rows;
    const int D = X.cols;
    assert(D == cfg_.dim);
    if (Y.rows != K || Y.cols != D) Y.resize(K, D);

    // activations_g_ has size n_layers + 1; [0] = X clone, [i+1] = block i out.
    if ((int)activations_g_.size() != cfg_.n_layers + 1) {
        activations_g_.clear();
        activations_g_.reserve(cfg_.n_layers + 1);
        for (int i = 0; i <= cfg_.n_layers; ++i) {
            activations_g_.emplace_back(K, D);
        }
    } else {
        for (auto& a : activations_g_) {
            if (a.rows != K || a.cols != D) a.resize(K, D);
        }
    }
    activations_g_[0] = X.clone();
    for (int i = 0; i < cfg_.n_layers; ++i) {
        blocks_[i]->forward(activations_g_[i], mask_dev, activations_g_[i + 1]);
    }
    const gpu::GpuTensor& last = activations_g_[cfg_.n_layers];
    if (has_final_ln_) {
        if (pre_final_ln_g_.rows != K || pre_final_ln_g_.cols != D) {
            pre_final_ln_g_.resize(K, D);
        }
        pre_final_ln_g_ = last.clone();
        final_ln_.forward(pre_final_ln_g_, Y);
    } else {
        // Y = last (clone copy).
        Y = last.clone();
    }
}

void TransformerEncoder::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    const int K = dY.rows;
    const int D = dY.cols;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);

    gpu::GpuTensor d_top(K, D);
    if (has_final_ln_) {
        final_ln_.backward(dY, d_top);
    } else {
        d_top = dY.clone();
    }

    gpu::GpuTensor d_cur = std::move(d_top);
    for (int i = cfg_.n_layers - 1; i >= 0; --i) {
        gpu::GpuTensor d_in(K, D);
        blocks_[i]->backward(d_cur, d_in);
        d_cur = std::move(d_in);
    }
    dX = std::move(d_cur);
}
#endif

void TransformerEncoder::to(Device d) {
    if (d == device_) return;
    device_require_cuda("TransformerEncoder");
    for (auto& b : blocks_) b->to(d);
    if (has_final_ln_) final_ln_.to(d);
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
