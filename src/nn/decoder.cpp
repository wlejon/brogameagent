#include "brogameagent/nn/decoder.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

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
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        self_dW1_g_.zero(); self_db1_g_.zero();
        self_dW2_g_.zero(); self_db2_g_.zero();
        enemy_dW1_g_.zero(); enemy_db1_g_.zero();
        enemy_dW2_g_.zero(); enemy_db2_g_.zero();
        ally_dW1_g_.zero(); ally_db1_g_.zero();
        ally_dW2_g_.zero(); ally_db2_g_.zero();
        return;
    }
#endif
    self_fc1_.zero_grad(); self_fc2_.zero_grad();
    enemy_fc1_.zero_grad(); enemy_fc2_.zero_grad();
    ally_fc1_.zero_grad();  ally_fc2_.zero_grad();
}

void DeepSetsDecoder::sgd_step(float lr, float momentum) {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::sgd_step_gpu(self_W1_g_, self_dW1_g_, self_vW1_g_, lr, momentum);
        gpu::sgd_step_gpu(self_b1_g_, self_db1_g_, self_vb1_g_, lr, momentum);
        gpu::sgd_step_gpu(self_W2_g_, self_dW2_g_, self_vW2_g_, lr, momentum);
        gpu::sgd_step_gpu(self_b2_g_, self_db2_g_, self_vb2_g_, lr, momentum);
        gpu::sgd_step_gpu(enemy_W1_g_, enemy_dW1_g_, enemy_vW1_g_, lr, momentum);
        gpu::sgd_step_gpu(enemy_b1_g_, enemy_db1_g_, enemy_vb1_g_, lr, momentum);
        gpu::sgd_step_gpu(enemy_W2_g_, enemy_dW2_g_, enemy_vW2_g_, lr, momentum);
        gpu::sgd_step_gpu(enemy_b2_g_, enemy_db2_g_, enemy_vb2_g_, lr, momentum);
        gpu::sgd_step_gpu(ally_W1_g_, ally_dW1_g_, ally_vW1_g_, lr, momentum);
        gpu::sgd_step_gpu(ally_b1_g_, ally_db1_g_, ally_vb1_g_, lr, momentum);
        gpu::sgd_step_gpu(ally_W2_g_, ally_dW2_g_, ally_vW2_g_, lr, momentum);
        gpu::sgd_step_gpu(ally_b2_g_, ally_db2_g_, ally_vb2_g_, lr, momentum);
        return;
    }
#endif
    self_fc1_.sgd_step(lr, momentum); self_fc2_.sgd_step(lr, momentum);
    enemy_fc1_.sgd_step(lr, momentum); enemy_fc2_.sgd_step(lr, momentum);
    ally_fc1_.sgd_step(lr, momentum);  ally_fc2_.sgd_step(lr, momentum);
}

void DeepSetsDecoder::adam_step(float lr, float b1, float b2, float eps, int step) {
    // Adam path operates on host Linear members; GPU dispatch uses sgd_step.
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_fc1_.adam_step(lr, b1, b2, eps, step); enemy_fc2_.adam_step(lr, b1, b2, eps, step);
    ally_fc1_.adam_step(lr, b1, b2, eps, step);  ally_fc2_.adam_step(lr, b1, b2, eps, step);
}

void DeepSetsDecoder::save_to(std::vector<uint8_t>& out) const {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        auto* self = const_cast<DeepSetsDecoder*>(this);
        gpu::download(self_W1_g_, self->self_fc1_.W()); gpu::download(self_b1_g_, self->self_fc1_.b());
        gpu::download(self_W2_g_, self->self_fc2_.W()); gpu::download(self_b2_g_, self->self_fc2_.b());
        gpu::download(enemy_W1_g_, self->enemy_fc1_.W()); gpu::download(enemy_b1_g_, self->enemy_fc1_.b());
        gpu::download(enemy_W2_g_, self->enemy_fc2_.W()); gpu::download(enemy_b2_g_, self->enemy_fc2_.b());
        gpu::download(ally_W1_g_, self->ally_fc1_.W()); gpu::download(ally_b1_g_, self->ally_fc1_.b());
        gpu::download(ally_W2_g_, self->ally_fc2_.W()); gpu::download(ally_b2_g_, self->ally_fc2_.b());
        gpu::cuda_sync();
    }
#endif
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_fc1_.save_to(out); enemy_fc2_.save_to(out);
    ally_fc1_.save_to(out);  ally_fc2_.save_to(out);
}

void DeepSetsDecoder::load_from(const uint8_t* data, size_t& offset, size_t size) {
    self_fc1_.load_from(data, offset, size);  self_fc2_.load_from(data, offset, size);
    enemy_fc1_.load_from(data, offset, size); enemy_fc2_.load_from(data, offset, size);
    ally_fc1_.load_from(data, offset, size);  ally_fc2_.load_from(data, offset, size);
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        gpu::upload(self_fc1_.W(), self_W1_g_); gpu::upload(self_fc1_.b(), self_b1_g_);
        gpu::upload(self_fc2_.W(), self_W2_g_); gpu::upload(self_fc2_.b(), self_b2_g_);
        gpu::upload(enemy_fc1_.W(), enemy_W1_g_); gpu::upload(enemy_fc1_.b(), enemy_b1_g_);
        gpu::upload(enemy_fc2_.W(), enemy_W2_g_); gpu::upload(enemy_fc2_.b(), enemy_b2_g_);
        gpu::upload(ally_fc1_.W(), ally_W1_g_); gpu::upload(ally_fc1_.b(), ally_b1_g_);
        gpu::upload(ally_fc2_.W(), ally_W2_g_); gpu::upload(ally_fc2_.b(), ally_b2_g_);
    }
#endif
}

#ifdef BGA_HAS_CUDA
void DeepSetsDecoder::to(Device d) {
    if (d == device_) return;
    device_require_cuda("DeepSetsDecoder");
    if (d == Device::GPU) {
        const int H = cfg_.hidden;
        const int E = cfg_.embed_dim;
        const int Sf = observation::SELF_FEATURES;
        const int Ef = observation::ENEMY_FEATURES;
        const int Af = observation::ALLY_FEATURES;

        gpu::upload(self_fc1_.W(), self_W1_g_); gpu::upload(self_fc1_.b(), self_b1_g_);
        gpu::upload(self_fc2_.W(), self_W2_g_); gpu::upload(self_fc2_.b(), self_b2_g_);
        gpu::upload(enemy_fc1_.W(), enemy_W1_g_); gpu::upload(enemy_fc1_.b(), enemy_b1_g_);
        gpu::upload(enemy_fc2_.W(), enemy_W2_g_); gpu::upload(enemy_fc2_.b(), enemy_b2_g_);
        gpu::upload(ally_fc1_.W(), ally_W1_g_); gpu::upload(ally_fc1_.b(), ally_b1_g_);
        gpu::upload(ally_fc2_.W(), ally_W2_g_); gpu::upload(ally_fc2_.b(), ally_b2_g_);

        self_dW1_g_.resize(H, E);  self_dW1_g_.zero(); self_db1_g_.resize(H, 1); self_db1_g_.zero();
        self_dW2_g_.resize(Sf, H); self_dW2_g_.zero(); self_db2_g_.resize(Sf, 1); self_db2_g_.zero();
        enemy_dW1_g_.resize(H, E);  enemy_dW1_g_.zero(); enemy_db1_g_.resize(H, 1); enemy_db1_g_.zero();
        enemy_dW2_g_.resize(Ef, H); enemy_dW2_g_.zero(); enemy_db2_g_.resize(Ef, 1); enemy_db2_g_.zero();
        ally_dW1_g_.resize(H, E);  ally_dW1_g_.zero(); ally_db1_g_.resize(H, 1); ally_db1_g_.zero();
        ally_dW2_g_.resize(Af, H); ally_dW2_g_.zero(); ally_db2_g_.resize(Af, 1); ally_db2_g_.zero();

        self_vW1_g_.resize(H, E);  self_vW1_g_.zero(); self_vb1_g_.resize(H, 1); self_vb1_g_.zero();
        self_vW2_g_.resize(Sf, H); self_vW2_g_.zero(); self_vb2_g_.resize(Sf, 1); self_vb2_g_.zero();
        enemy_vW1_g_.resize(H, E);  enemy_vW1_g_.zero(); enemy_vb1_g_.resize(H, 1); enemy_vb1_g_.zero();
        enemy_vW2_g_.resize(Ef, H); enemy_vW2_g_.zero(); enemy_vb2_g_.resize(Ef, 1); enemy_vb2_g_.zero();
        ally_vW1_g_.resize(H, E);  ally_vW1_g_.zero(); ally_vb1_g_.resize(H, 1); ally_vb1_g_.zero();
        ally_vW2_g_.resize(Af, H); ally_vW2_g_.zero(); ally_vb2_g_.resize(Af, 1); ally_vb2_g_.zero();

        x_g_cache_.resize(3 * E, 1);
        self_h_raw_g_.resize(H, 1);
        self_h_g_.resize(H, 1);
        e_h_raw_g_.resize(observation::K_ENEMIES, H);
        e_h_g_.resize(observation::K_ENEMIES, H);
        a_h_raw_g_.resize(observation::K_ALLIES, H);
        a_h_g_.resize(observation::K_ALLIES, H);

        device_ = Device::GPU;
    } else {
        gpu::download(self_W1_g_, self_fc1_.W()); gpu::download(self_b1_g_, self_fc1_.b());
        gpu::download(self_W2_g_, self_fc2_.W()); gpu::download(self_b2_g_, self_fc2_.b());
        gpu::download(enemy_W1_g_, enemy_fc1_.W()); gpu::download(enemy_b1_g_, enemy_fc1_.b());
        gpu::download(enemy_W2_g_, enemy_fc2_.W()); gpu::download(enemy_b2_g_, enemy_fc2_.b());
        gpu::download(ally_W1_g_, ally_fc1_.W()); gpu::download(ally_b1_g_, ally_fc1_.b());
        gpu::download(ally_W2_g_, ally_fc2_.W()); gpu::download(ally_b2_g_, ally_fc2_.b());
        gpu::cuda_sync();
        device_ = Device::CPU;
    }
}
#else
void DeepSetsDecoder::to(Device d) {
    if (d == device_) return;
    device_require_cuda("DeepSetsDecoder");
}
#endif

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

#ifdef BGA_HAS_CUDA

// GPU forward: split x → 3 embed views, run self/enemy/ally MLPs, write
// reconstruction directly into row-views of y.
void DeepSetsDecoder::forward(const gpu::GpuTensor& x, gpu::GpuTensor& y) {
    assert(device_ == Device::GPU);
    assert(x.size() == in_dim());
    if (y.rows != out_dim() || y.cols != 1) y.resize(out_dim(), 1);
    const int E = cfg_.embed_dim;
    const int H = cfg_.hidden;

    // Cache x for backward (we'll view the three sub-vectors out of it).
    x_g_cache_ = x.clone();

    gpu::GpuTensor self_in_view = gpu::GpuTensor::view(
        x_g_cache_.data + 0 * E, E, 1);
    gpu::GpuTensor pooled_e_view = gpu::GpuTensor::view(
        x_g_cache_.data + 1 * E, E, 1);
    gpu::GpuTensor pooled_a_view = gpu::GpuTensor::view(
        x_g_cache_.data + 2 * E, E, 1);

    // ── Self stream ──
    gpu::linear_forward_gpu(self_W1_g_, self_b1_g_, self_in_view, self_h_raw_g_);
    gpu::relu_forward_gpu(self_h_raw_g_, self_h_g_);
    gpu::GpuTensor y_self_view = gpu::GpuTensor::view(
        y.data, observation::SELF_FEATURES, 1);
    gpu::linear_forward_gpu(self_W2_g_, self_b2_g_, self_h_g_, y_self_view);

    // ── Enemy stream (broadcast pooled_e) ──
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            e_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            e_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::linear_forward_gpu(enemy_W1_g_, enemy_b1_g_, pooled_e_view, h_raw_row);
        gpu::relu_forward_gpu(h_raw_row, h_row);
        const int base = off_e + k * observation::ENEMY_FEATURES;
        gpu::GpuTensor y_slot_view = gpu::GpuTensor::view(
            y.data + base, observation::ENEMY_FEATURES, 1);
        gpu::linear_forward_gpu(enemy_W2_g_, enemy_b2_g_, h_row, y_slot_view);
    }

    // ── Ally stream (broadcast pooled_a) ──
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            a_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            a_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::linear_forward_gpu(ally_W1_g_, ally_b1_g_, pooled_a_view, h_raw_row);
        gpu::relu_forward_gpu(h_raw_row, h_row);
        const int base = off_a + k * observation::ALLY_FEATURES;
        gpu::GpuTensor y_slot_view = gpu::GpuTensor::view(
            y.data + base, observation::ALLY_FEATURES, 1);
        gpu::linear_forward_gpu(ally_W2_g_, ally_b2_g_, h_row, y_slot_view);
    }
}

void DeepSetsDecoder::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    assert(dY.size() == out_dim());
    if (dX.rows != in_dim() || dX.cols != 1) dX.resize(in_dim(), 1);
    dX.zero();
    const int E = cfg_.embed_dim;
    const int H = cfg_.hidden;

    // dX views over the three embed segments.
    gpu::GpuTensor dX_self_view  = gpu::GpuTensor::view(dX.data + 0 * E, E, 1);
    gpu::GpuTensor dX_pooled_e   = gpu::GpuTensor::view(dX.data + 1 * E, E, 1);
    gpu::GpuTensor dX_pooled_a   = gpu::GpuTensor::view(dX.data + 2 * E, E, 1);

    // ── Self stream ──
    gpu::GpuTensor dY_self_view = gpu::GpuTensor::view(
        const_cast<float*>(dY.data), observation::SELF_FEATURES, 1);
    gpu::GpuTensor dSelfH(H, 1);
    gpu::linear_backward_gpu(self_W2_g_, self_h_g_, dY_self_view,
                             dSelfH, self_dW2_g_, self_db2_g_);
    gpu::GpuTensor dSelfHraw(H, 1);
    gpu::relu_backward_gpu(self_h_raw_g_, dSelfH, dSelfHraw);
    // Cached self_in is the first E of x_g_cache_.
    gpu::GpuTensor self_in_view = gpu::GpuTensor::view(
        x_g_cache_.data + 0 * E, E, 1);
    // linear_backward overwrites dX — write into dX_self_view directly.
    gpu::linear_backward_gpu(self_W1_g_, self_in_view, dSelfHraw,
                             dX_self_view, self_dW1_g_, self_db1_g_);

    // ── Enemy stream — accumulate into dX_pooled_e ──
    gpu::GpuTensor pooled_e_view = gpu::GpuTensor::view(
        x_g_cache_.data + 1 * E, E, 1);
    gpu::GpuTensor dEnc_tmp(E, 1);  // per-slot scratch
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            e_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            e_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        const int base = off_e + k * observation::ENEMY_FEATURES;
        gpu::GpuTensor dY_slot_view = gpu::GpuTensor::view(
            const_cast<float*>(dY.data) + base, observation::ENEMY_FEATURES, 1);
        gpu::GpuTensor dHk(H, 1);
        gpu::linear_backward_gpu(enemy_W2_g_, h_row, dY_slot_view,
                                 dHk, enemy_dW2_g_, enemy_db2_g_);
        gpu::GpuTensor dHk_raw(H, 1);
        gpu::relu_backward_gpu(h_raw_row, dHk, dHk_raw);
        gpu::linear_backward_gpu(enemy_W1_g_, pooled_e_view, dHk_raw,
                                 dEnc_tmp, enemy_dW1_g_, enemy_db1_g_);
        gpu::add_inplace_gpu(dX_pooled_e, dEnc_tmp);
    }

    // ── Ally stream — accumulate into dX_pooled_a ──
    gpu::GpuTensor pooled_a_view = gpu::GpuTensor::view(
        x_g_cache_.data + 2 * E, E, 1);
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            a_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            a_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        const int base = off_a + k * observation::ALLY_FEATURES;
        gpu::GpuTensor dY_slot_view = gpu::GpuTensor::view(
            const_cast<float*>(dY.data) + base, observation::ALLY_FEATURES, 1);
        gpu::GpuTensor dHk(H, 1);
        gpu::linear_backward_gpu(ally_W2_g_, h_row, dY_slot_view,
                                 dHk, ally_dW2_g_, ally_db2_g_);
        gpu::GpuTensor dHk_raw(H, 1);
        gpu::relu_backward_gpu(h_raw_row, dHk, dHk_raw);
        gpu::linear_backward_gpu(ally_W1_g_, pooled_a_view, dHk_raw,
                                 dEnc_tmp, ally_dW1_g_, ally_db1_g_);
        gpu::add_inplace_gpu(dX_pooled_a, dEnc_tmp);
    }
}
#endif

} // namespace brogameagent::nn
