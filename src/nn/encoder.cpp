#include "brogameagent/nn/encoder.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

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

void DeepSetsEncoder::sgd_step(float lr, float momentum) {
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

void DeepSetsEncoder::adam_step(float lr, float b1, float b2, float eps, int step) {
    // Adam path operates on the host Linear members. Layers using the GPU
    // dispatch with separate mirror tensors don't yet have Adam mirrors;
    // callers training on GPU should fall back to sgd_step. This delegates
    // to Linear::adam_step which uses the host m/v buffers.
    self_fc1_.adam_step(lr, b1, b2, eps, step); self_fc2_.adam_step(lr, b1, b2, eps, step);
    enemy_fc1_.adam_step(lr, b1, b2, eps, step); enemy_fc2_.adam_step(lr, b1, b2, eps, step);
    ally_fc1_.adam_step(lr, b1, b2, eps, step);  ally_fc2_.adam_step(lr, b1, b2, eps, step);
}

#ifdef BGA_HAS_CUDA
// Helper: sync a Linear's W/b from GPU mirror back into the host Linear so
// save_to / Linear::W() observers see fresh values.
static void sync_linear_to_host(Linear& L,
                                const gpu::GpuTensor& W_g,
                                const gpu::GpuTensor& b_g) {
    gpu::download(W_g, L.W());
    gpu::download(b_g, L.b());
}
#endif

void DeepSetsEncoder::save_to(std::vector<uint8_t>& out) const {
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        auto* self = const_cast<DeepSetsEncoder*>(this);
        sync_linear_to_host(self->self_fc1_, self_W1_g_, self_b1_g_);
        sync_linear_to_host(self->self_fc2_, self_W2_g_, self_b2_g_);
        sync_linear_to_host(self->enemy_fc1_, enemy_W1_g_, enemy_b1_g_);
        sync_linear_to_host(self->enemy_fc2_, enemy_W2_g_, enemy_b2_g_);
        sync_linear_to_host(self->ally_fc1_, ally_W1_g_, ally_b1_g_);
        sync_linear_to_host(self->ally_fc2_, ally_W2_g_, ally_b2_g_);
        gpu::cuda_sync();
    }
#endif
    self_fc1_.save_to(out); self_fc2_.save_to(out);
    enemy_fc1_.save_to(out); enemy_fc2_.save_to(out);
    ally_fc1_.save_to(out);  ally_fc2_.save_to(out);
}

void DeepSetsEncoder::load_from(const uint8_t* data, size_t& offset, size_t size) {
    self_fc1_.load_from(data, offset, size);  self_fc2_.load_from(data, offset, size);
    enemy_fc1_.load_from(data, offset, size); enemy_fc2_.load_from(data, offset, size);
    ally_fc1_.load_from(data, offset, size);  ally_fc2_.load_from(data, offset, size);
#ifdef BGA_HAS_CUDA
    if (device_ == Device::GPU) {
        // Re-upload weights so GPU mirror matches loaded host values.
        gpu::upload(self_fc1_.W(), self_W1_g_);
        gpu::upload(self_fc1_.b(), self_b1_g_);
        gpu::upload(self_fc2_.W(), self_W2_g_);
        gpu::upload(self_fc2_.b(), self_b2_g_);
        gpu::upload(enemy_fc1_.W(), enemy_W1_g_);
        gpu::upload(enemy_fc1_.b(), enemy_b1_g_);
        gpu::upload(enemy_fc2_.W(), enemy_W2_g_);
        gpu::upload(enemy_fc2_.b(), enemy_b2_g_);
        gpu::upload(ally_fc1_.W(), ally_W1_g_);
        gpu::upload(ally_fc1_.b(), ally_b1_g_);
        gpu::upload(ally_fc2_.W(), ally_W2_g_);
        gpu::upload(ally_fc2_.b(), ally_b2_g_);
    }
#endif
}

#ifdef BGA_HAS_CUDA
void DeepSetsEncoder::to(Device d) {
    if (d == device_) return;
    device_require_cuda("DeepSetsEncoder");
    if (d == Device::GPU) {
        const int H = cfg_.hidden;
        const int E = cfg_.embed_dim;
        // Upload weights/biases.
        gpu::upload(self_fc1_.W(), self_W1_g_);
        gpu::upload(self_fc1_.b(), self_b1_g_);
        gpu::upload(self_fc2_.W(), self_W2_g_);
        gpu::upload(self_fc2_.b(), self_b2_g_);
        gpu::upload(enemy_fc1_.W(), enemy_W1_g_);
        gpu::upload(enemy_fc1_.b(), enemy_b1_g_);
        gpu::upload(enemy_fc2_.W(), enemy_W2_g_);
        gpu::upload(enemy_fc2_.b(), enemy_b2_g_);
        gpu::upload(ally_fc1_.W(), ally_W1_g_);
        gpu::upload(ally_fc1_.b(), ally_b1_g_);
        gpu::upload(ally_fc2_.W(), ally_W2_g_);
        gpu::upload(ally_fc2_.b(), ally_b2_g_);
        // Allocate grad mirrors (zeroed).
        self_dW1_g_.resize(H, observation::SELF_FEATURES); self_dW1_g_.zero();
        self_db1_g_.resize(H, 1);                          self_db1_g_.zero();
        self_dW2_g_.resize(E, H);                          self_dW2_g_.zero();
        self_db2_g_.resize(E, 1);                          self_db2_g_.zero();
        enemy_dW1_g_.resize(H, observation::ENEMY_FEATURES); enemy_dW1_g_.zero();
        enemy_db1_g_.resize(H, 1);                            enemy_db1_g_.zero();
        enemy_dW2_g_.resize(E, H);                            enemy_dW2_g_.zero();
        enemy_db2_g_.resize(E, 1);                            enemy_db2_g_.zero();
        ally_dW1_g_.resize(H, observation::ALLY_FEATURES); ally_dW1_g_.zero();
        ally_db1_g_.resize(H, 1);                           ally_db1_g_.zero();
        ally_dW2_g_.resize(E, H);                           ally_dW2_g_.zero();
        ally_db2_g_.resize(E, 1);                           ally_db2_g_.zero();
        // Velocities (zeroed).
        self_vW1_g_.resize(H, observation::SELF_FEATURES); self_vW1_g_.zero();
        self_vb1_g_.resize(H, 1);                          self_vb1_g_.zero();
        self_vW2_g_.resize(E, H);                          self_vW2_g_.zero();
        self_vb2_g_.resize(E, 1);                          self_vb2_g_.zero();
        enemy_vW1_g_.resize(H, observation::ENEMY_FEATURES); enemy_vW1_g_.zero();
        enemy_vb1_g_.resize(H, 1);                            enemy_vb1_g_.zero();
        enemy_vW2_g_.resize(E, H);                            enemy_vW2_g_.zero();
        enemy_vb2_g_.resize(E, 1);                            enemy_vb2_g_.zero();
        ally_vW1_g_.resize(H, observation::ALLY_FEATURES); ally_vW1_g_.zero();
        ally_vb1_g_.resize(H, 1);                           ally_vb1_g_.zero();
        ally_vW2_g_.resize(E, H);                           ally_vW2_g_.zero();
        ally_vb2_g_.resize(E, 1);                           ally_vb2_g_.zero();
        // Activation/cache mirrors.
        x_g_cache_.resize(observation::TOTAL, 1);
        self_h_raw_g_.resize(H, 1);
        self_h_g_.resize(H, 1);
        self_z_g_.resize(E, 1);
        e_h_raw_g_.resize(observation::K_ENEMIES, H);
        e_h_g_.resize(observation::K_ENEMIES, H);
        e_z_g_.resize(observation::K_ENEMIES, E);
        a_h_raw_g_.resize(observation::K_ALLIES, H);
        a_h_g_.resize(observation::K_ALLIES, H);
        a_z_g_.resize(observation::K_ALLIES, E);
        pooled_e_g_.resize(E, 1);
        pooled_a_g_.resize(E, 1);
        e_mask_g_.resize(observation::K_ENEMIES, 1);
        a_mask_g_.resize(observation::K_ALLIES, 1);
        device_ = Device::GPU;
    } else {
        // Download W/b back to host Linears; keep grads/velocities GPU-only.
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
void DeepSetsEncoder::to(Device d) {
    if (d == device_) return;
    device_require_cuda("DeepSetsEncoder");
}
#endif

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

#ifdef BGA_HAS_CUDA

// GPU forward: per-slot Linear (linear_*_gpu), masked mean-pool over slots,
// concat[self_z | pooled_e | pooled_a] via concat_rows_gpu.
void DeepSetsEncoder::forward(const gpu::GpuTensor& x, gpu::GpuTensor& y) {
    assert(device_ == Device::GPU);
    assert(x.size() == observation::TOTAL);
    if (y.rows != out_dim() || y.cols != 1) y.resize(out_dim(), 1);
    const int H = cfg_.hidden;
    const int E = cfg_.embed_dim;

    // Cache x (clone) for backward — we'll view slot rows out of it.
    x_g_cache_ = x.clone();

    // Mirror the input on host to read slot-validity flags. This is a tiny
    // copy (~64 floats) and lets us avoid an asynchronous device-side mask
    // construction kernel.
    gpu::download(x, x_cache_);
    gpu::cuda_sync();

    // ── Self stream ────────────────────────────────────────────────────────
    // Use a non-owning view over the first SELF_FEATURES of the cached x for
    // both forward and backward (linear_backward needs the same buffer).
    gpu::GpuTensor self_in_view = gpu::GpuTensor::view(
        x_g_cache_.data, observation::SELF_FEATURES, 1);
    gpu::linear_forward_gpu(self_W1_g_, self_b1_g_, self_in_view, self_h_raw_g_);
    gpu::relu_forward_gpu(self_h_raw_g_, self_h_g_);
    gpu::linear_forward_gpu(self_W2_g_, self_b2_g_, self_h_g_, self_z_g_);

    // ── Enemy stream ───────────────────────────────────────────────────────
    const int off_e = observation::SELF_FEATURES;
    e_n_valid_ = 0;
    Tensor e_mask_h(observation::K_ENEMIES, 1);
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        const bool valid = x_cache_[base] > 0.5f;
        e_valid_[k] = valid ? 1 : 0;
        e_mask_h[k] = valid ? 1.0f : 0.0f;
        if (valid) ++e_n_valid_;
    }
    gpu::upload(e_mask_h, e_mask_g_);
    // Per-slot Linear forward into rows of e_h_raw_g_ / e_h_g_ / e_z_g_.
    // Slot input view points into x_g_cache_ — backward reuses it.
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        gpu::GpuTensor x_slot_view = gpu::GpuTensor::view(
            x_g_cache_.data + base, observation::ENEMY_FEATURES, 1);
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            e_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            e_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor z_row = gpu::GpuTensor::view(
            e_z_g_.data + static_cast<size_t>(k) * E, E, 1);
        // Always run forward (kernels are cheap; mask zeros pool contribution
        // for invalid slots). Only valid-slot caches are consumed in backward.
        gpu::linear_forward_gpu(enemy_W1_g_, enemy_b1_g_, x_slot_view, h_raw_row);
        gpu::relu_forward_gpu(h_raw_row, h_row);
        gpu::linear_forward_gpu(enemy_W2_g_, enemy_b2_g_, h_row, z_row);
    }
    // Pool: pooled_e = masked_mean_pool(e_z_g_, e_mask_g_).
    gpu::masked_mean_pool_forward_gpu(e_z_g_, e_mask_g_.data, pooled_e_g_);

    // ── Ally stream ────────────────────────────────────────────────────────
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    a_n_valid_ = 0;
    Tensor a_mask_h(observation::K_ALLIES, 1);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        const bool valid = x_cache_[base] > 0.5f;
        a_valid_[k] = valid ? 1 : 0;
        a_mask_h[k] = valid ? 1.0f : 0.0f;
        if (valid) ++a_n_valid_;
    }
    gpu::upload(a_mask_h, a_mask_g_);
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        gpu::GpuTensor x_slot_view = gpu::GpuTensor::view(
            x_g_cache_.data + base, observation::ALLY_FEATURES, 1);
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            a_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor h_row = gpu::GpuTensor::view(
            a_h_g_.data + static_cast<size_t>(k) * H, H, 1);
        gpu::GpuTensor z_row = gpu::GpuTensor::view(
            a_z_g_.data + static_cast<size_t>(k) * E, E, 1);
        gpu::linear_forward_gpu(ally_W1_g_, ally_b1_g_, x_slot_view, h_raw_row);
        gpu::relu_forward_gpu(h_raw_row, h_row);
        gpu::linear_forward_gpu(ally_W2_g_, ally_b2_g_, h_row, z_row);
    }
    gpu::masked_mean_pool_forward_gpu(a_z_g_, a_mask_g_.data, pooled_a_g_);

    // ── Concat into y = [self_z | pooled_e | pooled_a] ─────────────────────
    std::vector<const gpu::GpuTensor*> parts = {
        &self_z_g_, &pooled_e_g_, &pooled_a_g_
    };
    gpu::concat_rows_gpu(parts, y);
}

void DeepSetsEncoder::backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX) {
    assert(device_ == Device::GPU);
    assert(dY.size() == out_dim());
    if (dX.rows != observation::TOTAL || dX.cols != 1) dX.resize(observation::TOTAL, 1);
    dX.zero();
    const int H = cfg_.hidden;
    const int E = cfg_.embed_dim;

    // Split dY into [dSelfZ | dPooled_e | dPooled_a].
    gpu::GpuTensor dSelfZ_view = gpu::GpuTensor::view(
        const_cast<float*>(dY.data) + 0 * E, E, 1);
    gpu::GpuTensor dPooledE_view = gpu::GpuTensor::view(
        const_cast<float*>(dY.data) + 1 * E, E, 1);
    gpu::GpuTensor dPooledA_view = gpu::GpuTensor::view(
        const_cast<float*>(dY.data) + 2 * E, E, 1);

    // ── Self backward ──────────────────────────────────────────────────────
    gpu::GpuTensor dSelfH(H, 1);
    gpu::linear_backward_gpu(self_W2_g_, self_h_g_, dSelfZ_view,
                             dSelfH, self_dW2_g_, self_db2_g_);
    gpu::GpuTensor dSelfHraw(H, 1);
    gpu::relu_backward_gpu(self_h_raw_g_, dSelfH, dSelfHraw);
    // Reuse self_in_view as input for backward, and write dX[0..SELF) directly.
    gpu::GpuTensor self_in_view = gpu::GpuTensor::view(
        x_g_cache_.data, observation::SELF_FEATURES, 1);
    gpu::GpuTensor dX_self_view = gpu::GpuTensor::view(
        dX.data, observation::SELF_FEATURES, 1);
    gpu::linear_backward_gpu(self_W1_g_, self_in_view, dSelfHraw,
                             dX_self_view, self_dW1_g_, self_db1_g_);

    // ── Enemy backward ─────────────────────────────────────────────────────
    // dE_z (K, E) from masked_mean_pool_backward — overwritten output (per spec).
    gpu::GpuTensor dE_z(observation::K_ENEMIES, E);
    gpu::masked_mean_pool_backward_gpu(dPooledE_view, e_mask_g_.data,
                                       observation::K_ENEMIES, dE_z);
    const int off_e = observation::SELF_FEATURES;
    // Match CPU semantics: CPU's Linear caches only the LAST forwarded input
    // (x_cache_ is overwritten each iteration). The per-slot fc1/fc2 weight
    // gradients on CPU therefore accumulate using the LAST valid slot's
    // cached input across every backward call. We replicate this exactly so
    // post-sgd parameters match — see the CPU `Linear::forward(x){ x_cache_=x; }`
    // + per-slot loop in `DeepSetsEncoder::forward`.
    int last_e_valid = -1;
    for (int k = 0; k < observation::K_ENEMIES; ++k) if (e_valid_[k]) last_e_valid = k;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        if (!e_valid_[k]) continue;
        gpu::GpuTensor dZk_view = gpu::GpuTensor::view(
            dE_z.data + static_cast<size_t>(k) * E, E, 1);
        // Per-slot relu mask (post-relu cache) is correct in CPU; use slot k.
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            e_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        // CPU caches LAST valid slot's e_h_ for fc2 backward — match.
        gpu::GpuTensor h_row_last = gpu::GpuTensor::view(
            e_h_g_.data + static_cast<size_t>(last_e_valid) * H, H, 1);
        gpu::GpuTensor dHk(H, 1);
        gpu::linear_backward_gpu(enemy_W2_g_, h_row_last, dZk_view,
                                 dHk, enemy_dW2_g_, enemy_db2_g_);
        gpu::GpuTensor dHk_raw(H, 1);
        gpu::relu_backward_gpu(h_raw_row, dHk, dHk_raw);
        // CPU caches LAST valid slot's slot_in for fc1 backward — match.
        const int base_last = off_e + last_e_valid * observation::ENEMY_FEATURES;
        gpu::GpuTensor x_slot_last_view = gpu::GpuTensor::view(
            x_g_cache_.data + base_last, observation::ENEMY_FEATURES, 1);
        // Per-slot dX is correct (computed via W^T @ dHk, no cache dependence).
        const int base = off_e + k * observation::ENEMY_FEATURES;
        gpu::GpuTensor dX_slot_view = gpu::GpuTensor::view(
            dX.data + base, observation::ENEMY_FEATURES, 1);
        gpu::linear_backward_gpu(enemy_W1_g_, x_slot_last_view, dHk_raw,
                                 dX_slot_view, enemy_dW1_g_, enemy_db1_g_);
    }

    // ── Ally backward ──────────────────────────────────────────────────────
    gpu::GpuTensor dA_z(observation::K_ALLIES, E);
    gpu::masked_mean_pool_backward_gpu(dPooledA_view, a_mask_g_.data,
                                       observation::K_ALLIES, dA_z);
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    int last_a_valid = -1;
    for (int k = 0; k < observation::K_ALLIES; ++k) if (a_valid_[k]) last_a_valid = k;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        if (!a_valid_[k]) continue;
        gpu::GpuTensor dZk_view = gpu::GpuTensor::view(
            dA_z.data + static_cast<size_t>(k) * E, E, 1);
        gpu::GpuTensor h_raw_row = gpu::GpuTensor::view(
            a_h_raw_g_.data + static_cast<size_t>(k) * H, H, 1);
        // Match CPU's "last x_cache" fc2 input.
        gpu::GpuTensor h_row_last = gpu::GpuTensor::view(
            a_h_g_.data + static_cast<size_t>(last_a_valid) * H, H, 1);
        gpu::GpuTensor dHk(H, 1);
        gpu::linear_backward_gpu(ally_W2_g_, h_row_last, dZk_view,
                                 dHk, ally_dW2_g_, ally_db2_g_);
        gpu::GpuTensor dHk_raw(H, 1);
        gpu::relu_backward_gpu(h_raw_row, dHk, dHk_raw);
        // Match CPU's "last x_cache" fc1 input.
        const int base_last = off_a + last_a_valid * observation::ALLY_FEATURES;
        gpu::GpuTensor x_slot_last_view = gpu::GpuTensor::view(
            x_g_cache_.data + base_last, observation::ALLY_FEATURES, 1);
        const int base = off_a + k * observation::ALLY_FEATURES;
        gpu::GpuTensor dX_slot_view = gpu::GpuTensor::view(
            dX.data + base, observation::ALLY_FEATURES, 1);
        gpu::linear_backward_gpu(ally_W1_g_, x_slot_last_view, dHk_raw,
                                 dX_slot_view, ally_dW1_g_, ally_db1_g_);
    }
}
#endif

} // namespace brogameagent::nn
