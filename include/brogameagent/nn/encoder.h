#pragma once

#include "circuits.h"
#include "device.h"
#include "tensor.h"
#include "brogameagent/observation.h"

#ifdef BGA_HAS_GPU
#include "gpu/tensor.h"
#endif

#include <cstdint>

namespace brogameagent::nn {

// ─── DeepSetsEncoder ───────────────────────────────────────────────────────
//
// Egocentric set encoder tailored to observation::build.
//
// Input: a TOTAL-dim vector in the layout that observation::build emits —
//   self[SELF_FEATURES], enemies[K_ENEMIES*ENEMY_FEATURES], allies[K_ALLIES*ALLY_FEATURES].
//
// Processing:
//   - Self block goes through its own small MLP: phi_self.
//   - Each enemy slot (K_ENEMIES of them) goes through a shared phi_enemy
//     MLP, output mean-pooled across valid slots. A slot is "valid" iff its
//     first feature (the valid flag) is > 0.5; invalid slots are masked out
//     of the pool (do not contribute, divide by count of valid only).
//   - Same for allies via phi_ally.
//   - Outputs concatenated into a single embedding vector of dim out_dim().
//
// Permutation invariance is structural: the pool is order-invariant and the
// slots are already sorted nearest-first by observation::build, so slot
// identity is stable. Backward is hand-authored to match — each slot's dE
// is distributed from the pool, scaled by 1/n_valid, and invalid slots
// receive zero gradient.
//
// GPU dispatch: device_ tracks where parameters live. `to(Device)` migrates
// host↔device. CPU forward/backward are unchanged. GPU overloads call
// linear_*_gpu / relu_*_gpu / masked_mean_pool_*_gpu / concat_rows_gpu.

class DeepSetsEncoder : public ICircuit {
public:
    DeepSetsEncoder() = default;

    struct Config {
        int hidden    = 32;   // per-stream hidden width (self/enemy/ally MLPs)
        int embed_dim = 32;   // per-stream output embedding width
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int out_dim() const { return 3 * cfg_.embed_dim; }

    // x: size observation::TOTAL. y: size out_dim().
    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

#ifdef BGA_HAS_GPU
    void forward(const gpu::GpuTensor& x, gpu::GpuTensor& y);
    void backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX);
#endif

    Device device() const { return device_; }
    void to(Device d);

    const char* name() const override { return "DeepSetsEncoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    const Config& config() const { return cfg_; }

private:
    Config cfg_{};

    // Self stream: SELF_FEATURES -> hidden -> embed
    Linear self_fc1_, self_fc2_;
    Relu   self_act_;
    Tensor self_h_, self_z_;     // hidden after fc1+relu, embed after fc2

    // Enemy stream: ENEMY_FEATURES -> hidden -> embed (applied per slot)
    Linear enemy_fc1_, enemy_fc2_;
    // Ally stream
    Linear ally_fc1_, ally_fc2_;

    // Per-slot caches. Sized by init().
    std::vector<Tensor> e_h_, e_z_;
    std::vector<Tensor> a_h_, a_z_;
    std::vector<uint8_t> e_valid_, a_valid_;
    int e_n_valid_ = 0, a_n_valid_ = 0;

    Tensor x_cache_;  // copy of input for backward shape

    // scratch for per-slot backward (input grad for an entire slot row)
    Tensor slot_grad_in_;

    Device device_ = Device::CPU;
#ifdef BGA_HAS_GPU
    // ── GPU mirrors ────────────────────────────────────────────────────────
    // Per-Linear weights/grads/velocities.
    gpu::GpuTensor self_W1_g_, self_b1_g_, self_W2_g_, self_b2_g_;
    gpu::GpuTensor self_dW1_g_, self_db1_g_, self_dW2_g_, self_db2_g_;
    gpu::GpuTensor self_vW1_g_, self_vb1_g_, self_vW2_g_, self_vb2_g_;

    gpu::GpuTensor enemy_W1_g_, enemy_b1_g_, enemy_W2_g_, enemy_b2_g_;
    gpu::GpuTensor enemy_dW1_g_, enemy_db1_g_, enemy_dW2_g_, enemy_db2_g_;
    gpu::GpuTensor enemy_vW1_g_, enemy_vb1_g_, enemy_vW2_g_, enemy_vb2_g_;

    gpu::GpuTensor ally_W1_g_, ally_b1_g_, ally_W2_g_, ally_b2_g_;
    gpu::GpuTensor ally_dW1_g_, ally_db1_g_, ally_dW2_g_, ally_db2_g_;
    gpu::GpuTensor ally_vW1_g_, ally_vb1_g_, ally_vW2_g_, ally_vb2_g_;

    // Forward caches on GPU.
    gpu::GpuTensor x_g_cache_;        // (TOTAL, 1) — clone of forward x
    gpu::GpuTensor self_h_raw_g_;     // (hidden, 1) — pre-relu
    gpu::GpuTensor self_h_g_;         // (hidden, 1) — post-relu
    gpu::GpuTensor self_z_g_;         // (embed_dim, 1)

    // Per-slot caches: each stored as a (K, hidden) and (K, embed_dim) matrix
    // packed by row (slot k occupies row k). Per-slot Linear forward writes
    // into row-views of these matrices.
    gpu::GpuTensor e_h_raw_g_;        // (K_ENEMIES, hidden)
    gpu::GpuTensor e_h_g_;            // (K_ENEMIES, hidden)
    gpu::GpuTensor e_z_g_;            // (K_ENEMIES, embed_dim)

    gpu::GpuTensor a_h_raw_g_;        // (K_ALLIES, hidden)
    gpu::GpuTensor a_h_g_;            // (K_ALLIES, hidden)
    gpu::GpuTensor a_z_g_;            // (K_ALLIES, embed_dim)

    // Pool inputs / outputs.
    gpu::GpuTensor pooled_e_g_;       // (embed_dim, 1)
    gpu::GpuTensor pooled_a_g_;       // (embed_dim, 1)

    // Mask buffers on device (one float per slot).
    gpu::GpuTensor e_mask_g_;         // (K_ENEMIES, 1)
    gpu::GpuTensor a_mask_g_;         // (K_ALLIES, 1)
#endif
};

} // namespace brogameagent::nn
