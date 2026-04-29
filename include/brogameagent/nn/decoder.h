#pragma once

#include "circuits.h"
#include "device.h"
#include "tensor.h"
#include "brogameagent/observation.h"

#ifdef BGA_HAS_CUDA
#include "gpu/tensor.h"
#endif

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── DeepSetsDecoder ───────────────────────────────────────────────────────
//
// Mirror of DeepSetsEncoder for unsupervised autoencoding pretraining.
//
// Input: embedding of size 3 * embed_dim — [self_z, pooled_e, pooled_a].
// Output: reconstructed observation vector of size observation::TOTAL.
//
// Architecture:
//   - Self stream:  embed_dim -> hidden -> SELF_FEATURES  via Linear+ReLU+Linear.
//   - Enemy stream: the pooled enemy embedding is broadcast to each of the
//       K_ENEMIES slots (same pooled vector per slot), and each slot goes
//       through a shared embed_dim -> hidden -> ENEMY_FEATURES MLP. Because
//       we only have a single pooled vector to reconstruct K_ENEMIES slots
//       from, every slot's reconstruction is identical. This is an expected
//       property of pooling-based autoencoders — the pool is a lossy summary
//       and cannot recover per-slot identity. The reconstruction still
//       carries useful signal (mean per-feature values, valid-flag rate,
//       mean distance/hp, etc.) which is enough for pretraining.
//   - Ally stream: same pattern for K_ALLIES with ALLY_FEATURES.
//
// No bottleneck beyond what the encoder already produces. Hand-authored
// forward/backward — see decoder.cpp.
//
// GPU dispatch mirrors DeepSetsEncoder: device_ field, GpuTensor mirrors,
// to(Device) migration, GPU forward/backward overloads using linear_*_gpu /
// relu_*_gpu / split_rows_gpu.

class DeepSetsDecoder : public ICircuit {
public:
    DeepSetsDecoder() = default;

    struct Config {
        int embed_dim = 32;   // per-stream input embed width (matches encoder)
        int hidden    = 32;   // per-stream hidden width
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int in_dim() const { return 3 * cfg_.embed_dim; }
    int out_dim() const { return observation::TOTAL; }

    // x: size 3*embed_dim. y: size observation::TOTAL.
    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

#ifdef BGA_HAS_CUDA
    void forward(const gpu::GpuTensor& x, gpu::GpuTensor& y);
    void backward(const gpu::GpuTensor& dY, gpu::GpuTensor& dX);
#endif

    Device device() const { return device_; }
    void to(Device d);

    const char* name() const override { return "DeepSetsDecoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    const Config& config() const { return cfg_; }

private:
    Config cfg_{};

    // Self stream.
    Linear self_fc1_, self_fc2_;
    Tensor self_h_raw_, self_h_;   // post-fc1, post-relu

    // Enemy stream (shared across slots; per-slot caches).
    Linear enemy_fc1_, enemy_fc2_;
    std::vector<Tensor> e_h_raw_, e_h_;  // per-slot hidden caches

    // Ally stream.
    Linear ally_fc1_, ally_fc2_;
    std::vector<Tensor> a_h_raw_, a_h_;

    // Cached split of input for backward.
    Tensor self_in_, pooled_e_, pooled_a_;

    Device device_ = Device::CPU;
#ifdef BGA_HAS_CUDA
    gpu::GpuTensor self_W1_g_, self_b1_g_, self_W2_g_, self_b2_g_;
    gpu::GpuTensor self_dW1_g_, self_db1_g_, self_dW2_g_, self_db2_g_;
    gpu::GpuTensor self_vW1_g_, self_vb1_g_, self_vW2_g_, self_vb2_g_;

    gpu::GpuTensor enemy_W1_g_, enemy_b1_g_, enemy_W2_g_, enemy_b2_g_;
    gpu::GpuTensor enemy_dW1_g_, enemy_db1_g_, enemy_dW2_g_, enemy_db2_g_;
    gpu::GpuTensor enemy_vW1_g_, enemy_vb1_g_, enemy_vW2_g_, enemy_vb2_g_;

    gpu::GpuTensor ally_W1_g_, ally_b1_g_, ally_W2_g_, ally_b2_g_;
    gpu::GpuTensor ally_dW1_g_, ally_db1_g_, ally_dW2_g_, ally_db2_g_;
    gpu::GpuTensor ally_vW1_g_, ally_vb1_g_, ally_vW2_g_, ally_vb2_g_;

    // Cached input split (views into x_g_cache_).
    gpu::GpuTensor x_g_cache_;        // (3*embed_dim, 1) — clone of forward x

    // Self stream caches.
    gpu::GpuTensor self_h_raw_g_;     // (hidden, 1)
    gpu::GpuTensor self_h_g_;         // (hidden, 1)

    // Per-slot caches (broadcast input is the same pooled vec for every slot).
    gpu::GpuTensor e_h_raw_g_;        // (K_ENEMIES, hidden)
    gpu::GpuTensor e_h_g_;            // (K_ENEMIES, hidden)

    gpu::GpuTensor a_h_raw_g_;        // (K_ALLIES, hidden)
    gpu::GpuTensor a_h_g_;            // (K_ALLIES, hidden)
#endif
};

} // namespace brogameagent::nn
