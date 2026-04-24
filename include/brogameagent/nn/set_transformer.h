#pragma once

#include "attention.h"
#include "circuits.h"
#include "layernorm.h"
#include "tensor.h"
#include "brogameagent/observation.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── SetTransformerEncoder ────────────────────────────────────────────────
//
// Drop-in alternative to DeepSetsEncoder. Same input (observation::TOTAL)
// and same output dim (3 * embed_dim). Per-slot Linear+ReLU projection,
// a single self-attention block over the slots, post-norm LayerNorm per
// row (post-norm chosen for simpler caching — LayerNorm sees the
// attention output directly), then masked mean-pool over valid slots.
// Self stream is identical to DeepSetsEncoder (two Linear+ReLU).

class SetTransformerEncoder : public ICircuit {
public:
    SetTransformerEncoder() = default;

    struct Config {
        int hidden    = 32;
        int embed_dim = 32;
    };

    void init(const Config& cfg, uint64_t& rng_state);

    int out_dim() const { return 3 * cfg_.embed_dim; }

    void forward(const Tensor& x, Tensor& y);
    void backward(const Tensor& dY, Tensor& dX);

    const char* name() const override { return "SetTransformerEncoder"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

private:
    Config cfg_{};

    // Self stream (identical to DeepSetsEncoder's self path).
    Linear self_fc1_, self_fc2_;
    Tensor self_h_, self_z_;   // post-relu hidden, post-fc2 embed

    // Enemy stream.
    Linear enemy_proj_;                        // ENEMY_FEATURES -> embed_dim
    ScaledDotProductAttention enemy_attn_;     // (K_ENEMIES, embed_dim)
    std::vector<LayerNorm> enemy_ln_;          // one per slot (shared semantics, separate params)
    // Caches.
    Tensor enemy_proj_raw_;   // (K, D) pre-relu
    Tensor enemy_proj_act_;   // (K, D) post-relu (attn input)
    Tensor enemy_attn_out_;   // (K, D) attn output
    Tensor enemy_ln_out_;     // (K, D) after per-row LN
    std::vector<uint8_t> e_valid_;
    int e_n_valid_ = 0;

    // Ally stream.
    Linear ally_proj_;
    ScaledDotProductAttention ally_attn_;
    std::vector<LayerNorm> ally_ln_;
    Tensor ally_proj_raw_;
    Tensor ally_proj_act_;
    Tensor ally_attn_out_;
    Tensor ally_ln_out_;
    std::vector<uint8_t> a_valid_;
    int a_n_valid_ = 0;

    Tensor x_cache_;
};

} // namespace brogameagent::nn
