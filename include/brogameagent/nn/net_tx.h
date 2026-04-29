#pragma once

#include "circuits.h"
#include "device.h"
#include "heads.h"
#include "layernorm.h"
#include "tensor.h"
#include "transformer_block.h"
#include "transformer_encoder.h"
#include "brogameagent/observation.h"

#ifdef BGA_HAS_CUDA
#include "gpu/tensor.h"
#endif

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── SingleHeroNetTX ──────────────────────────────────────────────────────
//
// Per-hero net with a multi-block, multi-head Transformer per entity stream.
//
// Pipeline (single sample):
//   self_vec(SELF_FEATURES) ── Linear(SELF→hidden) ── ReLU ── Linear(hidden→d_model)
//                                                                         │
//   enemy_slots(K_ENEMIES, ENEMY_FEATURES)                                 │
//        ── per-slot Linear(ENEMY→d_model) (zero on invalid slots)         │
//        ── TransformerEncoder<n_blocks, num_heads, d_ff>                  │
//        ── masked mean-pool (D)                                           │
//                                                                         │
//   ally_slots(K_ALLIES, ALLY_FEATURES)                                    │
//        ── per-slot Linear(ALLY→d_model) (zero on invalid)                │
//        ── TransformerEncoder (separate weights)                          │
//        ── masked mean-pool (D)                                           │
//                                                                         │
//   concat[self, pooled_e, pooled_a]  (3*d_model)                          │
//        ── Linear(3D → trunk_hidden) ── ReLU                              │
//        ── { ValueHead, FactoredPolicyHead }
//
// Implementation note: per-slot projections, self MLP, trunk Linear, and the
// policy/value heads run on CPU (the Linear circuit class is CPU-only). The
// two TransformerEncoder stacks dispatch to GPU when `to(GPU)` is called —
// they are the dominant compute. Inputs to the GPU encoders are uploaded per
// forward call and outputs are downloaded for the masked mean-pool / concat
// path. This is the simplest design that exercises real multi-block GPU
// transformer compute without retrofitting the Linear circuit.

class SingleHeroNetTX : public ICircuit {
public:
    struct Config {
        int self_hidden = 32;
        int slot_proj   = 32;   // = d_model (per-slot projection target dim)
        int d_model     = 32;
        int d_ff        = 64;
        int num_heads   = 4;
        int num_blocks  = 2;
        int trunk_hidden = 64;
        int value_hidden = 32;
        NormPlacement norm = NormPlacement::PreNorm;
        float ln_eps = 1e-5f;
        uint64_t seed = 0xC0DE7777ULL;
    };

    SingleHeroNetTX() = default;
    void init(const Config& cfg);
    const Config& config() const { return cfg_; }

    void forward(const Tensor& x, float& value, Tensor& logits);
    void backward(float dValue, const Tensor& dLogits);

    int embed_dim()      const { return 3 * cfg_.d_model; }
    int trunk_dim()      const { return cfg_.trunk_hidden; }
    int policy_logits()  const { return head_.total_logits(); }

    // ICircuit
    const char* name() const override { return "SingleHeroNetTX"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    // Whole-blob helpers (with magic+version) mirroring SingleHeroNet/ST.
    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

    // GPU dispatch — migrates the two TransformerEncoder stacks. CPU pieces
    // stay on host.
    Device device() const { return device_; }
    void to(Device d);

    // Inspection (tests).
    Linear& self_fc1()  { return self_fc1_; }
    Linear& self_fc2()  { return self_fc2_; }
    Linear& enemy_proj(){ return enemy_proj_; }
    Linear& ally_proj() { return ally_proj_; }
    Linear& trunk()     { return trunk_; }
    TransformerEncoder& enemy_enc() { return enemy_enc_; }
    TransformerEncoder& ally_enc()  { return ally_enc_; }
    ValueHead& value_head() { return value_head_; }
    FactoredPolicyHead& policy_head() { return head_; }

private:
    Config cfg_{};

    // Self stream.
    Linear self_fc1_, self_fc2_;
    Tensor self_h_raw_, self_h_act_, self_z_;

    // Enemy stream.
    Linear enemy_proj_;
    TransformerEncoder enemy_enc_;
    Tensor enemy_in_;       // (K_ENEMIES, d_model) post per-slot proj
    Tensor enemy_out_;      // (K_ENEMIES, d_model) post encoder
    std::vector<float> e_mask_;
    std::vector<uint8_t> e_valid_;
    int e_n_valid_ = 0;
    Tensor enemy_pooled_;   // (d_model)

    // Ally stream.
    Linear ally_proj_;
    TransformerEncoder ally_enc_;
    Tensor ally_in_;
    Tensor ally_out_;
    std::vector<float> a_mask_;
    std::vector<uint8_t> a_valid_;
    int a_n_valid_ = 0;
    Tensor ally_pooled_;

    // Trunk + heads.
    Linear trunk_;
    Relu   trunk_act_;
    Tensor concat_;          // (3 * d_model)
    Tensor trunk_raw_, trunk_act_out_;
    ValueHead value_head_;
    FactoredPolicyHead head_;

    // Caches.
    Tensor x_cache_;

    Device device_ = Device::CPU;
};

} // namespace brogameagent::nn
