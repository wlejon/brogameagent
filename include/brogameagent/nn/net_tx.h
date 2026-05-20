#pragma once

#include "circuits.h"
#include "heads.h"
#include "layernorm.h"
#include "brogameagent/learn/batched_net.h"
#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include "transformer_block.h"
#include "transformer_encoder.h"
#include "brogameagent/observation.h"

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
// Device dispatch: when to(d) is called every Linear, the two
// TransformerEncoder stacks, and the heads migrate to device. The brotensor
// ops dispatch on operand device at runtime, so one forward/backward path
// runs on whatever device the tensors live on.

class SingleHeroNetTX : public ICircuit, public learn::BatchedNet {
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

    // ─── ExIt-trainer compatibility surface ────────────────────────────────
    // Mirror of PolicyValueNet's accessors so GenericExItTrainer can drive
    // either net through the same code path. head_offsets() is in the same
    // "exclusive prefix sums + trailing total" format PolicyValueNet exposes.
    int in_dim()      const { return observation::TOTAL; }
    int num_actions() const { return head_.total_logits(); }
    int num_heads()   const { return static_cast<int>(head_sizes_.size()); }
    const std::vector<int>& head_sizes()   const { return head_sizes_; }
    const std::vector<int>& head_offsets() const { return head_offsets_; }

    void forward(const brotensor::Tensor& x, float& value, brotensor::Tensor& logits);
    void backward(float dValue, const brotensor::Tensor& dLogits);

    int input_dim()  const override { return observation::TOTAL; }
    int logits_dim() const override { return head_.total_logits(); }

    // BatchedNet: inference-only batched forward. SingleHeroNetTX has no true
    // batched kernels — this loops the single-sample forward over row views of
    // the (B, *) staging tensors. Ops dispatch by device, so it is device-
    // neutral. logits_BL / values_B1 are (re)allocated on X_BD's device.
    void forward_batched(const brotensor::Tensor& X_BD,
                         brotensor::Tensor& logits_BL,
                         brotensor::Tensor& values_B1) override;

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

    // Device migration — migrates every Linear, both TransformerEncoder
    // stacks, the heads, and the directly-owned activation caches.
    brotensor::Device device() const override { return device_; }
    void to(brotensor::Device d);

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
    Relu   self_act_;
    brotensor::Tensor self_h_raw_, self_h_act_, self_z_;
    brotensor::Tensor self_in_;        // sliced self input (device-resident)

    // Enemy stream. enemy_slotin_ holds the (K, ENEMY_FEATURES) sliced slot
    // inputs; e_mask_ is a device-resident (K, 1) validity mask built by
    // brotensor::build_slot_mask and consumed by the transformer + pool.
    Linear enemy_proj_;
    TransformerEncoder enemy_enc_;
    brotensor::Tensor enemy_slotin_;   // (K_ENEMIES, ENEMY_FEATURES)
    brotensor::Tensor enemy_in_;       // (K_ENEMIES, d_model) post per-slot proj
    brotensor::Tensor enemy_out_;      // (K_ENEMIES, d_model) post encoder
    brotensor::Tensor e_mask_;         // (K_ENEMIES, 1) device-resident mask
    brotensor::Tensor enemy_pooled_;   // (d_model)

    // Ally stream.
    Linear ally_proj_;
    TransformerEncoder ally_enc_;
    brotensor::Tensor ally_slotin_;    // (K_ALLIES, ALLY_FEATURES)
    brotensor::Tensor ally_in_;
    brotensor::Tensor ally_out_;
    brotensor::Tensor a_mask_;         // (K_ALLIES, 1) device-resident mask
    brotensor::Tensor ally_pooled_;

    // Trunk + heads.
    Linear trunk_;
    Relu   trunk_act_;
    brotensor::Tensor concat_;          // (3 * d_model)
    brotensor::Tensor trunk_raw_, trunk_act_out_;
    ValueHead value_head_;
    FactoredPolicyHead head_;

    // Caches.
    brotensor::Tensor x_cache_;

    // Resolved head shape (matches PolicyValueNet's contract):
    //   head_sizes_   = {N_MOVE, N_ATTACK, N_ABILITY} from FactoredPolicyHead.
    //   head_offsets_ = exclusive prefix sums of head_sizes_, with a trailing
    //                   entry equal to total_logits().
    std::vector<int> head_sizes_;
    std::vector<int> head_offsets_;

    brotensor::Device device_ = brotensor::Device::CPU;
};

} // namespace brogameagent::nn
