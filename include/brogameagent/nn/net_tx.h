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
#include "brogameagent/learn/batched_net.h"
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
// GPU dispatch: when to(Device::GPU) is called every Linear, the two
// TransformerEncoder stacks, and the heads migrate to device. The pipeline is
// fully GPU-resident — no host↔device shuttling per forward/backward beyond
// the unavoidable upload of `x` (CPU API entrypoint) and download of value /
// logits at the boundary. The native GPU overloads (taking GpuTensor) avoid
// even those.

class SingleHeroNetTX : public ICircuit
#ifdef BGA_HAS_CUDA
    , public brogameagent::learn::BatchedNet
#endif
{
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

    void forward(const Tensor& x, float& value, Tensor& logits);
    void backward(float dValue, const Tensor& dLogits);

#ifdef BGA_HAS_CUDA
    // GPU-native forward/backward. Net must be on Device::GPU.
    //   x:      (TOTAL, 1) device tensor — caller keeps alive until backward.
    //   logits: (policy_logits, 1) — overwritten.
    // Value is cached in value_gpu(). For backward, the caller writes
    // d(loss)/d(value) into dValue_gpu() and supplies dLogits.
    void forward(const gpu::GpuTensor& x, gpu::GpuTensor& logits);
    void backward(const gpu::GpuTensor& dLogits);

    const gpu::GpuTensor& value_gpu() const { return value_head_.value_gpu(); }
    gpu::GpuTensor&       dValue_gpu()      { return value_head_.dValue_gpu(); }

    // Batched-inference forward. The whole input is downloaded once,
    // staging buffers are built in a single host pass, and the network
    // runs end-to-end on device: self stream and per-slot projections are
    // truly batched (one Linear launch each); the encoder + masked
    // mean-pool and the heads are still looped per batch element but
    // queue into the default stream without host blocks. Outputs are
    // gathered into logits_BL / values_B1 via stream-ordered D2D copies.
    // Further speedups require batched (B, K, D) MHA / LayerNorm / FF
    // kernels and a batched value+policy head.
    void forward_batched(const gpu::GpuTensor& X_BD,
                         gpu::GpuTensor& logits_BL,
                         gpu::GpuTensor& values_B1) override;

    // ─── Training-time batched API ────────────────────────────────────────
    //
    // Training-time variants for GenericExItTrainer. These are correctness-
    // first per-batch-element loops over the existing single-sample GPU
    // forward/backward — NOT yet a true (B, K, D) batched dispatch. Each
    // backward re-runs forward(x_b) before backward(dLogits_b) to rebuild
    // the per-element layer caches the single-sample backward depends on.
    // That doubles forward FLOPs per training step; accepting the cost in
    // v2 because every layer cache (slot validity masks, encoder activations,
    // per-slot Linear inputs) is sized for one sample.
    //
    // A future optimization adds per-sample state arrays so forward and
    // backward run once each, or — better — true (B, K, D) attention/FF
    // kernels.
    void forward_batched_train(const gpu::GpuTensor& X_BD,
                               gpu::GpuTensor& logits_BL,
                               gpu::GpuTensor& values_B1);
    void backward_batched(const gpu::GpuTensor& dLogits_BL,
                          const gpu::GpuTensor& dValues_B1);

    // BatchedNet interface accessors.
    int input_dim()  const override { return observation::TOTAL; }
    int logits_dim() const override { return head_.total_logits(); }
#endif

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
    Relu   self_act_;
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

    // Resolved head shape (matches PolicyValueNet's contract):
    //   head_sizes_   = {N_MOVE, N_ATTACK, N_ABILITY} from FactoredPolicyHead.
    //   head_offsets_ = exclusive prefix sums of head_sizes_, with a trailing
    //                   entry equal to total_logits().
    std::vector<int> head_sizes_;
    std::vector<int> head_offsets_;

    Device device_ = Device::CPU;
#ifdef BGA_HAS_CUDA
    // GPU staging / activation buffers, allocated at to(GPU).
    // These are sized to fixed observation constants and reused.
    gpu::GpuTensor x_g_;                 // (TOTAL, 1) input copy (forward owns it for CPU-API path)
    gpu::GpuTensor self_in_g_;           // (SELF_FEATURES, 1)
    gpu::GpuTensor self_h_raw_g_, self_h_act_g_, self_z_g_;
    gpu::GpuTensor enemy_in_g_, enemy_out_g_;
    gpu::GpuTensor ally_in_g_,  ally_out_g_;
    gpu::GpuTensor e_mask_g_, a_mask_g_;     // (K_ENEMIES,1) / (K_ALLIES,1) device masks
    gpu::GpuTensor enemy_pooled_g_;          // (D, 1)
    gpu::GpuTensor ally_pooled_g_;           // (D, 1)
    gpu::GpuTensor concat_g_;                // (3D, 1)
    gpu::GpuTensor trunk_raw_g_, trunk_act_g_;
    // Per-slot projection scratch (one slot at a time).
    gpu::GpuTensor slot_in_e_g_;             // (ENEMY_FEATURES, 1)
    gpu::GpuTensor slot_in_a_g_;             // (ALLY_FEATURES, 1)
    gpu::GpuTensor slot_proj_g_;             // (D, 1)
    // Backward scratch.
    gpu::GpuTensor dTrunkAct_g_;
    gpu::GpuTensor dTrunkRaw_g_;
    gpu::GpuTensor dTrunkFromV_g_;
    gpu::GpuTensor dTrunkFromP_g_;
    gpu::GpuTensor dConcat_g_;
    gpu::GpuTensor dSelfZ_g_;
    gpu::GpuTensor dSelfHact_g_, dSelfHraw_g_, dSelfIn_g_;
    gpu::GpuTensor dEnemyOut_g_, dEnemyIn_g_;
    gpu::GpuTensor dAllyOut_g_,  dAllyIn_g_;
    gpu::GpuTensor dSlotProj_g_;
    gpu::GpuTensor dSlotInE_g_;
    gpu::GpuTensor dSlotInA_g_;

    // External input view used by the GPU-native forward path. Non-owning;
    // the caller's lifetime guarantees apply to it.
    const gpu::GpuTensor* x_external_ = nullptr;

    // Training-time batched scratch. Holds the last X_BD pointer that
    // forward_batched_train was called with so backward_batched can re-run
    // forward(x_b) per element.
    const gpu::GpuTensor* last_train_X_BD_ = nullptr;
    gpu::GpuTensor x_row_g_;          // (TOTAL, 1) per-element view buffer
    gpu::GpuTensor logits_row_g_;     // (L, 1)     per-element logits scratch
    gpu::GpuTensor dLogits_row_g_;    // (L, 1)     per-element dLogits scratch
#endif
};

} // namespace brogameagent::nn
