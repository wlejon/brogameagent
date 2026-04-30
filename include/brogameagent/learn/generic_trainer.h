#pragma once

#include "generic_replay_buffer.h"
#include "brogameagent/nn/device.h"
#include "brogameagent/nn/net.h"             // WeightsHandle
#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/net_tx.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/tensor.h"
#endif

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace brogameagent::learn {

// ─── GenericTrainerConfig / TrainStep ─────────────────────────────────────
//
// Same knob set as TrainerConfig, repeated here so the generic trainer
// doesn't depend on the MOBA-shaped `trainer.h`.

struct GenericTrainerConfig {
    float    lr             = 0.01f;
    float    momentum       = 0.9f;
    int      batch          = 32;
    float    policy_weight  = 1.0f;
    float    value_weight   = 1.0f;
    uint64_t rng_seed       = 0x1234567890ABCDEFULL;
    int      publish_every  = 100;
    // Where compute happens. CPU is the default, byte-identical to the
    // pre-GPU trainer. When set to GPU the trainer expects the net to have
    // already been moved with net->to(Device::GPU); per-step it uploads obs,
    // policy targets, masks, and value targets to layer-private GPU staging
    // tensors and runs forward, loss, backward, and the optimizer step on
    // device. Loss formulation matches CPU exactly (per-head softmax-xent
    // mean-reduced; value MSE).
    nn::Device device = nn::Device::CPU;
};

struct GenericTrainStep {
    float loss_value  = 0.0f;
    float loss_policy = 0.0f;
    float loss_total  = 0.0f;
    int   samples     = 0;
};

// ─── INetForExIt ──────────────────────────────────────────────────────────
//
// Minimal interface GenericExItTrainer relies on. Both PolicyValueNet (the
// MLP) and SingleHeroNetTX (the transformer) implement this surface; the
// trainer holds an INetForExIt* and the public set_net(...) overloads adapt
// each concrete net to it.
//
// Lives in this header (rather than its own file) because it's purely a
// trainer-private abstraction — callers shouldn't depend on it.

class INetForExIt {
public:
    virtual ~INetForExIt() = default;

    // Static shape.
    virtual int in_dim() const = 0;
    virtual int num_actions() const = 0;
    virtual int num_heads() const = 0;
    virtual const std::vector<int>& head_offsets() const = 0;

    // Param-tensor lifecycle.
    virtual void zero_grad() = 0;
    virtual void sgd_step(float lr, float momentum) = 0;
    virtual std::vector<uint8_t> save() const = 0;

    // CPU forward/backward. Mirrors the single-sample API both nets share.
    virtual void forward(const nn::Tensor& x, float& value, nn::Tensor& logits) = 0;
    virtual void backward(float dValue, const nn::Tensor& dLogits) = 0;

#ifdef BGA_HAS_CUDA
    // GPU batched-train forward/backward. PolicyValueNet implements these
    // with true batched kernels; SingleHeroNetTX implements them via
    // per-element loops over the single-sample GPU forward/backward.
    virtual void forward_batched_train(const nn::gpu::GpuTensor& X_BD,
                                       nn::gpu::GpuTensor& logits_BL,
                                       nn::gpu::GpuTensor& values_B1) = 0;
    virtual void backward_batched(const nn::gpu::GpuTensor& dLogits_BL,
                                  const nn::gpu::GpuTensor& dValues_B1) = 0;
#endif
};

// ─── GenericExItTrainer ───────────────────────────────────────────────────
//
// SGD+momentum mini-batch trainer for PolicyValueNet. Per step():
//   - sample `batch` situations from the buffer
//   - forward each, compute value MSE (scalar) + masked-softmax cross-entropy
//   - scale gradients by 1/batch and the configured weights
//   - backprop, then a single sgd_step
//
// Identical scheduling contract as ExItTrainer: caller invokes step() or
// step_n() at their cadence; trainer optionally publishes weights to a
// WeightsHandle every `publish_every` steps.

class GenericExItTrainer {
public:
    GenericExItTrainer() = default;
#ifdef BGA_HAS_CUDA
    ~GenericExItTrainer();
    GenericExItTrainer(const GenericExItTrainer&) = delete;
    GenericExItTrainer& operator=(const GenericExItTrainer&) = delete;
#endif

    // Set the net under training. Two overloads — one per concrete net — each
    // installs an INetForExIt adapter so step_cpu_/step_gpu_ stay net-agnostic.
    void set_net(nn::PolicyValueNet* net);
    void set_net(nn::SingleHeroNetTX* net);
    void set_buffer(const GenericReplayBuffer* buf) { buf_ = buf; }
    void set_config(const GenericTrainerConfig& cfg) { cfg_ = cfg; rng_.seed(cfg.rng_seed); }
    void set_weights_handle(nn::WeightsHandle* h) { handle_ = h; }

    const GenericTrainerConfig& config() const { return cfg_; }

    GenericTrainStep step();
    GenericTrainStep step_n(int n);

    int total_steps()     const { return steps_; }
    int total_publishes() const { return publishes_; }

private:
    void maybe_publish();

    GenericTrainStep step_cpu_();
#ifdef BGA_HAS_CUDA
    GenericTrainStep step_gpu_();
    void ensure_gpu_staging_();
#endif

    INetForExIt*                net_    = nullptr;
    std::unique_ptr<INetForExIt> net_owned_;          // adapter owns nothing externally
    const GenericReplayBuffer*  buf_    = nullptr;
    nn::WeightsHandle*          handle_ = nullptr;
    GenericTrainerConfig        cfg_{};
    std::mt19937_64             rng_{0x1234567890ABCDEFULL};
    int                         steps_     = 0;
    int                         publishes_ = 0;
    int                         adam_step_ = 0;     // unused (SGD), reserved

#ifdef BGA_HAS_CUDA
    // Reusable batched-step GPU staging buffers, allocated lazily on first
    // GPU step. All shaped (B, *) where B == cfg_.batch — sized once and
    // never re-resized, regardless of how many valid samples land in the
    // minibatch (we always upload exactly B rows per step).
    nn::gpu::GpuTensor X_BD_g_;          // (B, in_dim)         observations
    nn::gpu::GpuTensor T_BL_g_;          // (B, n_act)          policy target
    nn::gpu::GpuTensor M_BL_g_;          // (B, n_act)          optional mask
    nn::gpu::GpuTensor V_B1_g_;          // (B, 1)              value target
    nn::gpu::GpuTensor logits_BL_g_;     // (B, n_act)          forward output
    nn::gpu::GpuTensor values_B1_g_;     // (B, 1)              forward output
    nn::gpu::GpuTensor probs_BL_g_;      // (B, n_act)          softmax output
    nn::gpu::GpuTensor dLog_BL_g_;       // (B, n_act)          d(loss)/d(logits)
    nn::gpu::GpuTensor dV_B1_g_;         // (B, 1)              d(loss)/d(value)
    nn::gpu::GpuTensor lp_per_sample_g_; // (B, 1) policy loss per sample
    nn::gpu::GpuTensor lv_per_sample_g_; // (B, 1) value  loss per sample
    bool  has_mask_for_step_ = false;    // re-evaluated per step
    bool  gpu_ready_         = false;

    // Owned device int* for head_offsets (cumulative). Sized at gpu_ready_
    // time to net->head_offsets().size(); freed in the destructor. Plain
    // raw pointer so this header doesn't pull in <cuda_runtime.h>.
    int*  head_offsets_dev_   = nullptr;
    int   head_offsets_dev_n_ = 0;
#endif
};

} // namespace brogameagent::learn
