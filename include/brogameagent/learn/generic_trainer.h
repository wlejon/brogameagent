#pragma once

#include "generic_replay_buffer.h"
#include <brotensor/tensor.h>
#include "brogameagent/nn/net.h"             // WeightsHandle
#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/net_tx.h"

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
    // Where compute happens. CPU (the default) runs the per-sample step path,
    // byte-identical to the pre-GPU trainer. Any other device runs the batched
    // step path: the net must already be migrated with net->to(device), and
    // per step the trainer stages obs/targets/masks onto `device`, then runs
    // forward, loss, backward, and the optimizer step there. Loss formulation
    // matches the CPU path (per-head softmax-xent mean-reduced; value MSE).
    brotensor::Device device = brotensor::Device::CPU;
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

    // Single-sample forward/backward. Mirrors the API both nets share.
    virtual void forward(const brotensor::Tensor& x, float& value, brotensor::Tensor& logits) = 0;
    virtual void backward(float dValue, const brotensor::Tensor& dLogits) = 0;

    // Batched-train forward/backward over a (B, *) minibatch. PolicyValueNet
    // implements these with true batched kernels; SingleHeroNetTX is adapted
    // via a per-sample loop. Tensors live on whatever device the net does.
    virtual void forward_batched_train(const brotensor::Tensor& X_BD,
                                       brotensor::Tensor& logits_BL,
                                       brotensor::Tensor& values_B1) = 0;
    virtual void backward_batched(const brotensor::Tensor& dLogits_BL,
                                  const brotensor::Tensor& dValues_B1) = 0;
};

// ─── GenericExItTrainer ───────────────────────────────────────────────────
//
// SGD+momentum mini-batch trainer. Per step():
//   - sample `batch` situations from the buffer
//   - forward each, compute value MSE (scalar) + masked-softmax cross-entropy
//   - scale gradients by 1/batch and the configured weights
//   - backprop, then a single sgd_step
//
// Caller invokes step() or step_n() at their cadence; trainer optionally
// publishes weights to a WeightsHandle every `publish_every` steps.

class GenericExItTrainer {
public:
    GenericExItTrainer() = default;
    // Non-copyable: holds a unique_ptr adapter and device-resident staging.
    GenericExItTrainer(const GenericExItTrainer&) = delete;
    GenericExItTrainer& operator=(const GenericExItTrainer&) = delete;

    // Set the net under training. Two overloads — one per concrete net — each
    // installs an INetForExIt adapter so the step paths stay net-agnostic.
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
    GenericTrainStep step_batched_();
    void ensure_staging_();

    INetForExIt*                net_    = nullptr;
    std::unique_ptr<INetForExIt> net_owned_;
    const GenericReplayBuffer*  buf_    = nullptr;
    nn::WeightsHandle*          handle_ = nullptr;
    GenericTrainerConfig        cfg_{};
    std::mt19937_64             rng_{0x1234567890ABCDEFULL};
    int                         steps_     = 0;
    int                         publishes_ = 0;
    int                         adam_step_ = 0;     // unused (SGD), reserved

    // Reusable batched-step staging buffers, allocated lazily on first batched
    // step on cfg_.device. All shaped (B, *) with B == cfg_.batch.
    brotensor::Tensor X_BD_;          // (B, in_dim)         observations
    brotensor::Tensor T_BL_;          // (B, n_act)          policy target
    brotensor::Tensor M_BL_;          // (B, n_act)          optional mask
    brotensor::Tensor V_B1_;          // (B, 1)              value target
    brotensor::Tensor logits_BL_;     // (B, n_act)          forward output
    brotensor::Tensor values_B1_;     // (B, 1)              forward output
    brotensor::Tensor probs_BL_;      // (B, n_act)          softmax output
    brotensor::Tensor dLog_BL_;       // (B, n_act)          d(loss)/d(logits)
    brotensor::Tensor dV_B1_;         // (B, 1)              d(loss)/d(value)
    brotensor::Tensor lp_per_sample_; // (B, 1)              policy loss per sample
    brotensor::Tensor lv_per_sample_; // (B, 1)              value  loss per sample
    bool  staging_ready_ = false;

    // Device int buffer (Dtype::INT32) for head_offsets (cumulative),
    // resident on cfg_.device — passed to softmax_xent_fused_batched.
    brotensor::Tensor head_offsets_dev_;
};

} // namespace brogameagent::learn
