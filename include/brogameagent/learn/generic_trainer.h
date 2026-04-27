#pragma once

#include "generic_replay_buffer.h"
#include "brogameagent/nn/net.h"             // WeightsHandle
#include "brogameagent/nn/policy_value_net.h"

#include <cstdint>
#include <random>

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
};

struct GenericTrainStep {
    float loss_value  = 0.0f;
    float loss_policy = 0.0f;
    float loss_total  = 0.0f;
    int   samples     = 0;
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

    void set_net(nn::PolicyValueNet* net) { net_ = net; }
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

    nn::PolicyValueNet*         net_    = nullptr;
    const GenericReplayBuffer*  buf_    = nullptr;
    nn::WeightsHandle*          handle_ = nullptr;
    GenericTrainerConfig        cfg_{};
    std::mt19937_64             rng_{0x1234567890ABCDEFULL};
    int                         steps_     = 0;
    int                         publishes_ = 0;
};

} // namespace brogameagent::learn
