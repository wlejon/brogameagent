#pragma once

#include "replay_buffer.h"
#include "brogameagent/nn/net.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brogameagent::learn {

// ─── ExItTrainer ───────────────────────────────────────────────────────────
//
// Consumes Situations, runs SGD+momentum mini-batch updates against the
// (value, policy) targets, and optionally publishes new weights to a
// WeightsHandle after each publish_every steps. The trainer does not own
// the loop scheduling — callers invoke step() or step_n(). Intended uses:
//
//   - Synchronous in-process training (CLI tools) — call step_n() until
//     loss stabilises, then save().
//   - Asynchronous background thread — one thread calls step() forever;
//     the game thread snapshots via WeightsHandle.

struct TrainerConfig {
    float    lr        = 0.01f;
    float    momentum  = 0.9f;
    int      batch     = 32;
    float    policy_weight = 1.0f;
    float    value_weight  = 1.0f;
    uint64_t rng_seed  = 0x1234567890ABCDEFULL;
    int      publish_every = 100;   // steps between WeightsHandle publish
};

struct TrainStep {
    float loss_value = 0.0f;
    float loss_policy = 0.0f;
    float loss_total  = 0.0f;
    int   samples     = 0;
};

class ExItTrainer {
public:
    ExItTrainer() = default;

    void set_net(nn::SingleHeroNet* net) { net_ = net; }
    void set_buffer(const ReplayBuffer* buf) { buf_ = buf; }
    void set_config(const TrainerConfig& cfg) { cfg_ = cfg; rng_.seed(cfg.rng_seed); }
    void set_weights_handle(nn::WeightsHandle* h) { handle_ = h; }

    const TrainerConfig& config() const { return cfg_; }

    // Run one mini-batch SGD step. Returns loss components (mean over batch).
    TrainStep step();

    // Run n steps; returns the last step's losses.
    TrainStep step_n(int n);

    int  total_steps()    const { return steps_; }
    int  total_publishes() const { return publishes_; }

private:
    void maybe_publish();

    nn::SingleHeroNet*        net_    = nullptr;
    const ReplayBuffer*       buf_    = nullptr;
    nn::WeightsHandle*        handle_ = nullptr;
    TrainerConfig             cfg_{};
    std::mt19937_64           rng_{0x1234567890ABCDEFULL};
    int                       steps_     = 0;
    int                       publishes_ = 0;
};

} // namespace brogameagent::learn
