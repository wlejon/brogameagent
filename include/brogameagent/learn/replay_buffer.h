#pragma once

#include "brogameagent/action_mask.h"
#include "brogameagent/nn/heads.h"
#include "brogameagent/nn/tensor.h"
#include "brogameagent/observation.h"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace brogameagent::learn {

// ─── Situation ─────────────────────────────────────────────────────────────
//
// One training example: the observation at a decision, the legal-action
// mask, the MCTS-derived target distributions, and the eventual discounted
// return. Move target sums to 1 over 9 classes; attack target sums to 1
// over N_ATTACK; ability over N_ABILITY. Illegal entries are zero.

struct Situation {
    std::array<float, observation::TOTAL>                  obs{};
    std::array<float, action_mask::N_ENEMY_SLOTS>          atk_mask{};
    std::array<float, action_mask::N_ABILITY_SLOTS>        abil_mask{};
    std::array<float, nn::FactoredPolicyHead::N_MOVE>      target_move{};
    std::array<float, nn::FactoredPolicyHead::N_ATTACK>    target_attack{};
    std::array<float, nn::FactoredPolicyHead::N_ABILITY>   target_ability{};
    float value_target = 0.0f;   // in [-1, 1]
};

// ─── ReplayBuffer ──────────────────────────────────────────────────────────
//
// Fixed-capacity FIFO with uniform sampling. Thread-safe against one
// producer and one consumer (sampler).

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity = 4096) : cap_(capacity) {
        data_.reserve(capacity);
    }

    void push(const Situation& s) {
        if (data_.size() < cap_) {
            data_.push_back(s);
        } else {
            data_[write_idx_] = s;
            write_idx_ = (write_idx_ + 1) % cap_;
        }
    }

    size_t size() const { return data_.size(); }
    size_t capacity() const { return cap_; }

    // Sample n situations uniformly with replacement.
    std::vector<Situation> sample(size_t n, std::mt19937_64& rng) const {
        std::vector<Situation> out;
        out.reserve(n);
        if (data_.empty()) return out;
        std::uniform_int_distribution<size_t> d(0, data_.size() - 1);
        for (size_t i = 0; i < n; ++i) out.push_back(data_[d(rng)]);
        return out;
    }

    const std::vector<Situation>& all() const { return data_; }
    void clear() { data_.clear(); write_idx_ = 0; }

private:
    size_t cap_;
    size_t write_idx_ = 0;
    std::vector<Situation> data_;
};

} // namespace brogameagent::learn
