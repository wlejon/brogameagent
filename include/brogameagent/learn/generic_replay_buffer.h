#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace brogameagent::learn {

// ─── GenericSituation ──────────────────────────────────────────────────────
//
// One training tuple for PolicyValueNet:
//   obs            — raw observation vector (length = net.in_dim)
//   policy_target  — soft distribution over net.num_actions (sums ≈ 1)
//   action_mask    — optional legality mask (1.0 legal, 0.0 illegal). Empty
//                    means all actions are legal.
//   value_target   — scalar in [-1, 1] (typically discounted episode return,
//                    clipped).
//
// Sized at construction time so the buffer can hold heterogeneous-shape
// situations only if all situations in the buffer share net.in_dim and
// num_actions. The trainer enforces this implicitly by sampling whatever
// the producer pushed.

struct GenericSituation {
    std::vector<float> obs;
    std::vector<float> policy_target;
    std::vector<float> action_mask;   // empty == all legal
    float              value_target = 0.0f;
};

// ─── GenericReplayBuffer ───────────────────────────────────────────────────
//
// Fixed-capacity FIFO with uniform sampling. Same shape as ReplayBuffer but
// over GenericSituation. Single-producer / single-consumer safe; not
// designed for concurrent producers.

class GenericReplayBuffer {
public:
    explicit GenericReplayBuffer(size_t capacity = 4096) : cap_(capacity) {
        data_.reserve(capacity);
    }

    void push(const GenericSituation& s) {
        if (data_.size() < cap_) {
            data_.push_back(s);
        } else {
            data_[write_idx_] = s;
            write_idx_ = (write_idx_ + 1) % cap_;
        }
    }

    void push(GenericSituation&& s) {
        if (data_.size() < cap_) {
            data_.push_back(std::move(s));
        } else {
            data_[write_idx_] = std::move(s);
            write_idx_ = (write_idx_ + 1) % cap_;
        }
    }

    size_t size() const { return data_.size(); }
    size_t capacity() const { return cap_; }

    std::vector<GenericSituation> sample(size_t n, std::mt19937_64& rng) const {
        std::vector<GenericSituation> out;
        out.reserve(n);
        if (data_.empty()) return out;
        std::uniform_int_distribution<size_t> d(0, data_.size() - 1);
        for (size_t i = 0; i < n; ++i) out.push_back(data_[d(rng)]);
        return out;
    }

    const std::vector<GenericSituation>& all() const { return data_; }
    void clear() { data_.clear(); write_idx_ = 0; }

private:
    size_t cap_;
    size_t write_idx_ = 0;
    std::vector<GenericSituation> data_;
};

} // namespace brogameagent::learn
