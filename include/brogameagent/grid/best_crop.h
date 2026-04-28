#pragma once

#include <algorithm>
#include <any>
#include <cstdint>
#include <random>
#include <vector>

namespace brogameagent::grid {

// ─── BestCrop ─────────────────────────────────────────────────────────────
//
// Bounded ranked pool of "best so far" trajectories: each entry pairs a
// snapshot (the state at the start of the prefix) with an action prefix
// (sequence of actions to replay from that snapshot to reach the recorded
// state). Lets the live/inference policy seed its search from historical
// good positions instead of always replaying from episode spawn.
//
// Effective ranking score for an entry is:
//
//     E = score + depth_bonus * depth - age_decay * (age_now - born_age)
//
// Higher is better. depth_bonus rewards reaching the score later in the
// trajectory (preferring complete plays over short flukes); age_decay
// fades older entries so the pool tracks the current policy's frontier.
//
// Snapshot type is std::any so the caller can store whatever the env
// uses. The pool only treats snapshots opaquely.
//
// Thread-safety: not thread-safe.

struct BestEntry {
    std::any         snapshot;
    std::vector<int> prefix;   // action sequence to replay from snapshot
    float            score = 0.0f;
    int              depth = 0;
    uint64_t         born_age = 0;
};

struct BestCropConfig {
    int   capacity     = 64;
    float depth_bonus  = 0.0f;
    float age_decay    = 0.0f;
    // sample() picks uniformly from the top `seed_top_k` entries by E.
    // Clamped to [1, capacity].
    int   seed_top_k   = 4;
};

struct BestSeed {
    std::any         snapshot;
    std::vector<int> prefix;   // empty == apply snapshot only, no replay
};

class BestCrop {
public:
    explicit BestCrop(BestCropConfig cfg = {}) : cfg_(cfg) {}

    const BestCropConfig& config() const { return cfg_; }
    int  size()     const { return static_cast<int>(entries_.size()); }
    int  capacity() const { return cfg_.capacity; }
    bool empty()    const { return entries_.empty(); }

    // Insert a candidate. age_counter advances by 1 per push so newer
    // entries have lower "age" in subsequent rankings.
    void push(BestEntry e) {
        e.born_age = age_counter_++;
        entries_.push_back(std::move(e));
        prune_to_capacity();
    }

    // Same shape as push but accepting fields directly for ergonomics
    // when the caller doesn't pre-build a struct.
    void push(std::any snap, std::vector<int> prefix, float score, int depth) {
        BestEntry e;
        e.snapshot = std::move(snap);
        e.prefix   = std::move(prefix);
        e.score    = score;
        e.depth    = depth;
        push(std::move(e));
    }

    // Pick a seed for a fresh search. Picks uniformly from the top-K
    // entries by current effective score E. Returns {empty, empty} if
    // the pool is empty.
    BestSeed seed(std::mt19937_64& rng) const {
        if (entries_.empty()) return {};
        std::vector<int> idx = ranked_indices();
        int k = std::clamp(cfg_.seed_top_k, 1, static_cast<int>(idx.size()));
        std::uniform_int_distribution<int> d(0, k - 1);
        const auto& e = entries_[static_cast<size_t>(idx[d(rng)])];
        return { e.snapshot, e.prefix };
    }

    // Read-only view of entries sorted best-first, useful for
    // inspection / debug UIs / tests.
    std::vector<const BestEntry*> sorted() const {
        auto idx = ranked_indices();
        std::vector<const BestEntry*> out;
        out.reserve(idx.size());
        for (int i : idx) out.push_back(&entries_[static_cast<size_t>(i)]);
        return out;
    }

    void clear() { entries_.clear(); age_counter_ = 0; }

    // Effective score for entry `i` at the current age clock. Exposed
    // for tests / introspection.
    float effective_score(size_t i) const {
        const auto& e = entries_[i];
        uint64_t age = age_counter_ > e.born_age ? age_counter_ - e.born_age : 0;
        return e.score + cfg_.depth_bonus * static_cast<float>(e.depth)
             - cfg_.age_decay * static_cast<float>(age);
    }

private:
    std::vector<int> ranked_indices() const {
        std::vector<int> idx(entries_.size());
        for (size_t i = 0; i < entries_.size(); ++i) idx[i] = static_cast<int>(i);
        std::sort(idx.begin(), idx.end(), [this](int a, int b) {
            return effective_score(static_cast<size_t>(a))
                 > effective_score(static_cast<size_t>(b));
        });
        return idx;
    }

    void prune_to_capacity() {
        if (cfg_.capacity <= 0) { entries_.clear(); return; }
        if (static_cast<int>(entries_.size()) <= cfg_.capacity) return;
        // Drop worst-scoring entries until we fit.
        auto idx = ranked_indices();
        std::vector<BestEntry> kept;
        kept.reserve(static_cast<size_t>(cfg_.capacity));
        for (int i = 0; i < cfg_.capacity; ++i)
            kept.push_back(std::move(entries_[static_cast<size_t>(idx[i])]));
        entries_ = std::move(kept);
    }

    BestCropConfig          cfg_;
    std::vector<BestEntry>  entries_;
    uint64_t                age_counter_ = 0;
};

} // namespace brogameagent::grid
