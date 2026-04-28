#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace brogameagent::grid {

// ─── FailureTape ──────────────────────────────────────────────────────────
//
// Records (state-signature, action) pairs from the tail of failed episodes
// and emits per-action prior multipliers that *suppress* repeating the
// recorded mistakes. Use it as a post-multiplier on the prior fed to
// GenericMcts: prior[a] *= max(floor, penalty^count) where count is how
// many times (sig, a) appears in the tape.
//
// Bookkeeping:
//   - One ring buffer of (sig, action) entries with cap = ring_capacity.
//   - One sig → per-action-count map kept in sync with the ring.
//   - record_failure() walks at most tape_depth entries from the tail of
//     the supplied trajectory.
//
// Signature is opaque bytes (std::string is just a byte container). The
// caller picks the encoding — typically a small hash of (cell, kind,
// near-tile pattern, ...). Cheaper signatures generalize across more
// states; richer signatures localize the penalty.
//
// Thread-safety: not thread-safe. Owned by the trainer thread; producers
// hand failure tails over a queue.

struct FailureStep {
    std::string sig;
    int action = -1;
};

struct FailureTapeConfig {
    int   tape_depth    = 16;     // tail steps recorded per failed episode
    int   ring_capacity = 4096;   // total live entries
    float penalty       = 0.5f;   // multiplier applied per repeated count
    float floor         = 0.05f;  // lower bound on emitted multiplier
};

class FailureTape {
public:
    explicit FailureTape(FailureTapeConfig cfg = {})
        : cfg_(cfg) {
        if (cfg_.ring_capacity < 1) cfg_.ring_capacity = 1;
        ring_.assign(static_cast<size_t>(cfg_.ring_capacity), {});
    }

    const FailureTapeConfig& config() const { return cfg_; }
    int   size()    const { return live_; }
    int   capacity() const { return cfg_.ring_capacity; }

    // Record the tail of a failed episode. Up to tape_depth entries are
    // taken from the END of `tail` (the steps closest to the failure).
    void record_failure(const std::vector<FailureStep>& tail) {
        int n = static_cast<int>(tail.size());
        int take = std::min(n, cfg_.tape_depth);
        int start = n - take;
        for (int i = 0; i < take; ++i) push(tail[start + i]);
    }

    // Multiply `prior` (length = num_actions) in place by penalty^count for
    // every (sig, action) entry in the ring matching `sig`, clamped at
    // `floor`. Actions outside [0, prior_size) are ignored.
    void apply_priors(const std::string& sig, float* prior, int prior_size) const {
        auto it = counts_.find(sig);
        if (it == counts_.end()) return;
        const auto& counts = it->second;
        int n = std::min<int>(prior_size, static_cast<int>(counts.size()));
        for (int a = 0; a < n; ++a) {
            uint32_t c = counts[a];
            if (c == 0) continue;
            float m = 1.0f;
            // penalty^c with explicit clamp; guards penalty <= 0 → floor.
            for (uint32_t k = 0; k < c; ++k) m *= cfg_.penalty;
            if (m < cfg_.floor) m = cfg_.floor;
            prior[a] *= m;
        }
    }

    // Convenience: return a multiplier vector (length = num_actions) of
    // floor-clamped penalty^count for `sig`. Untouched actions get 1.0.
    std::vector<float> multipliers(const std::string& sig, int num_actions) const {
        std::vector<float> m(static_cast<size_t>(num_actions), 1.0f);
        apply_priors(sig, m.data(), num_actions);
        return m;
    }

    // Drop everything.
    void clear() {
        for (auto& e : ring_) e = {};
        write_idx_ = 0;
        live_ = 0;
        counts_.clear();
    }

private:
    void push(const FailureStep& s) {
        // Evict the slot we're about to overwrite (if it holds a live
        // entry) so the count map stays consistent with the ring.
        FailureStep& slot = ring_[static_cast<size_t>(write_idx_)];
        if (!slot.sig.empty() || slot.action >= 0) {
            auto it = counts_.find(slot.sig);
            if (it != counts_.end()) {
                auto& v = it->second;
                if (slot.action >= 0 && slot.action < static_cast<int>(v.size())
                    && v[static_cast<size_t>(slot.action)] > 0) {
                    v[static_cast<size_t>(slot.action)]--;
                }
                bool any = false;
                for (auto c : v) { if (c) { any = true; break; } }
                if (!any) counts_.erase(it);
            }
            --live_;
        }

        // Write the new entry, increment map count.
        slot = s;
        if (s.action >= 0) {
            auto& v = counts_[s.sig];
            if (static_cast<int>(v.size()) <= s.action)
                v.resize(static_cast<size_t>(s.action + 1), 0u);
            v[static_cast<size_t>(s.action)]++;
            ++live_;
        }

        write_idx_ = (write_idx_ + 1) % cfg_.ring_capacity;
    }

    FailureTapeConfig                                       cfg_;
    std::vector<FailureStep>                                ring_;
    int                                                     write_idx_ = 0;
    int                                                     live_      = 0;
    std::unordered_map<std::string, std::vector<uint32_t>>  counts_;
};

} // namespace brogameagent::grid
