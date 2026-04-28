#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace brogameagent::grid {

// ─── PotentialShaper ──────────────────────────────────────────────────────
//
// Potential-based reward shaping (Ng, Harada, Russell 1999): on each step,
// emit F = γ·Φ(s') − Φ(s) as an additive bonus to the env's true reward.
// PBRS preserves the optimal policy of the underlying MDP — the shaped
// optimum never disagrees with the unshaped one — while accelerating
// learning by densifying the reward signal.
//
// The caller computes the scalar potential Φ themselves (typically a
// negative distance-to-goal or a normalized progress score) and pushes
// it in via step(). reset(initial_phi) re-baselines at episode start
// so the first step's bonus is γΦ(s') − Φ(s_0) and not noise.
//
// Stateless w.r.t. snapshots — make a fresh PotentialShaper per env if
// you need to tree-search through it.

class PotentialShaper {
public:
    explicit PotentialShaper(float gamma = 0.99f) : gamma_(gamma) {}

    void  set_gamma(float g)  { gamma_ = g; }
    float gamma()       const { return gamma_; }
    float last_phi()    const { return last_phi_; }
    bool  primed()      const { return primed_; }

    // Set the baseline potential Φ(s_0). Call at episode start.
    void reset(float initial_phi) {
        last_phi_ = initial_phi;
        primed_   = true;
    }

    // Compute the shaping bonus for a transition into state with potential
    // `phi_new` and re-latch. Returns 0 on the very first call after a
    // fresh-construct (no baseline yet) to avoid emitting bogus bonus.
    float step(float phi_new) {
        if (!primed_) {
            last_phi_ = phi_new;
            primed_   = true;
            return 0.0f;
        }
        float bonus = gamma_ * phi_new - last_phi_;
        last_phi_   = phi_new;
        return bonus;
    }

private:
    float gamma_    = 0.99f;
    float last_phi_ = 0.0f;
    bool  primed_   = false;
};

// ─── StallDetector ────────────────────────────────────────────────────────
//
// Watches a scalar progress metric (e.g. monotonic distance covered, score,
// goal proximity) and reports whether it has advanced by at least ε over
// the last `patience` decisions. Use it as an early-termination signal so
// episodes that have wandered into stuck states don't waste rollouts.
//
// Idempotent until patience values have been seen; after that, stalled()
// = (max(window) − min(window)) < ε.

class StallDetector {
public:
    StallDetector(float epsilon = 0.0f, int patience = 60)
        : epsilon_(epsilon),
          patience_(patience > 0 ? patience : 1),
          ring_(static_cast<size_t>(patience_), 0.0f) {}

    void set_epsilon(float e)  { epsilon_  = e; }
    void set_patience(int p)   {
        patience_ = p > 0 ? p : 1;
        ring_.assign(static_cast<size_t>(patience_), 0.0f);
        write_idx_ = 0;
        filled_    = 0;
    }
    float epsilon()  const { return epsilon_; }
    int   patience() const { return patience_; }
    int   filled()   const { return filled_; }

    // Drop history. Call at episode start.
    void reset() {
        write_idx_ = 0;
        filled_    = 0;
    }

    // Push the latest progress sample. Returns true once enough samples
    // are present and the spread within the window is below epsilon.
    bool tick(float progress) {
        ring_[static_cast<size_t>(write_idx_)] = progress;
        write_idx_ = (write_idx_ + 1) % patience_;
        if (filled_ < patience_) ++filled_;
        if (filled_ < patience_) return false;
        float lo =  std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < patience_; ++i) {
            float v = ring_[static_cast<size_t>(i)];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return (hi - lo) < epsilon_;
    }

private:
    float              epsilon_;
    int                patience_;
    std::vector<float> ring_;
    int                write_idx_ = 0;
    int                filled_    = 0;
};

} // namespace brogameagent::grid
