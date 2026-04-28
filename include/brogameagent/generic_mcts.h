#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace brogameagent::mcts {

// ─── GenericMcts ───────────────────────────────────────────────────────────
//
// PUCT MCTS over a fully-opaque environment. Unlike Mcts/DecoupledMcts/etc.,
// this class does not assume the brogameagent World/Agent/CombatAction model
// — the env is described entirely by callbacks supplied at construction
// time. State is held as std::any so the env can choose its own snapshot
// representation (a struct, a shared_ptr, a JSValue, anything).
//
// Use this when the env you want to plan over is something other than the
// bundled combat sim — a custom physics sim, a JS-side env via FFI, a board
// game, a planner test harness. For combat-style RTS/MOBA scenarios, the
// existing Mcts family is more efficient (it avoids the std::function and
// std::any indirection by being typed against the in-library World).
//
// Algorithm matches the AlphaZero PUCT formulation: at each tree node,
//
//     score(a) = Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a))
//
// Q is the action's mean discounted return; P is the prior (uniform when
// no prior_fn is provided); N is visit count. Leaf evaluation uses
// value_fn when set, otherwise a random rollout of cfg.rollout_depth steps.
// Optional Dirichlet noise on the root prior matches the AlphaZero
// exploration scheme.
//
// Snapshot semantics: search() captures one snapshot at the root and uses
// it to reset the env at the start of every iteration. The MCTS does not
// snapshot at every node; per-edge rewards are recorded at expansion time
// so backprop can replay the discounted return without further snapshots.
// The env owns deep-copy / refcount semantics for its snapshot type — the
// MCTS only treats it as opaque.
//
// Thread-safety: not thread-safe. One instance per searcher. For root-
// parallel search across multiple threads, allocate one GenericMcts (and
// one env) per thread and merge root visits externally.

// Outcome of one env step. Reward is the per-decision delta; done flips to
// true on terminal states (death, win, timeout, …).
struct GenericStepResult {
    float reward = 0.0f;
    bool  done   = false;
};

// Env contract. All callbacks are required.
struct GenericEnv {
    // Capture current env state. The MCTS holds the result for the
    // duration of one search() call and hands it back to restore_fn at
    // the start of every iteration.
    std::function<std::any()> snapshot_fn;

    // Restore env to a previously snapshotted state. Called once per
    // iteration before tree descent.
    std::function<void(const std::any&)> restore_fn;

    // Apply an action; mutates env. Reward is the per-decision delta.
    std::function<GenericStepResult(int action)> step_fn;

    // Legal action indices in [0, num_actions). Re-queried at every
    // expansion, so it must reflect the current env state.
    std::function<std::vector<int>()> legal_actions_fn;

    // Observation vector forwarded to prior_fn / value_fn. Size and
    // semantics are entirely up to the env-consumer pair; MCTS does
    // not interpret it.
    std::function<std::vector<float>()> observe_fn;

    // Total action space size (max action index + 1). Mask shape.
    int num_actions = 0;
};

// Optional learned-policy prior. Returns a probability per action over
// [0, num_actions). Only entries at `legal` indices are read; the rest
// can be anything. Leave unset for a uniform prior (plain UCT modulated
// by c_puct).
using GenericPriorFn = std::function<std::vector<float>(
    const std::vector<float>& obs,
    const std::vector<int>& legal_actions)>;

// Optional learned-value function in [-1, 1]. When set, leaf evaluation
// uses this scalar instead of running a random rollout. With a strong
// value head this dominates random rollout; with a weak one it's worse.
using GenericValueFn = std::function<float(const std::vector<float>& obs)>;

struct GenericMctsConfig {
    int   iterations    = 100;       // total iterations per search()
    float c_puct        = 1.5f;      // PUCT exploration constant
    float gamma         = 0.99f;     // discount per decision step
    int   rollout_depth = 8;         // used only when value_fn unset

    // AlphaZero-style root noise. Mixed into the root prior at the start
    // of each search() call. Off by default (alpha=0); typical values
    // (0.3, 0.25) when on.
    float    dirichlet_alpha   = 0.0f;
    float    dirichlet_epsilon = 0.0f;
    uint64_t seed              = 0xC0DE1234ULL;
};

struct GenericSearchStats {
    int iterations  = 0;   // iterations actually executed
    int tree_size   = 0;   // total nodes ever allocated under the current root
    int best_visits = 0;   // visit count of the chosen action
    int best_action = -1;  // most-visited root action, or -1 if no search ran
};

class GenericMcts {
public:
    // Internal node type, defined in the .cpp. Forward declared so unique_ptr
    // members can name it; not part of the public surface.
    class Node;

    explicit GenericMcts(GenericEnv env);
    ~GenericMcts();

    GenericMcts(const GenericMcts&) = delete;
    GenericMcts& operator=(const GenericMcts&) = delete;
    GenericMcts(GenericMcts&&) noexcept;
    GenericMcts& operator=(GenericMcts&&) noexcept;

    void set_config(const GenericMctsConfig& cfg) { cfg_ = cfg; }
    const GenericMctsConfig& config() const { return cfg_; }

    void set_prior_fn(GenericPriorFn fn) { prior_fn_ = std::move(fn); }
    void set_value_fn(GenericValueFn fn) { value_fn_ = std::move(fn); }

    // Run a search starting from the env's current state. Returns the
    // most-visited root action, or -1 if the action space is empty.
    // Snapshot/restore is used internally; the env is left in its
    // pre-search state on exit.
    int search();

    // Normalized root visit distribution over [0, num_actions). Empty
    // (zero-sized) before the first search; otherwise sums to 1 over
    // visited actions. Useful as a policy target for ExIt training.
    std::vector<float> root_visits() const;

    // Promote the subtree rooted at `action` to the new root, reusing
    // its statistics. If `action` was never expanded (or is out of
    // range), the tree is dropped — next search() rebuilds from scratch.
    void advance_root(int action);

    // Drop the search tree.
    void reset();

    const GenericSearchStats& last_stats() const { return stats_; }

private:
    GenericEnv         env_;
    GenericMctsConfig  cfg_{};
    GenericPriorFn     prior_fn_;
    GenericValueFn     value_fn_;
    GenericSearchStats stats_{};
    std::unique_ptr<Node> root_;
    int tree_size_ = 0;
};

} // namespace brogameagent::mcts
