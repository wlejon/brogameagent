#pragma once

#include "action_mask.h"
#include "agent.h"
#include "world.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace brogameagent::mcts {

// ─── CombatAction ──────────────────────────────────────────────────────────
//
// A discrete, self-contained action one agent can commit to for one decision
// window. "Decision window" = ACTION_REPEAT sim ticks in the Simulation
// harness; MCTS plans at that granularity, not per-tick.
//
// Field layout is deliberately narrow so the action fits in 4 bytes and
// vectors of actions stay cache-friendly for large branching factors.

enum class MoveDir : int8_t {
    Hold = 0,  // velocity zero
    N    = 1,  // -Z in local frame (FPS forward)
    NE   = 2,
    E    = 3,
    SE   = 4,
    S    = 5,
    SW   = 6,
    W    = 7,
    NW   = 8,
    COUNT = 9,
};

struct CombatAction {
    MoveDir move_dir    = MoveDir::Hold;
    int8_t  attack_slot = -1;   // -1 = no auto-attack; else 0..N_ENEMY_SLOTS-1
    int8_t  ability_slot= -1;   // -1 = no cast;       else 0..N_ABILITY_SLOTS-1

    bool operator==(const CombatAction&) const = default;
};

/// Legal-action enumeration against a (world, agent) pair. Uses action_mask
/// to decide attack / ability legality; movement is always legal in all 9
/// directions. Targeting semantics: the ability target is the same enemy as
/// attack_slot when both are set; otherwise the nearest enemy slot (0), or
/// -1 for self-cast. That choice is baked in at apply-time, not here.
///
/// Returned vector contains every meaningful combination (product of 9 move
/// dirs × legal attack set × legal ability set), deduplicated and filtered
/// to avoid e.g. "attack -1, ability -1" from being emitted 9 times under
/// different move dirs — each move dir is a distinct action.
std::vector<CombatAction> legal_actions(const Agent& self, const World& world);

/// Apply a CombatAction to (agent, world) for `dt`. Drives Agent::applyAction
/// internally for movement + aim + auto-attack, then resolves the ability
/// separately so attack-only and ability-only are both expressible without
/// coupling targets.
void apply(Agent& agent, World& world, const CombatAction& action, float dt);


// ─── Opponent policy ───────────────────────────────────────────────────────
//
// MCTS plans for the hero; other agents in the world are driven by a caller-
// supplied policy during rollouts. For M1 this is a plain std::function so
// scripted bots, hand-coded AIs, or even a nested MCTS instance can slot in.
// One policy covers all non-hero agents the caller wants simulated;
// unregistered agents are left idle (no action).

using OpponentPolicy = std::function<CombatAction(Agent& self, const World& world)>;

/// Always-idle policy (hold position, no combat). Useful as a default.
CombatAction policy_idle(Agent& self, const World& world);

/// Aggressive scripted policy: close on nearest enemy and auto-attack when
/// in range. No abilities. Useful as a punching-bag opponent in tests.
CombatAction policy_aggressive(Agent& self, const World& world);


// ─── Evaluator ─────────────────────────────────────────────────────────────
//
// Scores a world state from the perspective of the hero agent. Called at
// terminal states (one side dead) and at the rollout horizon when no
// terminal was reached. Values are in [-1, 1]: +1 = hero wins decisively,
// -1 = hero dies, 0 = even.
//
// Virtual so callers can plug in heuristics without touching the engine;
// the default HpDeltaEvaluator is the obvious "whose HP is higher."

class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    virtual float evaluate(const World& world, int heroId) const = 0;
};

class HpDeltaEvaluator : public IEvaluator {
public:
    float evaluate(const World& world, int heroId) const override;
};


// ─── Rollout policy ────────────────────────────────────────────────────────
//
// Chooses an action for a single agent during a rollout. Separate from
// OpponentPolicy because rollout policies need to play *both* sides from
// inside MCTS iterations — they don't get to know which agent is "hero."
// World::rng() provides the randomness, so rollouts are reproducible with
// a seeded world snapshot.

class IRolloutPolicy {
public:
    virtual ~IRolloutPolicy() = default;
    virtual CombatAction choose(Agent& self, World& world) const = 0;
};

/// Uniform random over legal_actions. Default rollout policy.
class RandomRollout : public IRolloutPolicy {
public:
    CombatAction choose(Agent& self, World& world) const override;
};


// ─── Tree node (internal, exposed for debug/test) ──────────────────────────

struct Node {
    CombatAction action{};        // action that produced this node from parent
    Node*        parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    std::vector<CombatAction>           untried;  // legal actions not yet expanded

    int   visits      = 0;
    float total_value = 0.0f;     // sum of rollout values, hero's perspective

    float mean() const { return visits > 0 ? total_value / static_cast<float>(visits) : 0.0f; }
    bool  is_leaf() const { return children.empty(); }
    bool  fully_expanded() const { return untried.empty(); }
};


// ─── Engine ────────────────────────────────────────────────────────────────

struct MctsConfig {
    int   iterations = 1000;          // hard cap on iterations (0 = unlimited)
    int   budget_ms  = 0;             // wall-time cap in ms (0 = unlimited).
                                      // Search exits when either cap fires.
                                      // At least one of the two must be > 0.
    int   rollout_horizon = 32;       // cap per rollout (decision windows)
    float sim_dt = 0.016f;            // one sim tick = 16 ms
    int   action_repeat = 4;          // sim ticks per MCTS decision
    float uct_c = 1.41421356f;        // √2 — exploration constant
    uint64_t seed = 0xC0FFEEULL;      // rollout RNG seed (per search() call)
};

struct SearchStats {
    int   iterations = 0;
    int   root_children = 0;
    int   tree_size = 0;
    float best_mean = 0.0f;
    int   best_visits = 0;
    int   elapsed_ms = 0;
    bool  reused_root = false;        // true if tree from prior search was kept
};

class Mcts {
public:
    Mcts() = default;
    explicit Mcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    void set_evaluator(std::shared_ptr<IEvaluator> ev)      { evaluator_ = std::move(ev); }
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p) { rollout_policy_ = std::move(p); }
    void set_opponent_policy(OpponentPolicy p)              { opponent_ = std::move(p); }

    /// Search the best action for `hero` in the current `world`. The search
    /// clones world state via snapshot/restore internally; the caller's world
    /// is not mutated. Returns the action with the highest visit count at
    /// the root (most robust under UCT).
    ///
    /// If a tree from a prior search remains (see advance_root), search
    /// resumes using it — budgets stack across consecutive decisions when
    /// the committed action is tracked via advance_root.
    CombatAction search(World& world, Agent& hero);

    /// After the caller commits an action in the real game, call this to
    /// promote the matching child subtree to the new root. On next search()
    /// the statistics accumulated in that subtree are reused. If no matching
    /// child exists (e.g. budget was too low to expand it), the tree is
    /// discarded and the next search starts fresh.
    void advance_root(const CombatAction& committed);

    /// Drop the search tree. Next search() starts from scratch.
    void reset_tree() { root_.reset(); }

    /// Stats from the most recent search() call. Cleared on next search.
    const SearchStats& last_stats() const { return stats_; }

    /// Root node of the last search tree (owned by the Mcts instance).
    /// Useful for debug visualisation / tests. Null if search() not yet run.
    const Node* last_root() const { return root_.get(); }

private:
    // Internal iteration phases.
    Node* select_(Node* node, World& world, Agent& hero);
    Node* expand_(Node* node, World& world, Agent& hero);
    float rollout_(World& world, Agent& hero);
    static void backprop_(Node* node, float value);

    // Advance the world by one MCTS decision window (action_repeat sim ticks)
    // with `hero_action` driving the hero and opponent_ driving all others.
    void step_decision_(World& world, Agent& hero, const CombatAction& hero_action);

    MctsConfig                    cfg_{};
    std::shared_ptr<IEvaluator>   evaluator_;
    std::shared_ptr<IRolloutPolicy> rollout_policy_;
    OpponentPolicy                opponent_;
    std::unique_ptr<Node>         root_;
    SearchStats                   stats_{};
};

} // namespace brogameagent::mcts
