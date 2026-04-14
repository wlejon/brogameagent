#pragma once

#include "action_mask.h"
#include "agent.h"
#include "world.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
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


// ─── Team evaluator ────────────────────────────────────────────────────────
//
// Team-scoped companion to IEvaluator. Used by TeamMcts: the value at the
// root is a single scalar representing "how well did this team do", not
// per-hero credit assignment. That matches coop-MCTS literature — individual
// heroes share a reward so the search optimises coordinated play.

class ITeamEvaluator {
public:
    virtual ~ITeamEvaluator() = default;
    virtual float evaluate(const World& world, int team_id) const = 0;
};

class TeamHpDeltaEvaluator : public ITeamEvaluator {
public:
    float evaluate(const World& world, int team_id) const override;
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

/// Scripted "close and auto-attack the nearest enemy" rollout. Wraps
/// policy_aggressive. Typically produces materially stronger search than
/// RandomRollout under equal iteration budgets, at the cost of bias toward
/// melee play — prefer it when the caller's heroes are expected to engage.
class AggressiveRollout : public IRolloutPolicy {
public:
    CombatAction choose(Agent& self, World& world) const override;
};


// ─── Action priors (PUCT) ──────────────────────────────────────────────────
//
// A prior assigns non-negative weights to each legal action of a node. With
// MctsConfig::prior_c > 0, Mcts selection uses PUCT instead of plain UCT:
//
//     score(child) = Q(child) + prior_c * P(child) * √N(parent) / (1 + n(child))
//
// where P(child) is the child's normalized prior probability. Good priors
// compress the search into promising subtrees and routinely double effective
// strength per iteration. Weights are normalized by Mcts internally — callers
// just return relative weights; zero weights are allowed.

class IPrior {
public:
    virtual ~IPrior() = default;
    /// Return a weight for each action in `actions`. Output size must equal
    /// `actions.size()`. All weights must be >= 0; at least one should be
    /// positive (if all are zero, Mcts falls back to a uniform prior).
    virtual std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<CombatAction>& actions) const = 0;
};

/// Uniform prior — every action gets the same weight. Equivalent to plain
/// UCT when prior_c > 0 (selection then reduces to Q + c * √N / (1 + n)).
class UniformPrior : public IPrior {
public:
    std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<CombatAction>& actions) const override;
};

/// Weights attack actions > ability actions > pure-move actions. Cheap
/// heuristic: if action.attack_slot >= 0 → weight 4, else if
/// action.ability_slot >= 0 → weight 2, else weight 1. Encourages the tree
/// to prefer engagement actions early, which then win on Q once they pay off.
class AttackBiasPrior : public IPrior {
public:
    std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<CombatAction>& actions) const override;
};


// ─── Tree node (internal, exposed for debug/test) ──────────────────────────

struct Node {
    CombatAction action{};        // action that produced this node from parent
    Node*        parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    std::vector<CombatAction>           untried;        // legal actions not yet expanded
    std::vector<float>                  untried_priors; // aligned with `untried`; normalized

    int   visits      = 0;
    float total_value = 0.0f;     // sum of rollout values, hero's perspective
    float prior_p     = 1.0f;     // prior probability of taking this child's action at parent

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

    // TacticMcts only: how many decision windows a committed tactic runs for
    // before the planner re-chooses. A window of 4 × action_repeat(4) × 16ms
    // ≈ 256 ms of game time per tactic — coarse enough to matter, fine
    // enough to respond to new threats.
    int   tactic_window_decisions = 4;

    // Progressive widening (Mcts only). When > 0, a node expands a new child
    // only while children.size() < ceil(visits^pw_alpha); otherwise selection
    // descends into the existing children via UCT. Keeps the tree from going
    // wide-and-shallow over large action sets (9 move dirs × N enemy slots ×
    // ability slots). 0 (default) disables PW — every untried action is
    // expanded before UCT takes over, matching classical MCTS.
    // Typical values: 0.5 (aggressive depth) to 0.8 (near-classical).
    float pw_alpha = 0.0f;

    // PUCT exploration weight (Mcts only). 0 (default) = plain UCT using
    // uct_c. When > 0, Mcts selection uses
    //     score = Q + prior_c * P * √N_parent / (1 + n_child)
    // and the uct_c term is ignored. Typical values: 1.0–3.0. Pair with
    // set_prior() — without a set prior, the prior is uniform and PUCT
    // reduces to an exploration-constant-rescaled UCT variant.
    float prior_c = 0.0f;
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
    void set_prior(std::shared_ptr<IPrior> p)               { prior_ = std::move(p); }

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
    std::shared_ptr<IPrior>       prior_;
    OpponentPolicy                opponent_;
    std::unique_ptr<Node>         root_;
    SearchStats                   stats_{};
};


// ─── Decoupled MCTS (simultaneous-move 1v1) ────────────────────────────────
//
// Both players are searched simultaneously using Decoupled UCT. At every
// tree node, per-player action stats are tracked independently; selection
// picks one action per player via UCT (hero maximises, opponent minimises
// hero value), and child nodes are keyed by the (hero_idx, opp_idx) pair.
//
// This is the right engine for real-time 1v1 combat where neither side
// observes the other's choice before acting. For turn-based games where
// one side moves at a time, use the single-player Mcts with an opponent
// policy callback.

class DecoupledMcts {
public:
    struct Joint {
        CombatAction hero;
        CombatAction opp;
        bool operator==(const Joint&) const = default;
    };

    struct PlayerStats {
        std::vector<CombatAction> actions;
        std::vector<int>          visits;
        std::vector<float>        total_value;   // always hero's perspective
        std::vector<float>        priors;         // normalized; empty ⇒ uniform
    };

    struct DNode {
        PlayerStats hero_stats;
        PlayerStats opp_stats;
        // Sparse children: key = pack(hero_idx, opp_idx); only visited pairs.
        std::unordered_map<uint32_t, std::unique_ptr<DNode>> children;
        int   visits = 0;
        DNode* parent = nullptr;
        int   parent_hero_idx = -1;
        int   parent_opp_idx  = -1;
    };

    DecoupledMcts() = default;
    explicit DecoupledMcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    void set_evaluator(std::shared_ptr<IEvaluator> ev)         { evaluator_ = std::move(ev); }
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p) { rollout_policy_ = std::move(p); }
    void set_prior(std::shared_ptr<IPrior> p)                  { prior_ = std::move(p); }

    /// Search for the best joint action under simultaneous-move assumptions.
    /// The returned hero action is what the caller should commit; the opp
    /// action is the planner's best-response prediction (useful for debug).
    Joint search(World& world, Agent& hero, Agent& opp);

    /// Promote the child matching both observed actions. If the pair wasn't
    /// expanded during search (small budget), resets the tree.
    void advance_root(const CombatAction& hero_committed,
                       const CombatAction& opp_committed);

    void reset_tree() { root_.reset(); }

    const SearchStats& last_stats() const { return stats_; }
    const DNode* last_root() const { return root_.get(); }

private:
    static uint32_t pack_key_(int h, int o) {
        return (static_cast<uint32_t>(h) << 16)
             | (static_cast<uint32_t>(o) & 0xFFFFu);
    }

    PlayerStats build_stats_(const Agent& self, const World& world) const;

    int  pick_action_idx_(const PlayerStats& stats, int node_visits,
                          bool minimize) const;

    void step_joint_(World& world,
                     Agent& hero, const CombatAction& hero_act,
                     Agent& opp,  const CombatAction& opp_act);

    float rollout_(World& world, Agent& hero, Agent& opp);

    MctsConfig                      cfg_{};
    std::shared_ptr<IEvaluator>     evaluator_;
    std::shared_ptr<IRolloutPolicy> rollout_policy_;
    std::shared_ptr<IPrior>         prior_;
    std::unique_ptr<DNode>          root_;
    SearchStats                     stats_{};
};


// ─── TeamMcts (cooperative multi-agent) ────────────────────────────────────
//
// Plans joint actions for a team of N heroes cooperating against scripted
// opponents. Uses multi-agent Decoupled UCT: at every node, each hero
// maintains independent action stats; selection picks one action per hero
// (all maximising the shared team value); joint actions index sparse
// children. Enemies (and any non-team agents) are driven by OpponentPolicy
// during expansion and rollout.
//
// Target scenario: MOBA-style "intelligent heroes vs heavy PvE pressure".
// Use DecoupledMcts instead when two sides are simultaneously planning
// against each other (1v1 or symmetric PvP).

class TeamMcts {
public:
    struct JointAction {
        std::vector<CombatAction> per_hero;
        bool operator==(const JointAction&) const = default;
    };

    struct PlayerStats {
        std::vector<CombatAction> actions;
        std::vector<int>          visits;
        std::vector<float>        total_value;
        std::vector<float>        priors;            // normalized; empty ⇒ uniform
    };

    struct TNode {
        std::vector<PlayerStats> per_hero;          // size = heroes.size()
        std::map<std::vector<int>, std::unique_ptr<TNode>> children;
        int   visits = 0;
        TNode* parent = nullptr;
        std::vector<int> parent_action_idx;         // size = heroes.size()
    };

    TeamMcts() = default;
    explicit TeamMcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    void set_evaluator(std::shared_ptr<ITeamEvaluator> ev)      { evaluator_ = std::move(ev); }
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p)  { rollout_policy_ = std::move(p); }
    void set_opponent_policy(OpponentPolicy p)                  { opponent_ = std::move(p); }
    void set_prior(std::shared_ptr<IPrior> p)                   { prior_ = std::move(p); }

    /// Search the best joint action for the team. All heroes are treated as
    /// cooperating (shared value). The returned JointAction has one entry
    /// per input hero, in the same order as `heroes`.
    JointAction search(World& world, const std::vector<Agent*>& heroes);

    /// Promote the subtree corresponding to the committed joint action.
    /// Size of committed.per_hero must match the team at the last search.
    void advance_root(const JointAction& committed);

    void reset_tree() { root_.reset(); }

    const SearchStats& last_stats() const { return stats_; }
    const TNode* last_root() const { return root_.get(); }

private:
    PlayerStats build_stats_(const Agent& self, const World& world) const;

    int  pick_action_idx_(const PlayerStats& stats, int node_visits) const;

    void step_joint_(World& world,
                     const std::vector<Agent*>& heroes,
                     const std::vector<CombatAction>& hero_actions);

    float rollout_(World& world, const std::vector<Agent*>& heroes, int team_id);

    MctsConfig                       cfg_{};
    std::shared_ptr<ITeamEvaluator>  evaluator_;
    std::shared_ptr<IRolloutPolicy>  rollout_policy_;
    std::shared_ptr<IPrior>          prior_;
    OpponentPolicy                   opponent_;
    std::unique_ptr<TNode>           root_;
    SearchStats                      stats_{};
};


// ─── Hierarchical tactic layer ─────────────────────────────────────────────
//
// Above the raw-action layer, TacticMcts plans at the granularity of named
// team tactics — "hold", "focus-fire the weakest enemy", "scatter and
// attack nearest", "retreat". This is how MOBA/RTS AIs typically organise:
// a small discrete set of coordinated plays at a coarse cadence, with
// per-unit execution handled by a scripted interpreter of the chosen play.
//
// Branching: ~COUNT tactics per node, vs. the flat team's 18^N joint space.
// That makes tactic search *much* shallower/wider, so it's the right tool
// for long-horizon coordination (several seconds into the future).
//
// TeamMcts and TacticMcts compose: a real game AI can run tactic planning
// at 4 Hz and fine-grained per-hero planning at 16 Hz, layered.

enum class TacticKind : int {
    Hold          = 0,   // stay put; auto-attack anything in range
    FocusLowestHp = 1,   // all heroes pile on the weakest enemy
    Scatter       = 2,   // each hero attacks their nearest enemy
    Retreat       = 3,   // move away from nearest enemy; no attacks
    COUNT
};

struct Tactic {
    TacticKind kind = TacticKind::Hold;
    bool operator==(const Tactic&) const = default;
};

/// Convert a team tactic into a concrete per-hero CombatAction for the
/// current world state. Stateless — same (tactic, hero, world) always
/// produces the same action. Heroes not alive get a no-op.
CombatAction tactic_to_action(const Tactic& t, const Agent& hero, const World& world);

/// Tactics available for the current team + world. Pruned contextually:
///   - Always includes Hold.
///   - FocusLowestHp and Scatter only when at least one living enemy exists.
///   - Retreat only when at least one living enemy is within ~1.5× its own
///     attack range of some living hero (i.e. there is something to flee
///     from). Pruning away Retreat in safe states keeps TacticMcts from
///     wasting budget exploring obviously-dominated plays.
std::vector<Tactic> legal_tactics(const std::vector<Agent*>& heroes, const World& world);

class TacticMcts {
public:
    struct TNode {
        Tactic  action{};            // tactic that produced this node from parent
        TNode*  parent = nullptr;
        std::vector<std::unique_ptr<TNode>> children;
        std::vector<Tactic> untried;
        int     visits = 0;
        float   total_value = 0.0f;

        float mean() const { return visits > 0 ? total_value / static_cast<float>(visits) : 0.0f; }
        bool  is_leaf() const { return children.empty(); }
        bool  fully_expanded() const { return untried.empty(); }
    };

    TacticMcts() = default;
    explicit TacticMcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    void set_evaluator(std::shared_ptr<ITeamEvaluator> ev)      { evaluator_ = std::move(ev); }
    void set_opponent_policy(OpponentPolicy p)                  { opponent_ = std::move(p); }

    /// Search for the best tactic for the team. The returned Tactic is the
    /// most-visited root child; caller commits it by executing
    /// tactic_to_action per hero for cfg.tactic_window_decisions windows.
    Tactic search(World& world, const std::vector<Agent*>& heroes);

    void advance_root(const Tactic& committed);
    void reset_tree() { root_.reset(); }

    const SearchStats& last_stats() const { return stats_; }
    const TNode* last_root() const { return root_.get(); }

private:
    TNode* select_(TNode* node, World& world, const std::vector<Agent*>& heroes);
    TNode* expand_(TNode* node, World& world, const std::vector<Agent*>& heroes);
    float  rollout_(World& world, const std::vector<Agent*>& heroes, int team_id);

    // Execute `tactic` for cfg_.tactic_window_decisions windows. Each window
    // applies tactic_to_action per hero and opponent_ per non-hero, repeated
    // cfg_.action_repeat sim ticks per window. Early-exits on terminal.
    void step_tactic_(World& world, const std::vector<Agent*>& heroes,
                       const Tactic& tactic);

    MctsConfig                       cfg_{};
    std::shared_ptr<ITeamEvaluator>  evaluator_;
    OpponentPolicy                   opponent_;
    std::unique_ptr<TNode>           root_;
    SearchStats                      stats_{};
};


// ─── Root-parallel search ──────────────────────────────────────────────────
//
// Run N independent MCTS trees in N threads, then merge root visit counts
// across trees to pick the final action. The caller owns N Worlds, each
// pre-seeded to the same game state; each thread gets one World to mutate
// during its search (snapshot/restore internally). Per-thread engines get
// seed = cfg.seed + thread_idx so their rollouts diverge.
//
// Unlike sequential "parallel" MCTS (which is just more iterations on one
// tree), this gives real speedup on multi-core machines at the cost of
// some noise in the merged distribution vs. one huge tree.

struct ParallelSearchStats {
    int  num_threads    = 0;
    int  total_iterations = 0;
    int  elapsed_ms     = 0;
    int  merged_best_visits = 0;
};

/// Single-player root-parallel search. Each thread's opponent is driven by
/// `opponent_policy`. Each world must have an agent with id == hero_id.
CombatAction root_parallel_search(
    const std::vector<World*>& worlds,
    int hero_id,
    const MctsConfig& cfg,
    std::shared_ptr<IEvaluator>     evaluator,
    std::shared_ptr<IRolloutPolicy> rollout_policy,
    OpponentPolicy                  opponent_policy,
    ParallelSearchStats*            out_stats = nullptr);

/// Decoupled root-parallel search. Each world must contain both hero_id and
/// opp_id; the returned joint merges each player's per-root visit counts.
DecoupledMcts::Joint root_parallel_search_decoupled(
    const std::vector<World*>& worlds,
    int hero_id, int opp_id,
    const MctsConfig& cfg,
    std::shared_ptr<IEvaluator>     evaluator,
    std::shared_ptr<IRolloutPolicy> rollout_policy,
    ParallelSearchStats*            out_stats = nullptr);

} // namespace brogameagent::mcts
