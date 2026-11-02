#pragma once

#include "mcts.h"
#include <string>

namespace brogameagent::mcts {

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


// ─── TacticPrior ───────────────────────────────────────────────────────────
//
// IPrior that biases a per-hero search toward the CombatAction dictated by
// a committed team tactic (via tactic_to_action). The match action gets a
// large weight; everything else gets 1. Used by LayeredPlanner to couple
// coarse TacticMcts with fine TeamMcts without hard-constraining the fine
// search — it can still deviate when payoff is better.

class TacticPrior : public IPrior {
public:
    void set_tactic(Tactic t) { tactic_ = t; }
    Tactic tactic() const     { return tactic_; }

    void set_match_weight(float w) { match_weight_ = w; }
    void set_other_weight(float w) { other_weight_ = w; }

    std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<CombatAction>& actions) const override;

private:
    Tactic tactic_{};
    float  match_weight_ = 8.0f;
    float  other_weight_ = 1.0f;
};


// ─── Options (temporally-extended actions) ────────────────────────────────
//
// An Option is a policy π with an initiation predicate I and a termination
// predicate β — the standard Sutton/Precup options framework. MCTS over
// options plans at a variable temporal grain: committing an option advances
// the world by many ticks in a single tree edge, so the effective search
// horizon is (tree depth) × (option length) rather than (tree depth) × (one
// decision window). This is the right tool for maneuvers that take multiple
// ticks to pay off — "peek and shoot", "retreat to cover", "flank left" —
// where plain CombatAction-level search wastes its budget on tick-level
// differences that don't matter.
//
// Branching collapses from ~18 (move×attack×ability) per node to the size of
// the option set (typically 4–8), so priors become optional and iteration
// budgets can be much smaller. Pair with use_leaf_value + a heuristic
// evaluator for real-time MOBA/shooter AI.
//
// Options are caller-authored: write a subclass, or — from JS — build one
// with bro.ai.game.createOption({ canInitiate, step, shouldTerminate }).

class Option {
public:
    virtual ~Option() = default;

    /// Stable human-readable name. Used as the key in advance_root matching
    /// and in debug output. Must be unique within an OptionMcts's option set.
    virtual const std::string& name() const = 0;

    /// True iff this option can be started from the current (self, world)
    /// state. OptionMcts filters its untried list through this predicate
    /// every expansion, so it's fine (expected) for can_initiate to depend
    /// on dynamic state like HP, cooldowns, or visible enemies.
    virtual bool can_initiate(const Agent& self, const World& world) const = 0;

    /// Produce the CombatAction to apply this tick of option execution.
    /// `ticks_in_option` starts at 0 when the option is committed and
    /// increments per decision window (not per sim tick). Options are free
    /// to read/write ephemeral state on `self` — but see should_terminate()
    /// for how to signal completion without stateful callers.
    virtual CombatAction step(Agent& self, World& world, int ticks_in_option) const = 0;

    /// True iff the option should stop running and the next option (or
    /// rollout step / search level) should take over. Called after each
    /// window; MCTS also terminates an option unconditionally once
    /// cfg.option_max_windows is reached.
    virtual bool should_terminate(const Agent& self, const World& world,
                                   int ticks_in_option) const = 0;
};

/// Team-scoped companion to Option. A TeamOption governs all heroes on the
/// controlled team simultaneously — use this for coordinated plays (focus
/// fire, formation moves, synchronised retreat) where per-hero options would
/// diverge. Single-hero Options are the right choice for per-role planning
/// inside a Commander.
class TeamOption {
public:
    virtual ~TeamOption() = default;
    virtual const std::string& name() const = 0;
    virtual bool can_initiate(const std::vector<Agent*>& heroes,
                              const World& world) const = 0;
    /// Return exactly heroes.size() actions, in the same order. Dead heroes
    /// should receive a no-op action; caller does not skip them.
    virtual std::vector<CombatAction> step(const std::vector<Agent*>& heroes,
                                            World& world,
                                            int ticks_in_option) const = 0;
    virtual bool should_terminate(const std::vector<Agent*>& heroes,
                                   const World& world,
                                   int ticks_in_option) const = 0;
};


// ─── OptionMcts (single-hero option search) ────────────────────────────────
//
// Tree search over a caller-supplied option set, driving one hero agent.
// Other agents in the world are driven by OpponentPolicy during in-tree
// execution and rollouts (same contract as Mcts). Each tree edge represents
// "commit option X until it terminates or cap" and advances the world by
// potentially many windows — so depth-3 tree with options that average 4
// windows gives a 12-window (≈ 3 s) planning horizon for the iteration cost
// of a 3-deep search.

class OptionMcts {
public:
    struct ONode {
        const Option* action = nullptr;   // option committed at this node's in-edge; null at root
        ONode*        parent = nullptr;
        std::vector<std::unique_ptr<ONode>> children;
        std::vector<const Option*> untried;
        int   visits = 0;
        float total_value = 0.0f;

        float mean() const { return visits > 0 ? total_value / static_cast<float>(visits) : 0.0f; }
        bool  is_leaf() const { return children.empty(); }
        bool  fully_expanded() const { return untried.empty(); }
    };

    OptionMcts() = default;
    explicit OptionMcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    /// Register the options available to the search. OptionMcts holds these
    /// via shared_ptr so rollout / commander / JS wrappers can share the
    /// same objects. Must be called before search(). Safe to replace between
    /// searches (resets the tree).
    void set_options(std::vector<std::shared_ptr<Option>> options) {
        options_ = std::move(options);
        root_.reset();
    }
    const std::vector<std::shared_ptr<Option>>& options() const { return options_; }

    void set_evaluator(std::shared_ptr<IEvaluator> ev)         { evaluator_ = std::move(ev); }
    void set_opponent_policy(OpponentPolicy p)                 { opponent_ = std::move(p); }

    /// Search for the best option to commit from the current state. Returns
    /// the Option pointer (stable across calls — owned by options_), or
    /// nullptr if no option can_initiate here (caller should fall back).
    const Option* search(World& world, Agent& hero);

    /// Promote the subtree matching the committed option by name. Unknown or
    /// un-expanded options reset the tree.
    void advance_root(const Option* committed);

    void reset_tree() { root_.reset(); }

    const SearchStats& last_stats() const { return stats_; }
    const ONode* last_root() const { return root_.get(); }

    /// Execute an option against the live world — advance state by windows
    /// until option.should_terminate() returns true, terminal is reached,
    /// or cfg.option_max_windows is hit. Returns windows executed. Useful
    /// when committing an option outside of search().
    int execute_option(World& world, Agent& hero, const Option& option);

private:
    ONode* select_(ONode* node, World& world, Agent& hero);
    ONode* expand_(ONode* node, World& world, Agent& hero);
    float  rollout_(World& world, Agent& hero);
    std::vector<const Option*> legal_options_(const Agent& hero, const World& world) const;
    void   step_option_(World& world, Agent& hero, const Option& option);

    MctsConfig                      cfg_{};
    std::vector<std::shared_ptr<Option>> options_;
    std::shared_ptr<IEvaluator>     evaluator_;
    OpponentPolicy                  opponent_;
    std::unique_ptr<ONode>          root_;
    SearchStats                     stats_{};
};


// ─── TeamOptionMcts (team-scoped option search) ───────────────────────────

class TeamOptionMcts {
public:
    struct TNode {
        const TeamOption* action = nullptr;
        TNode*            parent = nullptr;
        std::vector<std::unique_ptr<TNode>> children;
        std::vector<const TeamOption*> untried;
        int   visits = 0;
        float total_value = 0.0f;

        float mean() const { return visits > 0 ? total_value / static_cast<float>(visits) : 0.0f; }
        bool  is_leaf() const { return children.empty(); }
        bool  fully_expanded() const { return untried.empty(); }
    };

    TeamOptionMcts() = default;
    explicit TeamOptionMcts(MctsConfig cfg) : cfg_(cfg) {}

    void set_config(const MctsConfig& cfg) { cfg_ = cfg; }
    const MctsConfig& config() const { return cfg_; }

    void set_options(std::vector<std::shared_ptr<TeamOption>> options) {
        options_ = std::move(options);
        root_.reset();
    }
    const std::vector<std::shared_ptr<TeamOption>>& options() const { return options_; }

    void set_evaluator(std::shared_ptr<ITeamEvaluator> ev)      { evaluator_ = std::move(ev); }
    void set_opponent_policy(OpponentPolicy p)                   { opponent_ = std::move(p); }

    const TeamOption* search(World& world, const std::vector<Agent*>& heroes);
    void advance_root(const TeamOption* committed);
    void reset_tree() { root_.reset(); }

    const SearchStats& last_stats() const { return stats_; }
    const TNode* last_root() const { return root_.get(); }

    int execute_option(World& world, const std::vector<Agent*>& heroes,
                       const TeamOption& option);

private:
    TNode* select_(TNode* node, World& world, const std::vector<Agent*>& heroes);
    TNode* expand_(TNode* node, World& world, const std::vector<Agent*>& heroes);
    float  rollout_(World& world, const std::vector<Agent*>& heroes, int team_id);
    std::vector<const TeamOption*> legal_options_(const std::vector<Agent*>& heroes,
                                                    const World& world) const;
    void   step_option_(World& world, const std::vector<Agent*>& heroes,
                         const TeamOption& option);

    MctsConfig                          cfg_{};
    std::vector<std::shared_ptr<TeamOption>> options_;
    std::shared_ptr<ITeamEvaluator>     evaluator_;
    OpponentPolicy                      opponent_;
    std::unique_ptr<TNode>              root_;
    SearchStats                         stats_{};
};


// ─── LayeredPlanner ────────────────────────────────────────────────────────
//
// Hierarchical composition of TacticMcts (coarse) over TeamMcts (fine). The
// tactic is re-planned every `tactic_cfg.tactic_window_decisions` fine
// decisions; in between, fine per-hero search runs every decide() call,
// priored toward the currently committed tactic via TacticPrior. Evaluator,
// rollout policy, and opponent policy are shared across both layers.
//
// Intended usage:
//   LayeredPlanner planner({tactic_cfg, fine_cfg});
//   planner.set_team_evaluator(std::make_shared<TeamHpDeltaEvaluator>());
//   planner.set_rollout_policy(std::make_shared<AggressiveRollout>());
//   planner.set_opponent_policy(policy_aggressive);
//   while (game_running) {
//       auto joint = planner.decide(world, heroes);
//       commit(joint);            // apply each hero action externally
//   }

class LayeredPlanner {
public:
    struct Config {
        MctsConfig tactic_cfg{};   // config for the coarse TacticMcts
        MctsConfig fine_cfg{};     // config for the fine TeamMcts
        // TacticPrior weights used to bias the fine per-hero search toward
        // the committed tactic's action. Higher match/other ratio → the
        // fine search barely deviates from tactic_to_action; lower ratio →
        // the fine search can diverge per-hero. Default 8/1 preserves the
        // original behavior.
        float tactic_match_weight = 8.0f;
        float tactic_other_weight = 1.0f;
    };

    struct Stats {
        Tactic      committed_tactic{};
        int         windows_until_replan = 0;
        bool        replanned_this_call  = false;
        SearchStats tactic_stats{};    // from the last tactic search (stale
                                        // when replanned_this_call is false)
        SearchStats fine_stats{};
    };

    LayeredPlanner() = default;
    explicit LayeredPlanner(Config cfg) : cfg_(cfg) {}

    void set_config(const Config& cfg) { cfg_ = cfg; }
    const Config& config() const       { return cfg_; }

    void set_team_evaluator(std::shared_ptr<ITeamEvaluator> ev) { evaluator_ = std::move(ev); }
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p)  { rollout_policy_ = std::move(p); }
    void set_opponent_policy(OpponentPolicy p)                   { opponent_ = std::move(p); }

    /// Forget the committed tactic and both search trees. Call between
    /// episodes so the next decide() starts fresh.
    void reset();

    /// Produce per-hero actions for the current frame. Runs TacticMcts only
    /// when the current tactic has expired; always runs TeamMcts.
    TeamMcts::JointAction decide(World& world, const std::vector<Agent*>& heroes);

    const Stats& last_stats() const     { return stats_; }
    Tactic committed_tactic() const     { return committed_tactic_; }
    int    windows_until_replan() const { return windows_left_; }

private:
    Config                            cfg_{};
    TacticMcts                        tactic_mcts_;
    TeamMcts                          fine_mcts_;
    std::shared_ptr<TacticPrior>      prior_;
    std::shared_ptr<ITeamEvaluator>   evaluator_;
    std::shared_ptr<IRolloutPolicy>   rollout_policy_;
    OpponentPolicy                    opponent_;

    Tactic committed_tactic_{};
    int    windows_left_ = 0;
    Stats  stats_{};
};


// ─── Commander (role-based hierarchical planner) ──────────────────────────
//
// Assigns each hero to one of N named roles, then runs a per-hero OptionMcts
// using that role's option set and evaluator. Role assignment runs at a
// coarser cadence than per-hero execution: re-assigned every
// `replan_every_windows` decide() calls, or immediately when the team
// composition changes.
//
// Why this shape, not TeamMcts? Joint team search over N heroes × M actions
// has N^M branching — combinatorial collapse at realtime budgets. Commander
// decomposes: a cheap role-assignment step at low frequency picks who is
// doing what job, then per-hero single-agent search plans *within* a role's
// smaller option space. Each hero only reasons about its own objectives
// and its own option set; coordination comes from role specialization, not
// joint search.
//
// Typical roles for a MOBA/shooter team:
//   - "lead"    : aggressive options — push, commit, peek-shoot
//   - "flank"   : flanking/reposition options
//   - "support" : follow-ally, heal, retreat-to-cover
// A Role can also specialise its evaluator so e.g. a support's value
// function weights ally HP, not damage dealt.
//
// Options inside a role commit for their natural lifetime (governed by
// should_terminate + option_max_windows). Commander only invokes search
// when a hero's current option terminates, not every frame — cheap at
// runtime even with a heavy MCTS budget.

class Commander {
public:
    struct Role {
        std::string name;
        std::vector<std::shared_ptr<Option>> options;   // option set for this role
        std::shared_ptr<IEvaluator> evaluator;          // optional; falls back to default_evaluator
    };

    struct Config {
        MctsConfig role_cfg{};
        int replan_every_windows = 4;   // decide() calls between role re-assignments
    };

    /// Caller-supplied role-assignment function. Returns a vector of role
    /// indices (same length as heroes). Index -1 leaves the hero idle this
    /// period. If null, Commander round-robins heroes across roles.
    using AssignFn = std::function<std::vector<int>(
        const std::vector<Agent*>& heroes, const World& world)>;

    Commander() = default;
    explicit Commander(Config cfg) : cfg_(cfg) {}

    void set_config(const Config& cfg) { cfg_ = cfg; }
    const Config& config() const { return cfg_; }

    /// Register a role. Order matters — the assignment function returns
    /// indices into this registered list.
    void add_role(std::string name,
                   std::vector<std::shared_ptr<Option>> options,
                   std::shared_ptr<IEvaluator> evaluator = nullptr) {
        Role r;
        r.name = std::move(name);
        r.options = std::move(options);
        r.evaluator = std::move(evaluator);
        roles_.push_back(std::move(r));
        reset();
    }
    const std::vector<Role>& roles() const { return roles_; }

    void set_assigner(AssignFn fn)                              { assigner_ = std::move(fn); reset(); }
    void set_opponent_policy(OpponentPolicy p)                   { opponent_ = std::move(p); }
    void set_default_evaluator(std::shared_ptr<IEvaluator> ev)   { default_eval_ = std::move(ev); }

    /// Forget per-hero option state and engines. The next decide() call
    /// will re-assign roles and re-search options for everyone.
    void reset();

    /// Per-hero actions for the current frame. Returns heroes.size() actions
    /// (dead or role=-1 heroes get a no-op). decide() only runs a per-hero
    /// OptionMcts::search when that hero's current option has terminated or
    /// their role reassignment reset it — cheap in the common case.
    std::vector<CombatAction> decide(World& world, const std::vector<Agent*>& heroes);

    /// Current role-index assignment. Index i corresponds to heroes[i] at
    /// the last decide() call. Empty before first decide().
    const std::vector<int>& current_assignments() const { return assignments_; }

    /// Name of the option currently committed by the given hero (or empty
    /// string if none). Useful for debug overlays.
    std::string committed_option_for_hero(size_t hero_idx) const;

    int windows_until_replan() const { return windows_left_; }

private:
    struct HeroState {
        const Option* current_option = nullptr;
        int           ticks_in_option = 0;
        std::unique_ptr<OptionMcts> engine;
    };

    Config                            cfg_{};
    std::vector<Role>                 roles_;
    AssignFn                          assigner_;
    OpponentPolicy                    opponent_;
    std::shared_ptr<IEvaluator>       default_eval_;

    std::vector<int>                  assignments_;
    std::vector<HeroState>            hero_states_;
    int                               windows_left_ = 0;
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
