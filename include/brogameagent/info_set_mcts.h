#pragma once

// This header is included from mcts.h; including it directly also works since
// it re-includes its own dependencies. Kept separate so the partial-
// observability layer is visibly distinct from the standard MCTS engines.

#include "belief.h"
#include "observability.h"
#include "snapshot.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace brogameagent {

class World;
class Agent;

namespace mcts {

class Mcts;
class IEvaluator;
class IRolloutPolicy;
class IPrior;
struct MctsConfig;
struct SearchStats;
struct CombatAction;

// ─── InfoSetMcts ───────────────────────────────────────────────────────────
//
// Determinized Information-Set MCTS for single-hero planning under partial
// observability. Each iteration samples one particle from the supplied
// belief::TeamBelief, patches the world into that determinized state, and
// runs one standard MCTS iteration against it. The tree is shared across
// determinizations (classic IS-MCTS). Root children are initialised against
// the belief-mean determinization for slot-order stability.
//
// Composes with the existing Mcts engine: InfoSetMcts owns an inner Mcts,
// forwards evaluator/rollout/prior/opponent plumbing, and only adds the
// belief layer and the per-iteration world-patching loop.

struct InfoSetSearchStats {
    int   iterations = 0;
    int   root_children = 0;
    int   tree_size = 0;
    float best_mean = 0.0f;
    int   best_visits = 0;
    int   elapsed_ms = 0;
    bool  reused_root = false;
    float mean_ess = 0.0f;        // effective sample size at search time
};

class InfoSetMcts {
public:
    InfoSetMcts();
    explicit InfoSetMcts(MctsConfig cfg);

    void set_config(const MctsConfig& cfg);
    const MctsConfig& config() const;

    void set_belief(std::shared_ptr<belief::TeamBelief> b)     { belief_ = std::move(b); }
    std::shared_ptr<belief::TeamBelief> belief() const         { return belief_; }

    void set_evaluator(std::shared_ptr<IEvaluator> ev);
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p);
    void set_opponent_policy(std::function<CombatAction(Agent&, const World&)> p);
    void set_prior(std::shared_ptr<IPrior> p);

    /// Search for the best action for `hero` under the current belief.
    /// `truth` is restored exactly on return — identical to Mcts::search.
    CombatAction search(World& truth, Agent& hero);

    /// Forward advance_root to the inner engine.
    void advance_root(const CombatAction& committed);

    /// Drop the tree.
    void reset_tree();

    const InfoSetSearchStats& last_stats() const { return stats_; }

    /// Access to the inner Mcts engine for debug / test inspection. Owned
    /// by InfoSetMcts; do not store beyond the lifetime of this object.
    Mcts& inner();
    const Mcts& inner() const;

private:
    std::shared_ptr<belief::TeamBelief> belief_;
    std::unique_ptr<Mcts>               inner_;
    InfoSetSearchStats                  stats_{};
};


// ─── InfoSetTeamMcts ───────────────────────────────────────────────────────
//
// Multi-hero cooperative analogue — same determinization pattern, but wraps
// TeamMcts instead of Mcts. The belief is team-scoped; sampled particles
// apply to all tracked enemies.

class TeamMcts;
class ITeamEvaluator;

struct InfoSetTeamStats {
    int   iterations = 0;
    int   tree_size = 0;
    float best_mean = 0.0f;
    int   best_visits = 0;
    int   elapsed_ms = 0;
    float mean_ess = 0.0f;
};

class InfoSetTeamMcts {
public:
    InfoSetTeamMcts();
    explicit InfoSetTeamMcts(MctsConfig cfg);

    void set_config(const MctsConfig& cfg);
    const MctsConfig& config() const;

    void set_belief(std::shared_ptr<belief::TeamBelief> b) { belief_ = std::move(b); }
    std::shared_ptr<belief::TeamBelief> belief() const     { return belief_; }

    void set_evaluator(std::shared_ptr<ITeamEvaluator> ev);
    void set_rollout_policy(std::shared_ptr<IRolloutPolicy> p);
    void set_opponent_policy(std::function<CombatAction(Agent&, const World&)> p);
    void set_prior(std::shared_ptr<IPrior> p);

    /// Search a joint action for the team. Caller's world is restored.
    struct JointOut { std::vector<CombatAction> per_hero; };
    JointOut search(World& truth, const std::vector<Agent*>& heroes);

    void reset_tree();
    const InfoSetTeamStats& last_stats() const { return stats_; }

    TeamMcts& inner();
    const TeamMcts& inner() const;

private:
    std::shared_ptr<belief::TeamBelief> belief_;
    std::unique_ptr<TeamMcts>           inner_;
    InfoSetTeamStats                    stats_{};
};

// ─── Determinization helper ────────────────────────────────────────────────
//
// Patch a WorldSnapshot in place so the listed hidden-enemy ids reflect the
// sampled particle states. Agents whose ids do not appear in `sampled` are
// left untouched. Exposed as a free helper so callers can build custom
// IS-MCTS variants without depending on the InfoSetMcts wrapper.

} // namespace mcts

namespace belief {
struct EnemyParticle;  // fwd, defined in belief.h
}

namespace mcts {
/// Apply a sampled particle map onto `snap` (produced by World::snapshot).
/// For each (enemy_id, particle), overwrites position, velocity, yaw, and hp
/// in the matching AgentSnapshot. Units retain their other state (cooldowns,
/// mana, buffs) from the caller's truth — a conservative default that makes
/// rollouts forgiving when a hidden enemy's cooldown is not part of the
/// belief.
void patch_snapshot_with_particles(
    WorldSnapshot& snap,
    const std::unordered_map<int, belief::EnemyParticle>& sampled);

} // namespace mcts
} // namespace brogameagent
