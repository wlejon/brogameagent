#include "brogameagent/info_set_mcts.h"

#include "brogameagent/agent.h"
#include "brogameagent/belief.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>

namespace brogameagent::mcts {

// ─── Determinization ───────────────────────────────────────────────────────

void patch_snapshot_with_particles(
    WorldSnapshot& snap,
    const std::unordered_map<int, belief::EnemyParticle>& sampled) {
    for (auto& as : snap.agents) {
        auto it = sampled.find(as.id);
        if (it == sampled.end()) continue;
        const belief::EnemyParticle& p = it->second;
        as.x  = p.pos.x;
        as.z  = p.pos.z;
        as.vx = p.vel.x;
        as.vz = p.vel.z;
        as.yaw = p.heading;
        // HP zero means the belief is convinced the enemy is dead; clamp to
        // [0, maxHp] to avoid negatives slipping into combat math.
        float hp = p.hp;
        if (hp < 0) hp = 0;
        if (as.unit.maxHp > 0 && hp > as.unit.maxHp) hp = as.unit.maxHp;
        as.unit.hp = hp;
    }
}

// ─── InfoSetMcts ───────────────────────────────────────────────────────────

InfoSetMcts::InfoSetMcts() : inner_(std::make_unique<Mcts>()) {}
InfoSetMcts::InfoSetMcts(MctsConfig cfg)
    : inner_(std::make_unique<Mcts>(cfg)) {}

void InfoSetMcts::set_config(const MctsConfig& cfg)         { inner_->set_config(cfg); }
const MctsConfig& InfoSetMcts::config() const               { return inner_->config(); }
void InfoSetMcts::set_evaluator(std::shared_ptr<IEvaluator> ev)      { inner_->set_evaluator(std::move(ev)); }
void InfoSetMcts::set_rollout_policy(std::shared_ptr<IRolloutPolicy> p) { inner_->set_rollout_policy(std::move(p)); }
void InfoSetMcts::set_opponent_policy(std::function<CombatAction(Agent&, const World&)> p) { inner_->set_opponent_policy(std::move(p)); }
void InfoSetMcts::set_prior(std::shared_ptr<IPrior> p)      { inner_->set_prior(std::move(p)); }
void InfoSetMcts::advance_root(const CombatAction& committed) { inner_->advance_root(committed); }
void InfoSetMcts::reset_tree()                              { inner_->reset_tree(); }
Mcts&       InfoSetMcts::inner()        { return *inner_; }
const Mcts& InfoSetMcts::inner() const  { return *inner_; }

CombatAction InfoSetMcts::search(World& truth, Agent& hero) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    WorldSnapshot truth_snap = truth.snapshot();
    const MctsConfig& cfg = inner_->config();

    // Seed root against the belief-mean determinization for stable slot
    // ordering. If there's no belief (caller misused the class), fall back to
    // plain MCTS against truth.
    const bool reused = (inner_->last_root() != nullptr);
    if (!reused) {
        WorldSnapshot mean_snap = truth_snap;
        if (belief_) {
            patch_snapshot_with_particles(mean_snap, belief_->mean());
        }
        truth.restore(mean_snap);
        inner_->ensure_root(truth, hero);
    }

    const int  iter_cap = cfg.iterations > 0 ? cfg.iterations
                                              : std::numeric_limits<int>::max();
    const bool time_cap = cfg.budget_ms > 0;

    std::mt19937_64 sample_rng(cfg.seed ^ 0x15ABCDEF12345ULL);

    int it = 0;
    for (; it < iter_cap; it++) {
        if (time_cap) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t_start).count();
            if (elapsed >= cfg.budget_ms) break;
        }

        // Sample a determinization; patch snapshot; restore.
        WorldSnapshot snap = truth_snap;
        if (belief_) {
            auto sampled = belief_->sample(sample_rng);
            patch_snapshot_with_particles(snap, sampled);
        }
        truth.restore(snap);
        truth.seed(cfg.seed + static_cast<uint64_t>(it) * 7919ull);

        inner_->run_iteration(truth, hero);
    }

    // Restore truth and gather stats.
    truth.restore(truth_snap);

    const auto t_end = clock::now();
    stats_.iterations  = it;
    stats_.elapsed_ms  = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count());
    stats_.reused_root = reused;
    stats_.mean_ess    = belief_ ? belief_->effective_sample_size() : 0.0f;

    const Node* root = inner_->last_root();
    if (root) {
        stats_.root_children = static_cast<int>(root->children.size());
        int tree_size = 1;
        // Cheap recursive size (duplicates mcts.cpp's count_tree, which is
        // static there). Fine — tree_size is debug-only.
        std::function<void(const Node*)> count = [&](const Node* n) {
            if (!n) return;
            for (const auto& ch : n->children) { tree_size++; count(ch.get()); }
        };
        count(root);
        stats_.tree_size = tree_size;

        const Node* best = nullptr;
        int best_visits = -1;
        for (const auto& up : root->children) {
            if (up->visits > best_visits) { best_visits = up->visits; best = up.get(); }
        }
        stats_.best_visits = best ? best->visits : 0;
        stats_.best_mean   = best ? best->mean() : 0.0f;
    } else {
        stats_.root_children = 0;
        stats_.tree_size = 0;
        stats_.best_visits = 0;
        stats_.best_mean = 0.0f;
    }

    return inner_->best_action();
}

// ─── InfoSetTeamMcts ───────────────────────────────────────────────────────
//
// Implementation note: TeamMcts does not expose single-iteration hooks, so
// this wrapper drives TeamMcts::search() per determinization with a very
// small per-call iteration budget. The root-tree-reuse path inside TeamMcts
// handles accumulation across calls (advance_root is not invoked between
// determinizations — the same game-state root is continuously re-searched).
// Stats returned are aggregated across determinizations.

InfoSetTeamMcts::InfoSetTeamMcts() : inner_(std::make_unique<TeamMcts>()) {}
InfoSetTeamMcts::InfoSetTeamMcts(MctsConfig cfg)
    : inner_(std::make_unique<TeamMcts>(cfg)) {}

void InfoSetTeamMcts::set_config(const MctsConfig& cfg) { inner_->set_config(cfg); }
const MctsConfig& InfoSetTeamMcts::config() const       { return inner_->config(); }
void InfoSetTeamMcts::set_evaluator(std::shared_ptr<ITeamEvaluator> ev) { inner_->set_evaluator(std::move(ev)); }
void InfoSetTeamMcts::set_rollout_policy(std::shared_ptr<IRolloutPolicy> p) { inner_->set_rollout_policy(std::move(p)); }
void InfoSetTeamMcts::set_opponent_policy(std::function<CombatAction(Agent&, const World&)> p) { inner_->set_opponent_policy(std::move(p)); }
void InfoSetTeamMcts::set_prior(std::shared_ptr<IPrior> p) { inner_->set_prior(std::move(p)); }
void InfoSetTeamMcts::reset_tree()                       { inner_->reset_tree(); }
TeamMcts&       InfoSetTeamMcts::inner()                 { return *inner_; }
const TeamMcts& InfoSetTeamMcts::inner() const           { return *inner_; }

InfoSetTeamMcts::JointOut InfoSetTeamMcts::search(
    World& truth, const std::vector<Agent*>& heroes) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    WorldSnapshot truth_snap = truth.snapshot();
    const MctsConfig base_cfg = inner_->config();

    // We split the base iteration budget across determinizations. 16 is a
    // reasonable default: fine enough to produce meaningful belief averaging,
    // coarse enough that each batch has enough iterations to explore.
    const int total_iters = base_cfg.iterations > 0 ? base_cfg.iterations : 256;
    const int batches = std::max(1, std::min(16, total_iters / 8));
    const int per_batch = std::max(1, total_iters / batches);

    MctsConfig batch_cfg = base_cfg;
    batch_cfg.iterations = per_batch;
    batch_cfg.budget_ms  = 0;  // we handle time at our layer

    // Pin root against belief-mean for stability on the first call.
    bool first_call = (inner_->last_root() == nullptr);
    if (first_call) {
        WorldSnapshot mean_snap = truth_snap;
        if (belief_) patch_snapshot_with_particles(mean_snap, belief_->mean());
        truth.restore(mean_snap);
        inner_->set_config(batch_cfg);
        (void)inner_->search(truth, heroes);
    }

    std::mt19937_64 sample_rng(base_cfg.seed ^ 0x15ABCDEF12345ULL);

    const bool time_cap = base_cfg.budget_ms > 0;
    int total = 0;
    int batch_idx = 0;
    while (batch_idx < batches) {
        if (time_cap) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t_start).count();
            if (elapsed >= base_cfg.budget_ms) break;
        }

        WorldSnapshot snap = truth_snap;
        if (belief_) {
            auto sampled = belief_->sample(sample_rng);
            patch_snapshot_with_particles(snap, sampled);
        }
        truth.restore(snap);
        MctsConfig c = batch_cfg;
        c.seed = base_cfg.seed + static_cast<uint64_t>(batch_idx) * 101ull;
        inner_->set_config(c);
        (void)inner_->search(truth, heroes);
        total += per_batch;
        batch_idx++;
    }

    inner_->set_config(base_cfg);
    truth.restore(truth_snap);

    // Pick the best joint from the accumulated tree. We don't have direct
    // access to the root, but inner_->search returned the most-visited joint
    // on its last call — use that. For correctness across batches, re-run
    // one final batch with zero iterations to get the visit-argmax...
    // Simpler: run one tiny search call to read back the best joint.
    JointOut out;
    {
        WorldSnapshot snap = truth_snap;
        if (belief_) patch_snapshot_with_particles(snap, belief_->mean());
        truth.restore(snap);
        MctsConfig c = batch_cfg; c.iterations = 1;
        inner_->set_config(c);
        auto joint = inner_->search(truth, heroes);
        out.per_hero = std::move(joint.per_hero);
        inner_->set_config(base_cfg);
        truth.restore(truth_snap);
    }

    const auto t_end = clock::now();
    stats_.iterations = total;
    stats_.elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count());
    stats_.mean_ess = belief_ ? belief_->effective_sample_size() : 0.0f;
    // tree_size / best stats: read from the inner stats.
    stats_.tree_size  = inner_->last_stats().tree_size;
    stats_.best_mean  = inner_->last_stats().best_mean;
    stats_.best_visits = inner_->last_stats().best_visits;

    return out;
}

} // namespace brogameagent::mcts
