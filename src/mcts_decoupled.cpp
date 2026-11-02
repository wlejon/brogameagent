#include "brogameagent/mcts.h"
#include "mcts_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <thread>

namespace brogameagent::mcts {

// ─── DecoupledMcts ─────────────────────────────────────────────────────────

DecoupledMcts::PlayerStats DecoupledMcts::build_stats_(const Agent& self, const World& world) const {
    PlayerStats s;
    s.actions = legal_actions(self, world);
    s.visits.assign(s.actions.size(), 0);
    s.total_value.assign(s.actions.size(), 0.0f);
    s.priors = compute_priors(prior_.get(), self, world, s.actions);
    return s;
}

int DecoupledMcts::pick_action_idx_(const PlayerStats& stats, int node_visits,
                                     bool minimize) const {
    const bool use_puct = cfg_.prior_c > 0.0f;

    if (!use_puct) {
        // Unvisited-first rule: any action with visits=0 is tried before any
        // UCT evaluation. Ensures each action on each side gets at least one
        // sample before we start optimising.
        for (size_t i = 0; i < stats.visits.size(); i++) {
            if (stats.visits[i] == 0) return static_cast<int>(i);
        }
        const float ln_n = std::log(static_cast<float>(std::max(1, node_visits)));
        int   best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        const float c = cfg_.uct_c;
        for (size_t i = 0; i < stats.visits.size(); i++) {
            float mean    = stats.total_value[i] / static_cast<float>(stats.visits[i]);
            float exploit = minimize ? -mean : mean;
            float explore = c * std::sqrt(ln_n / static_cast<float>(stats.visits[i]));
            float score   = exploit + explore;
            if (score > best_score) { best_score = score; best = static_cast<int>(i); }
        }
        return best;
    }

    // PUCT path: prior P already encodes which actions to prefer when
    // unvisited, so we don't need the unvisited-first rule.
    const float sqrt_N = std::sqrt(static_cast<float>(std::max(1, node_visits)));
    const float c = cfg_.prior_c;
    int   best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < stats.visits.size(); i++) {
        float mean = stats.visits[i] > 0
            ? stats.total_value[i] / static_cast<float>(stats.visits[i])
            : 0.0f;
        float exploit = minimize ? -mean : mean;
        float p = (i < stats.priors.size())
            ? stats.priors[i]
            : 1.0f / static_cast<float>(stats.actions.size());
        float explore = c * p * sqrt_N / (1.0f + static_cast<float>(stats.visits[i]));
        float score   = exploit + explore;
        if (score > best_score) { best_score = score; best = static_cast<int>(i); }
    }
    return best;
}

void DecoupledMcts::step_joint_(World& world,
                                 Agent& hero, const CombatAction& hero_act,
                                 Agent& opp,  const CombatAction& opp_act) {
    const float dt = cfg_.sim_dt;
    for (int t = 0; t < cfg_.action_repeat; t++) {
        mcts::apply(hero, world, hero_act, dt);
        mcts::apply(opp,  world, opp_act,  dt);
        world.stepProjectiles(dt);
        world.cullProjectiles();
        if (is_terminal_for(hero, world)) break;
    }
}

float DecoupledMcts::rollout_(World& world, Agent& hero, Agent& opp) {
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, hero.unit().id)
            : HpDeltaEvaluator{}.evaluate(world, hero.unit().id);
    };
    if (cfg_.use_leaf_value) return eval();

    int steps = 0;
    IRolloutPolicy* policy = rollout_policy_.get();
    RandomRollout fallback;
    if (!policy) policy = &fallback;

    while (steps < cfg_.rollout_horizon && !is_terminal_for(hero, world)) {
        CombatAction h = hero.unit().alive() ? policy->choose(hero, world) : CombatAction{};
        CombatAction o = opp.unit().alive()  ? policy->choose(opp, world)  : CombatAction{};
        step_joint_(world, hero, h, opp, o);
        steps++;
    }
    return eval();
}

DecoupledMcts::Joint DecoupledMcts::search(World& world, Agent& hero, Agent& opp) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_) {
        reused = true;
    } else {
        root_ = std::make_unique<DNode>();
        root_->hero_stats = build_stats_(hero, world);
        root_->opp_stats  = build_stats_(opp,  world);
    }

    const int  iter_cap = cfg_.iterations > 0 ? cfg_.iterations
                                              : std::numeric_limits<int>::max();
    const bool time_cap = cfg_.budget_ms > 0;

    int it = 0;
    for (; it < iter_cap; it++) {
        if (time_cap) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t_start).count();
            if (elapsed >= cfg_.budget_ms) break;
        }

        world.restore(saved);
        world.seed(cfg_.seed
                   + static_cast<uint64_t>(it)
                   + static_cast<uint64_t>(root_->visits) * 7919ull);

        // Descend: at each node pick a joint (h_idx, o_idx); walk or expand.
        DNode* node = root_.get();
        while (true) {
            if (is_terminal_for(hero, world)) break;
            if (node->hero_stats.actions.empty() || node->opp_stats.actions.empty()) break;

            int h_idx = pick_action_idx_(node->hero_stats, node->visits, false);
            int o_idx = pick_action_idx_(node->opp_stats,  node->visits, true);
            CombatAction h_act = node->hero_stats.actions[h_idx];
            CombatAction o_act = node->opp_stats.actions[o_idx];

            step_joint_(world, hero, h_act, opp, o_act);
            uint32_t key = pack_key_(h_idx, o_idx);

            auto it_child = node->children.find(key);
            if (it_child != node->children.end()) {
                node = it_child->second.get();
                continue;  // descend further
            }
            // Expansion: create child for this joint action.
            auto child = std::make_unique<DNode>();
            child->parent = node;
            child->parent_hero_idx = h_idx;
            child->parent_opp_idx  = o_idx;
            child->hero_stats = build_stats_(hero, world);
            child->opp_stats  = build_stats_(opp,  world);
            DNode* raw = child.get();
            node->children.emplace(key, std::move(child));
            node = raw;
            break;
        }

        float value = rollout_(world, hero, opp);

        // Backprop: walk up, update per-action stats at each ancestor.
        node->visits++;
        while (node->parent) {
            DNode* p = node->parent;
            p->hero_stats.visits[node->parent_hero_idx]++;
            p->hero_stats.total_value[node->parent_hero_idx] += value;
            p->opp_stats.visits[node->parent_opp_idx]++;
            p->opp_stats.total_value[node->parent_opp_idx]   += value;
            p->visits++;
            node = p;
        }
    }

    world.restore(saved);

    // Pick each player's most-visited root action.
    Joint out{};
    int best_h_visits = -1, best_o_visits = -1;
    int best_h_idx = -1, best_o_idx = -1;
    for (size_t i = 0; i < root_->hero_stats.visits.size(); i++) {
        if (root_->hero_stats.visits[i] > best_h_visits) {
            best_h_visits = root_->hero_stats.visits[i];
            best_h_idx = static_cast<int>(i);
        }
    }
    for (size_t i = 0; i < root_->opp_stats.visits.size(); i++) {
        if (root_->opp_stats.visits[i] > best_o_visits) {
            best_o_visits = root_->opp_stats.visits[i];
            best_o_idx = static_cast<int>(i);
        }
    }
    if (best_h_idx >= 0) out.hero = root_->hero_stats.actions[best_h_idx];
    if (best_o_idx >= 0) out.opp  = root_->opp_stats.actions[best_o_idx];

    // Count tree size (recursive, sparse children).
    auto count_d = [](auto& self_ref, const DNode* n) -> int {
        if (!n) return 0;
        int total = 1;
        for (const auto& [k, ch] : n->children) total += self_ref(self_ref, ch.get());
        return total;
    };
    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = count_d(count_d, root_.get());
    stats_.best_mean     = best_h_visits > 0
        ? root_->hero_stats.total_value[best_h_idx] / static_cast<float>(best_h_visits)
        : 0.0f;
    stats_.best_visits   = best_h_visits > 0 ? best_h_visits : 0;
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;
    return out;
}

void DecoupledMcts::advance_root(const CombatAction& hero_committed,
                                  const CombatAction& opp_committed) {
    if (!root_) return;
    int h_idx = -1, o_idx = -1;
    for (size_t i = 0; i < root_->hero_stats.actions.size(); i++) {
        if (root_->hero_stats.actions[i] == hero_committed) { h_idx = static_cast<int>(i); break; }
    }
    for (size_t i = 0; i < root_->opp_stats.actions.size(); i++) {
        if (root_->opp_stats.actions[i] == opp_committed)  { o_idx = static_cast<int>(i); break; }
    }
    if (h_idx < 0 || o_idx < 0) { root_.reset(); return; }

    uint32_t key = pack_key_(h_idx, o_idx);
    auto it = root_->children.find(key);
    if (it == root_->children.end()) { root_.reset(); return; }
    std::unique_ptr<DNode> promoted = std::move(it->second);
    promoted->parent = nullptr;
    promoted->parent_hero_idx = -1;
    promoted->parent_opp_idx  = -1;
    root_ = std::move(promoted);
}

// ─── Root-parallel search ──────────────────────────────────────────────────

namespace {

// Deterministic ordering key for CombatAction so merged maps are iteration-
// order independent.
struct CombatActionKey {
    int move_dir;
    int attack_slot;
    int ability_slot;
    bool operator<(const CombatActionKey& o) const {
        if (move_dir    != o.move_dir)    return move_dir    < o.move_dir;
        if (attack_slot != o.attack_slot) return attack_slot < o.attack_slot;
        return ability_slot < o.ability_slot;
    }
    static CombatActionKey from(const CombatAction& a) {
        return { static_cast<int>(a.move_dir), a.attack_slot, a.ability_slot };
    }
    CombatAction to() const {
        CombatAction a;
        a.move_dir    = static_cast<MoveDir>(move_dir);
        a.attack_slot = static_cast<int8_t>(attack_slot);
        a.ability_slot = static_cast<int8_t>(ability_slot);
        return a;
    }
};

} // namespace

CombatAction root_parallel_search(
    const std::vector<World*>& worlds,
    int hero_id,
    const MctsConfig& cfg,
    std::shared_ptr<IEvaluator>     evaluator,
    std::shared_ptr<IRolloutPolicy> rollout_policy,
    OpponentPolicy                  opponent_policy,
    ParallelSearchStats*            out_stats)
{
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    const int N = static_cast<int>(worlds.size());
    if (N == 0) return {};

    std::vector<Mcts>        engines(N);
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (int i = 0; i < N; i++) {
        MctsConfig thread_cfg = cfg;
        thread_cfg.seed = cfg.seed + static_cast<uint64_t>(i) * 1000003ull;
        engines[i].set_config(thread_cfg);
        engines[i].set_evaluator(evaluator);
        engines[i].set_rollout_policy(rollout_policy);
        engines[i].set_opponent_policy(opponent_policy);
    }

    for (int i = 0; i < N; i++) {
        threads.emplace_back([&, i]() {
            Agent* hero = worlds[i]->findById(hero_id);
            if (!hero) return;
            engines[i].search(*worlds[i], *hero);
        });
    }
    for (auto& t : threads) t.join();

    // Merge: sum root-child visits across all trees.
    std::map<CombatActionKey, int> merged;
    int total_iters = 0;
    for (int i = 0; i < N; i++) {
        const Node* root = engines[i].last_root();
        total_iters += engines[i].last_stats().iterations;
        if (!root) continue;
        for (const auto& child : root->children) {
            merged[CombatActionKey::from(child->action)] += child->visits;
        }
    }

    CombatAction best{};
    int best_visits = -1;
    for (const auto& [k, v] : merged) {
        if (v > best_visits) { best_visits = v; best = k.to(); }
    }

    const auto t_end = clock::now();
    if (out_stats) {
        out_stats->num_threads      = N;
        out_stats->total_iterations = total_iters;
        out_stats->elapsed_ms       = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        out_stats->merged_best_visits = best_visits > 0 ? best_visits : 0;
    }
    return best;
}

DecoupledMcts::Joint root_parallel_search_decoupled(
    const std::vector<World*>& worlds,
    int hero_id, int opp_id,
    const MctsConfig& cfg,
    std::shared_ptr<IEvaluator>     evaluator,
    std::shared_ptr<IRolloutPolicy> rollout_policy,
    ParallelSearchStats*            out_stats)
{
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    const int N = static_cast<int>(worlds.size());
    if (N == 0) return {};

    std::vector<DecoupledMcts> engines(N);
    std::vector<std::thread>   threads;
    threads.reserve(N);

    for (int i = 0; i < N; i++) {
        MctsConfig thread_cfg = cfg;
        thread_cfg.seed = cfg.seed + static_cast<uint64_t>(i) * 1000003ull;
        engines[i].set_config(thread_cfg);
        engines[i].set_evaluator(evaluator);
        engines[i].set_rollout_policy(rollout_policy);
    }

    for (int i = 0; i < N; i++) {
        threads.emplace_back([&, i]() {
            Agent* hero = worlds[i]->findById(hero_id);
            Agent* opp  = worlds[i]->findById(opp_id);
            if (!hero || !opp) return;
            engines[i].search(*worlds[i], *hero, *opp);
        });
    }
    for (auto& t : threads) t.join();

    std::map<CombatActionKey, int> hero_merged;
    std::map<CombatActionKey, int> opp_merged;
    int total_iters = 0;
    for (int i = 0; i < N; i++) {
        const auto* root = engines[i].last_root();
        total_iters += engines[i].last_stats().iterations;
        if (!root) continue;
        for (size_t a = 0; a < root->hero_stats.actions.size(); a++) {
            hero_merged[CombatActionKey::from(root->hero_stats.actions[a])]
                += root->hero_stats.visits[a];
        }
        for (size_t a = 0; a < root->opp_stats.actions.size(); a++) {
            opp_merged[CombatActionKey::from(root->opp_stats.actions[a])]
                += root->opp_stats.visits[a];
        }
    }

    DecoupledMcts::Joint out{};
    int best_h = -1, best_o = -1;
    for (const auto& [k, v] : hero_merged) {
        if (v > best_h) { best_h = v; out.hero = k.to(); }
    }
    for (const auto& [k, v] : opp_merged) {
        if (v > best_o) { best_o = v; out.opp = k.to(); }
    }

    const auto t_end = clock::now();
    if (out_stats) {
        out_stats->num_threads      = N;
        out_stats->total_iterations = total_iters;
        out_stats->elapsed_ms       = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        out_stats->merged_best_visits = best_h > 0 ? best_h : 0;
    }
    return out;
}

} // namespace brogameagent::mcts
