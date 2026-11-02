#include "brogameagent/mcts.h"
#include "mcts_internal.h"
#include "brogameagent/action_mask.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace brogameagent::mcts {

// ─── TeamHpDeltaEvaluator ──────────────────────────────────────────────────

float TeamHpDeltaEvaluator::evaluate(const World& world, int team_id) const {
    float team_sum = 0.0f;  int team_count = 0;
    float enemy_sum = 0.0f; int enemy_count = 0;
    bool  any_team_alive  = false;
    bool  any_enemy_alive = false;
    for (Agent* a : world.agents()) {
        bool alive = a->unit().alive();
        float frac = alive ? a->unit().hp / std::max(1e-6f, a->unit().maxHp) : 0.0f;
        if (a->unit().teamId == team_id) {
            team_count++;
            team_sum += frac;
            if (alive) any_team_alive = true;
        } else {
            enemy_count++;
            enemy_sum += frac;
            if (alive) any_enemy_alive = true;
        }
    }
    if (!any_team_alive)  return -1.0f;
    if (!any_enemy_alive) return  1.0f;

    float team_avg  = team_count  > 0 ? team_sum  / static_cast<float>(team_count)  : 0.0f;
    float enemy_avg = enemy_count > 0 ? enemy_sum / static_cast<float>(enemy_count) : 0.0f;
    float delta = team_avg - enemy_avg;
    if (delta < -1.0f) delta = -1.0f;
    if (delta >  1.0f) delta =  1.0f;
    return delta;
}


// ─── TeamAdvantageEvaluator ────────────────────────────────────────────────
//
// Blends HP delta with an alive-count delta so kills dominate scratches.
// Weighted 0.6 alive + 0.4 HP by default — a kill is worth ~(0.6/N_team + small
// HP swing), while a balanced HP trade with no deaths is near 0. This breaks
// the "Retreat scores 0 because nobody takes damage" degenerate case that
// pure HP-delta suffers from at short horizons, while leaving terminal wins
// and losses pegged at ±1.

float TeamAdvantageEvaluator::evaluate(const World& world, int team_id) const {
    int team_n = 0, team_alive = 0;
    int enemy_n = 0, enemy_alive = 0;
    float team_hp = 0.0f, enemy_hp = 0.0f;
    for (Agent* a : world.agents()) {
        bool alive = a->unit().alive();
        float frac = alive ? a->unit().hp / std::max(1e-6f, a->unit().maxHp) : 0.0f;
        if (a->unit().teamId == team_id) {
            team_n++;
            if (alive) team_alive++;
            team_hp += frac;
        } else {
            enemy_n++;
            if (alive) enemy_alive++;
            enemy_hp += frac;
        }
    }
    if (team_alive == 0)  return -1.0f;
    if (enemy_alive == 0) return  1.0f;

    float alive_delta = (team_n  > 0 ? static_cast<float>(team_alive)  / team_n  : 0.0f)
                      - (enemy_n > 0 ? static_cast<float>(enemy_alive) / enemy_n : 0.0f);
    float hp_delta    = (team_n  > 0 ? team_hp  / team_n  : 0.0f)
                      - (enemy_n > 0 ? enemy_hp / enemy_n : 0.0f);

    float score = 0.6f * alive_delta + 0.4f * hp_delta;
    if (score < -1.0f) score = -1.0f;
    if (score >  1.0f) score =  1.0f;
    return score;
}


// ─── TeamPositionEvaluator ─────────────────────────────────────────────────
//
// Blends the alive + HP deltas of TeamAdvantageEvaluator with a positional
// term: count the fraction of living heroes that have at least one enemy in
// their attack range (can fight back this window). Pure HP/kill evaluators
// score a full-HP team hiding in the corner identically to a full-HP team
// engaging — which lets the planner drift into dominated-but-safe states at
// short horizons. The positional bonus fixes that without needing LOS rays.

float TeamPositionEvaluator::evaluate(const World& world, int team_id) const {
    int   team_n = 0, team_alive = 0;
    int   enemy_n = 0, enemy_alive = 0;
    float team_hp = 0.0f, enemy_hp = 0.0f;
    int   team_in_range = 0, enemy_in_range = 0;

    const auto& all = world.agents();
    for (Agent* a : all) {
        bool alive = a->unit().alive();
        float frac = alive ? a->unit().hp / std::max(1e-6f, a->unit().maxHp) : 0.0f;
        if (a->unit().teamId == team_id) {
            team_n++;
            if (alive) team_alive++;
            team_hp += frac;
        } else {
            enemy_n++;
            if (alive) enemy_alive++;
            enemy_hp += frac;
        }
    }
    if (team_alive == 0)  return -1.0f;
    if (enemy_alive == 0) return  1.0f;

    // Positional pass — quadratic in agent count but tiny at this scale.
    for (Agent* a : all) {
        if (!a->unit().alive()) continue;
        float r2 = a->unit().attackRange * a->unit().attackRange;
        bool has_in_range = false;
        for (Agent* b : all) {
            if (a == b) continue;
            if (!b->unit().alive()) continue;
            if (b->unit().teamId == a->unit().teamId) continue;
            float dx = b->x() - a->x();
            float dz = b->z() - a->z();
            if (dx * dx + dz * dz <= r2) { has_in_range = true; break; }
        }
        if (!has_in_range) continue;
        if (a->unit().teamId == team_id) team_in_range++;
        else                              enemy_in_range++;
    }

    float alive_delta = (team_n  > 0 ? static_cast<float>(team_alive)  / team_n  : 0.0f)
                      - (enemy_n > 0 ? static_cast<float>(enemy_alive) / enemy_n : 0.0f);
    float hp_delta    = (team_n  > 0 ? team_hp  / team_n  : 0.0f)
                      - (enemy_n > 0 ? enemy_hp / enemy_n : 0.0f);
    float pos_delta   = (team_alive  > 0 ? static_cast<float>(team_in_range)  / team_alive  : 0.0f)
                      - (enemy_alive > 0 ? static_cast<float>(enemy_in_range) / enemy_alive : 0.0f);

    float score = 0.5f * alive_delta + 0.3f * hp_delta + 0.2f * pos_delta;
    if (score < -1.0f) score = -1.0f;
    if (score >  1.0f) score =  1.0f;
    return score;
}


// ─── TeamMcts ──────────────────────────────────────────────────────────────

TeamMcts::PlayerStats TeamMcts::build_stats_(const Agent& self, const World& world) const {
    PlayerStats s;
    s.actions = legal_actions(self, world);
    s.visits.assign(s.actions.size(), 0);
    s.total_value.assign(s.actions.size(), 0.0f);
    s.priors = compute_priors(prior_.get(), self, world, s.actions);
    return s;
}

int TeamMcts::pick_action_idx_(const PlayerStats& stats, int node_visits) const {
    const bool use_puct = cfg_.prior_c > 0.0f;

    if (!use_puct) {
        for (size_t i = 0; i < stats.visits.size(); i++) {
            if (stats.visits[i] == 0) return static_cast<int>(i);
        }
        const float ln_n = std::log(static_cast<float>(std::max(1, node_visits)));
        const float c = cfg_.uct_c;
        int   best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < stats.visits.size(); i++) {
            float mean    = stats.total_value[i] / static_cast<float>(stats.visits[i]);
            float explore = c * std::sqrt(ln_n / static_cast<float>(stats.visits[i]));
            float score   = mean + explore;   // all heroes maximise (cooperative)
            if (score > best_score) { best_score = score; best = static_cast<int>(i); }
        }
        return best;
    }

    const float sqrt_N = std::sqrt(static_cast<float>(std::max(1, node_visits)));
    const float c = cfg_.prior_c;
    int   best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < stats.visits.size(); i++) {
        float mean = stats.visits[i] > 0
            ? stats.total_value[i] / static_cast<float>(stats.visits[i])
            : 0.0f;
        float p = (i < stats.priors.size())
            ? stats.priors[i]
            : 1.0f / static_cast<float>(stats.actions.size());
        float explore = c * p * sqrt_N / (1.0f + static_cast<float>(stats.visits[i]));
        float score   = mean + explore;
        if (score > best_score) { best_score = score; best = static_cast<int>(i); }
    }
    return best;
}

void TeamMcts::step_joint_(World& world,
                            const std::vector<Agent*>& heroes,
                            const std::vector<CombatAction>& hero_actions) {
    // Opponents get one action per decision window, held across sub-ticks —
    // mirrors the hero action-persistence contract.
    std::vector<std::pair<Agent*, CombatAction>> enemy_acts;
    int team_id = heroes.empty() ? 0 : heroes.front()->unit().teamId;
    for (Agent* a : world.agents()) {
        bool is_hero = false;
        for (Agent* h : heroes) if (a == h) { is_hero = true; break; }
        if (is_hero) continue;
        if (a->unit().teamId == team_id) continue; // ally not in search
        CombatAction act{};
        if (a->unit().alive() && opponent_) act = opponent_(*a, world);
        enemy_acts.emplace_back(a, act);
    }

    const float dt = cfg_.sim_dt;
    for (int t = 0; t < cfg_.action_repeat; t++) {
        for (size_t i = 0; i < heroes.size(); i++) {
            mcts::apply(*heroes[i], world, hero_actions[i], dt);
        }
        for (auto& [a, act] : enemy_acts) {
            mcts::apply(*a, world, act, dt);
        }
        world.stepProjectiles(dt);
        world.cullProjectiles();

        bool any_team_alive  = false;
        bool any_enemy_alive = false;
        for (Agent* a : world.agents()) {
            if (!a->unit().alive()) continue;
            if (a->unit().teamId == team_id) any_team_alive  = true;
            else                              any_enemy_alive = true;
        }
        if (!any_team_alive || !any_enemy_alive) break;
    }
}

float TeamMcts::rollout_(World& world, const std::vector<Agent*>& heroes, int team_id) {
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, team_id)
            : TeamHpDeltaEvaluator{}.evaluate(world, team_id);
    };
    if (cfg_.use_leaf_value) return eval();

    IRolloutPolicy* policy = rollout_policy_.get();
    RandomRollout fallback;
    if (!policy) policy = &fallback;

    int steps = 0;
    while (steps < cfg_.rollout_horizon) {
        bool any_team_alive = false, any_enemy_alive = false;
        for (Agent* a : world.agents()) {
            if (!a->unit().alive()) continue;
            if (a->unit().teamId == team_id) any_team_alive = true;
            else                              any_enemy_alive = true;
        }
        if (!any_team_alive || !any_enemy_alive) break;

        std::vector<CombatAction> hero_actions(heroes.size());
        for (size_t i = 0; i < heroes.size(); i++) {
            hero_actions[i] = heroes[i]->unit().alive()
                ? policy->choose(*heroes[i], world)
                : CombatAction{};
        }
        step_joint_(world, heroes, hero_actions);
        steps++;
    }
    return evaluator_
        ? evaluator_->evaluate(world, team_id)
        : TeamHpDeltaEvaluator{}.evaluate(world, team_id);
}

TeamMcts::JointAction TeamMcts::search(World& world, const std::vector<Agent*>& heroes) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    JointAction out;
    out.per_hero.resize(heroes.size());
    if (heroes.empty()) return out;
    const int team_id = heroes.front()->unit().teamId;

    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_ && root_->per_hero.size() == heroes.size()) {
        reused = true;
    } else {
        root_ = std::make_unique<TNode>();
        root_->per_hero.reserve(heroes.size());
        for (Agent* h : heroes) root_->per_hero.push_back(build_stats_(*h, world));
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

        // Descend and (possibly) expand.
        TNode* node = root_.get();
        while (true) {
            // Terminal check before deciding.
            bool any_team_alive = false, any_enemy_alive = false;
            for (Agent* a : world.agents()) {
                if (!a->unit().alive()) continue;
                if (a->unit().teamId == team_id) any_team_alive = true;
                else                              any_enemy_alive = true;
            }
            if (!any_team_alive || !any_enemy_alive) break;

            std::vector<int> idxs(heroes.size());
            bool any_legal = false;
            for (size_t h = 0; h < heroes.size(); h++) {
                if (node->per_hero[h].actions.empty()) { idxs[h] = -1; continue; }
                idxs[h] = pick_action_idx_(node->per_hero[h], node->visits);
                any_legal = true;
            }
            if (!any_legal) break;

            std::vector<CombatAction> hero_actions(heroes.size());
            for (size_t h = 0; h < heroes.size(); h++) {
                hero_actions[h] = idxs[h] >= 0
                    ? node->per_hero[h].actions[idxs[h]]
                    : CombatAction{};
            }
            step_joint_(world, heroes, hero_actions);

            auto it_child = node->children.find(idxs);
            if (it_child != node->children.end()) {
                node = it_child->second.get();
                continue;
            }
            auto child = std::make_unique<TNode>();
            child->parent = node;
            child->parent_action_idx = idxs;
            child->per_hero.reserve(heroes.size());
            for (Agent* h : heroes) child->per_hero.push_back(build_stats_(*h, world));
            TNode* raw = child.get();
            node->children.emplace(std::move(idxs), std::move(child));
            node = raw;
            break;
        }

        float value = rollout_(world, heroes, team_id);

        node->visits++;
        while (node->parent) {
            TNode* p = node->parent;
            for (size_t h = 0; h < heroes.size(); h++) {
                int ai = node->parent_action_idx[h];
                if (ai < 0) continue;
                p->per_hero[h].visits[ai]++;
                p->per_hero[h].total_value[ai] += value;
            }
            p->visits++;
            node = p;
        }
    }

    world.restore(saved);

    // Pick each hero's most-visited root action.
    for (size_t h = 0; h < heroes.size(); h++) {
        const auto& st = root_->per_hero[h];
        int best_v = -1, best_i = -1;
        for (size_t i = 0; i < st.visits.size(); i++) {
            if (st.visits[i] > best_v) { best_v = st.visits[i]; best_i = static_cast<int>(i); }
        }
        out.per_hero[h] = best_i >= 0 ? st.actions[best_i] : CombatAction{};
    }

    // Tree-size walk.
    auto count_t = [](auto& self_ref, const TNode* n) -> int {
        if (!n) return 0;
        int total = 1;
        for (const auto& [k, ch] : n->children) total += self_ref(self_ref, ch.get());
        return total;
    };
    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = count_t(count_t, root_.get());
    stats_.best_visits   = root_->visits;
    stats_.best_mean     = 0.0f;  // team mean is not well-defined across per-hero stats
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;
    return out;
}

void TeamMcts::advance_root(const JointAction& committed) {
    if (!root_) return;
    if (committed.per_hero.size() != root_->per_hero.size()) { root_.reset(); return; }

    std::vector<int> idxs(committed.per_hero.size(), -1);
    for (size_t h = 0; h < committed.per_hero.size(); h++) {
        const auto& st = root_->per_hero[h];
        for (size_t i = 0; i < st.actions.size(); i++) {
            if (st.actions[i] == committed.per_hero[h]) { idxs[h] = static_cast<int>(i); break; }
        }
        if (idxs[h] < 0) { root_.reset(); return; }
    }
    auto it = root_->children.find(idxs);
    if (it == root_->children.end()) { root_.reset(); return; }

    std::unique_ptr<TNode> promoted = std::move(it->second);
    promoted->parent = nullptr;
    promoted->parent_action_idx.clear();
    root_ = std::move(promoted);
}


// ─── Tactic layer (hierarchical team MCTS) ────────────────────────────────

namespace {

Agent* lowest_hp_enemy(const Agent& self, const World& world) {
    Agent* best = nullptr;
    float  best_hp = std::numeric_limits<float>::infinity();
    for (Agent* a : world.agents()) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        if (a->unit().hp < best_hp) { best_hp = a->unit().hp; best = a; }
    }
    return best;
}

// Given a target enemy id, find its slot index in the hero's action_mask
// enemy ordering. -1 if not present or not attackable.
int enemy_slot_of_id(const Agent& self, const World& world, int enemy_id) {
    if (enemy_id < 0) return -1;
    float mask[action_mask::TOTAL];
    int   enemy_ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, enemy_ids);
    for (int k = 0; k < action_mask::N_ENEMY_SLOTS; k++) {
        if (enemy_ids[k] == enemy_id) {
            return mask[k] > 0.0f ? k : -1;  // only if legal to attack
        }
    }
    return -1;
}

} // namespace

// Standalone "apply joint team actions for one decision window" — shared by
// TacticMcts and used during rollout. Mirrors TeamMcts::step_joint_.
void step_team_window(World& world,
                      const std::vector<Agent*>& heroes,
                      const std::vector<CombatAction>& hero_actions,
                      const OpponentPolicy& opponent,
                      float dt, int action_repeat,
                      int team_id) {
    std::vector<std::pair<Agent*, CombatAction>> enemy_acts;
    for (Agent* a : world.agents()) {
        bool is_hero = false;
        for (Agent* h : heroes) if (a == h) { is_hero = true; break; }
        if (is_hero) continue;
        if (a->unit().teamId == team_id) continue;
        CombatAction act{};
        if (a->unit().alive() && opponent) act = opponent(*a, world);
        enemy_acts.emplace_back(a, act);
    }
    for (int t = 0; t < action_repeat; t++) {
        for (size_t i = 0; i < heroes.size(); i++) {
            mcts::apply(*heroes[i], world, hero_actions[i], dt);
        }
        for (auto& [a, act] : enemy_acts) {
            mcts::apply(*a, world, act, dt);
        }
        world.stepProjectiles(dt);
        world.cullProjectiles();
    }
}

bool team_terminal(const World& world, int team_id) {
    bool team_alive = false, enemy_alive = false;
    for (Agent* a : world.agents()) {
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == team_id) team_alive = true;
        else                              enemy_alive = true;
    }
    return !team_alive || !enemy_alive;
}

CombatAction tactic_to_action(const Tactic& t, const Agent& hero, const World& world) {
    if (!hero.unit().alive()) return {};
    CombatAction a;
    a.move_dir = MoveDir::Hold;
    a.attack_slot = -1;
    a.ability_slot = -1;

    switch (t.kind) {
        case TacticKind::Hold: {
            // Try to auto-attack any in-range enemy (slot 0 = nearest).
            float mask[action_mask::TOTAL];
            int   enemy_ids[action_mask::N_ENEMY_SLOTS];
            action_mask::build(hero, world, mask, enemy_ids);
            for (int k = 0; k < action_mask::N_ENEMY_SLOTS; k++) {
                if (mask[k] > 0.0f) { a.attack_slot = static_cast<int8_t>(k); break; }
            }
            break;
        }
        case TacticKind::FocusLowestHp: {
            Agent* target = lowest_hp_enemy(hero, world);
            if (!target) break;
            int slot = enemy_slot_of_id(hero, world, target->unit().id);
            float mask[action_mask::TOTAL];
            int   enemy_ids[action_mask::N_ENEMY_SLOTS];
            action_mask::build(hero, world, mask, enemy_ids);
            // First-available offensive ability. Slot 0 = heal by convention
            // (ai-arena scenario); skip it when picking offense.
            int ability = -1;
            for (int s = 1; s < action_mask::N_ABILITY_SLOTS; s++) {
                if (mask[action_mask::N_ENEMY_SLOTS + s] > 0.0f) {
                    ability = s;
                    break;
                }
            }
            if (slot >= 0) {
                a.attack_slot = static_cast<int8_t>(slot);
                a.ability_slot = static_cast<int8_t>(ability);
                a.move_dir = MoveDir::Hold;   // in range — stop and shoot
            } else {
                a.ability_slot = static_cast<int8_t>(ability);  // cast on the way
                a.move_dir = MoveDir::PathToTarget;
            }
            break;
        }
        case TacticKind::Scatter: {
            // Attack nearest in-range (slot 0), close otherwise. Cast the
            // first available offensive ability in parallel — abilities
            // typically have longer range than the auto-attack so they
            // chip the enemy down while closing.
            float mask[action_mask::TOTAL];
            int   enemy_ids[action_mask::N_ENEMY_SLOTS];
            action_mask::build(hero, world, mask, enemy_ids);
            int ability = -1;
            for (int s = 1; s < action_mask::N_ABILITY_SLOTS; s++) {
                if (mask[action_mask::N_ENEMY_SLOTS + s] > 0.0f) {
                    ability = s;
                    break;
                }
            }
            if (mask[0] > 0.0f) {
                a.attack_slot = 0;
                a.ability_slot = static_cast<int8_t>(ability);
            } else if (enemy_ids[0] >= 0) {
                a.move_dir = MoveDir::PathToTarget;
                a.ability_slot = static_cast<int8_t>(ability);
            }
            break;
        }
        case TacticKind::Retreat: {
            // Pathfind away from the nearest enemy so we route around
            // obstacles instead of sliding into a wall.
            a.move_dir = MoveDir::PathAway;
            break;
        }
        default: break;
    }
    return a;
}

std::vector<Tactic> legal_tactics(const std::vector<Agent*>& heroes,
                                   const World& world) {
    std::vector<Tactic> out;
    out.push_back({ TacticKind::Hold });  // always legal

    if (heroes.empty()) return out;
    const int team_id = heroes.front()->unit().teamId;

    bool any_enemy_alive = false;
    bool any_threat      = false;   // enemy within 1.5× its attack range of a hero
    for (Agent* e : world.agents()) {
        if (!e || !e->unit().alive()) continue;
        if (e->unit().teamId == team_id) continue;
        any_enemy_alive = true;

        const float threat_r  = std::max(1e-3f, e->unit().attackRange) * 1.5f;
        const float threat_r2 = threat_r * threat_r;
        for (Agent* h : heroes) {
            if (!h || !h->unit().alive()) continue;
            float dx = e->x() - h->x();
            float dz = e->z() - h->z();
            if (dx * dx + dz * dz <= threat_r2) { any_threat = true; break; }
        }
        if (any_threat) break;
    }

    if (any_enemy_alive) {
        out.push_back({ TacticKind::FocusLowestHp });
        out.push_back({ TacticKind::Scatter });
    }
    if (any_threat) {
        out.push_back({ TacticKind::Retreat });
    }
    return out;
}

namespace {

TacticMcts::TNode* tmcts_best_uct_child(TacticMcts::TNode* node, float c) {
    TacticMcts::TNode* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float ln_n = std::log(static_cast<float>(std::max(1, node->visits)));
    for (auto& up : node->children) {
        auto* ch = up.get();
        if (ch->visits == 0) return ch;
        float exploit = ch->mean();
        float explore = c * std::sqrt(ln_n / static_cast<float>(ch->visits));
        float score = exploit + explore;
        if (score > best_score) { best_score = score; best = ch; }
    }
    return best;
}

int tmcts_count_tree(const TacticMcts::TNode* n) {
    if (!n) return 0;
    int total = 1;
    for (const auto& ch : n->children) total += tmcts_count_tree(ch.get());
    return total;
}

} // namespace

void TacticMcts::step_tactic_(World& world, const std::vector<Agent*>& heroes,
                               const Tactic& tactic) {
    if (heroes.empty()) return;
    const int team_id = heroes.front()->unit().teamId;
    for (int w = 0; w < cfg_.tactic_window_decisions; w++) {
        std::vector<CombatAction> hero_actions(heroes.size());
        for (size_t i = 0; i < heroes.size(); i++) {
            hero_actions[i] = tactic_to_action(tactic, *heroes[i], world);
        }
        step_team_window(world, heroes, hero_actions, opponent_,
                         cfg_.sim_dt, cfg_.action_repeat, team_id);
        if (team_terminal(world, team_id)) break;
    }
}

TacticMcts::TNode* TacticMcts::select_(TNode* node, World& world,
                                        const std::vector<Agent*>& heroes) {
    const int team_id = heroes.front()->unit().teamId;
    while (node->fully_expanded() && !node->is_leaf()) {
        if (team_terminal(world, team_id)) return node;
        TNode* next = tmcts_best_uct_child(node, cfg_.uct_c);
        if (!next) return node;
        step_tactic_(world, heroes, next->action);
        node = next;
    }
    return node;
}

TacticMcts::TNode* TacticMcts::expand_(TNode* node, World& world,
                                        const std::vector<Agent*>& heroes) {
    const int team_id = heroes.front()->unit().teamId;
    if (team_terminal(world, team_id) || node->untried.empty()) return node;

    Tactic t = node->untried.back();
    node->untried.pop_back();

    step_tactic_(world, heroes, t);

    auto child = std::make_unique<TNode>();
    child->action = t;
    child->parent = node;
    child->untried = legal_tactics(heroes, world);
    TNode* raw = child.get();
    node->children.push_back(std::move(child));
    return raw;
}

float TacticMcts::rollout_(World& world, const std::vector<Agent*>& heroes, int team_id) {
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, team_id)
            : TeamHpDeltaEvaluator{}.evaluate(world, team_id);
    };
    if (cfg_.use_leaf_value) return eval();

    int steps = 0;
    while (steps < cfg_.rollout_horizon && !team_terminal(world, team_id)) {
        auto tactics = legal_tactics(heroes, world);
        if (tactics.empty()) break;
        int i = world.randInt(0, static_cast<int>(tactics.size()) - 1);
        step_tactic_(world, heroes, tactics[static_cast<size_t>(i)]);
        steps++;
    }
    return eval();
}

Tactic TacticMcts::search(World& world, const std::vector<Agent*>& heroes) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    if (heroes.empty()) return {};
    const int team_id = heroes.front()->unit().teamId;

    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_) {
        reused = true;
    } else {
        root_ = std::make_unique<TNode>();
        root_->untried = legal_tactics(heroes, world);
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

        TNode* leaf  = select_(root_.get(), world, heroes);
        TNode* child = expand_(leaf, world, heroes);
        float  value = rollout_(world, heroes, team_id);

        // Backprop.
        TNode* node = child;
        while (node) {
            node->visits++;
            node->total_value += value;
            node = node->parent;
        }
    }

    world.restore(saved);

    TNode* best = nullptr;
    int best_visits = -1;
    for (auto& up : root_->children) {
        if (up->visits > best_visits) { best_visits = up->visits; best = up.get(); }
    }

    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = tmcts_count_tree(root_.get());
    stats_.best_mean     = best ? best->mean() : 0.0f;
    stats_.best_visits   = best ? best->visits : 0;
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;

    return best ? best->action : Tactic{};
}

void TacticMcts::advance_root(const Tactic& committed) {
    if (!root_) return;
    std::unique_ptr<TNode> promoted;
    for (auto& up : root_->children) {
        if (up->action == committed) { promoted = std::move(up); break; }
    }
    if (promoted) {
        promoted->parent = nullptr;
        root_ = std::move(promoted);
    } else {
        root_.reset();
    }
}

} // namespace brogameagent::mcts
