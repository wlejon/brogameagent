#include "brogameagent/mcts.h"

#include "brogameagent/action_mask.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <thread>

namespace brogameagent::mcts {

// ─── CombatAction: move dir → local-frame vector ───────────────────────────

namespace {

struct MoveVec { float x; float z; };
constexpr float S = 0.70710678f;  // sin(45°) = cos(45°)

constexpr MoveVec MOVE_VECS[static_cast<int>(MoveDir::COUNT)] = {
    /*Hold*/ { 0,  0},
    /*N   */ { 0, -1},
    /*NE  */ { S, -S},
    /*E   */ { 1,  0},
    /*SE  */ { S,  S},
    /*S   */ { 0,  1},
    /*SW  */ {-S,  S},
    /*W   */ {-1,  0},
    /*NW  */ {-S, -S},
};

MoveVec move_vec(MoveDir d) {
    int i = static_cast<int>(d);
    if (i < 0 || i >= static_cast<int>(MoveDir::COUNT)) return {0, 0};
    return MOVE_VECS[i];
}

bool is_enemy_of(const Agent& a, const Agent& b) {
    return a.unit().alive() && b.unit().alive()
        && a.unit().teamId != b.unit().teamId;
}

Agent* find_nearest_enemy(const Agent& self, const World& world) {
    Agent* best = nullptr;
    float best_d2 = std::numeric_limits<float>::infinity();
    for (Agent* a : world.agents()) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float dx = a->x() - self.x();
        float dz = a->z() - self.z();
        float d2 = dx * dx + dz * dz;
        if (d2 < best_d2) { best_d2 = d2; best = a; }
    }
    return best;
}

float aim_yaw_toward(float fromX, float fromZ, float toX, float toZ) {
    float dx = toX - fromX;
    float dz = toZ - fromZ;
    // FPS convention: yaw=0 faces -Z; yaw increases toward +X.
    return std::atan2(dx, -dz);
}

bool is_terminal_for(const Agent& hero, const World& world) {
    if (!hero.unit().alive()) return true;
    for (Agent* a : world.agents()) {
        if (a == &hero) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != hero.unit().teamId) return false;
    }
    return true;
}

} // namespace

// ─── legal_actions / apply ─────────────────────────────────────────────────

std::vector<CombatAction> legal_actions(const Agent& self, const World& world) {
    std::vector<CombatAction> out;

    // Dead hero: only the no-op action. Keeps select/expand well-defined
    // but nothing interesting gets tried.
    if (!self.unit().alive()) {
        out.push_back({});
        return out;
    }

    float mask[action_mask::TOTAL];
    int   enemy_ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, enemy_ids);

    // Collect legal attack slots and ability slots (include -1 = "none" in each).
    std::vector<int8_t> attack_opts{-1};
    for (int k = 0; k < action_mask::N_ENEMY_SLOTS; k++) {
        if (mask[k] > 0.0f) attack_opts.push_back(static_cast<int8_t>(k));
    }
    std::vector<int8_t> ability_opts{-1};
    for (int s = 0; s < action_mask::N_ABILITY_SLOTS; s++) {
        if (mask[action_mask::N_ENEMY_SLOTS + s] > 0.0f) {
            ability_opts.push_back(static_cast<int8_t>(s));
        }
    }

    out.reserve(static_cast<size_t>(MoveDir::COUNT)
                * attack_opts.size()
                * ability_opts.size());
    for (int d = 0; d < static_cast<int>(MoveDir::COUNT); d++) {
        for (int8_t att : attack_opts) {
            for (int8_t ab : ability_opts) {
                out.push_back({static_cast<MoveDir>(d), att, ab});
            }
        }
    }
    return out;
}

void apply(Agent& agent, World& world, const CombatAction& action, float dt) {
    if (!agent.unit().alive()) {
        agent.unit().tickCooldowns(dt);
        return;
    }

    // Resolve slot-based targets → concrete enemy ids via action_mask, which
    // guarantees the same slot ordering used by legal_actions / the observer.
    float mask[action_mask::TOTAL];
    int   enemy_ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(agent, world, mask, enemy_ids);

    int attack_target_id = -1;
    if (action.attack_slot >= 0 && action.attack_slot < action_mask::N_ENEMY_SLOTS) {
        attack_target_id = enemy_ids[action.attack_slot];
    }

    // Ability target: reuse the attack target if set; else the nearest enemy
    // (slot 0); else self-cast (-1). Good enough for M1 — per-slot ability
    // targeting can be added when an ability cares about aiming separately.
    int ability_target_id = attack_target_id;
    if (ability_target_id < 0 && action.ability_slot >= 0) {
        ability_target_id = enemy_ids[0];
    }

    // Aim: toward whichever target is meaningful, else nearest enemy, else
    // hold current aim. Movement direction is always the policy's choice.
    float aim_yaw = agent.aimYaw();
    int aim_target_id = attack_target_id >= 0 ? attack_target_id
                      : ability_target_id >= 0 ? ability_target_id
                      : -1;
    if (aim_target_id < 0) {
        if (Agent* near = find_nearest_enemy(agent, world); near) {
            aim_target_id = near->unit().id;
        }
    }
    if (aim_target_id >= 0) {
        if (Agent* t = world.findById(aim_target_id); t) {
            aim_yaw = aim_yaw_toward(agent.x(), agent.z(), t->x(), t->z());
        }
    }

    // Lock movement-facing to the aim direction so that MoveDir::N (local
    // -Z, "forward") means "toward the aim target." Without this, yaw_ is
    // whatever the velocity-direction integrator left it at, and a fixed
    // MoveDir like E causes yaw_ to cycle (E→S→W→N), tracing a circle
    // instead of a straight strafe. Resetting yaw_ at the start of every
    // decision keeps the policy's frame and the integrator's frame aligned.
    agent.setYaw(aim_yaw);

    MoveVec mv = move_vec(action.move_dir);

    AgentAction aa;
    aa.moveX = mv.x;
    aa.moveZ = mv.z;
    aa.aimYaw = aim_yaw;
    aa.aimPitch = 0.0f;
    aa.attackTargetId = -1;      // resolved separately so ability targeting
    aa.useAbilityId   = -1;      // is independent of attack routing
    agent.applyAction(aa, dt);

    if (action.attack_slot >= 0 && attack_target_id >= 0) {
        world.resolveAttack(agent, attack_target_id);
    }
    if (action.ability_slot >= 0) {
        world.resolveAbility(agent, action.ability_slot, ability_target_id);
    }
}

// ─── Opponent policies ─────────────────────────────────────────────────────

CombatAction policy_idle(Agent& /*self*/, const World& /*world*/) {
    return {};
}

CombatAction policy_aggressive(Agent& self, const World& world) {
    CombatAction a;
    Agent* enemy = find_nearest_enemy(self, world);
    if (!enemy) return a;

    // Steer toward the enemy in local frame: +X right, -Z forward. Since we
    // will be re-aiming at the target inside apply(), local forward IS toward
    // the enemy — so always charge with "N" (forward). That keeps this
    // policy independent of the hero's current yaw.
    a.move_dir = MoveDir::N;

    // Attack if in range + cooldown ready.
    float dx = enemy->x() - self.x();
    float dz = enemy->z() - self.z();
    float d2 = dx * dx + dz * dz;
    float r  = self.unit().attackRange;
    if (self.unit().attackCooldown <= 0.0f && d2 <= r * r) {
        // Need the slot index of this enemy in the action mask. Cheapest path
        // is rebuilding the mask here; policy is not on the hot path.
        float mask[action_mask::TOTAL];
        int   enemy_ids[action_mask::N_ENEMY_SLOTS];
        action_mask::build(self, world, mask, enemy_ids);
        for (int k = 0; k < action_mask::N_ENEMY_SLOTS; k++) {
            if (enemy_ids[k] == enemy->unit().id && mask[k] > 0.0f) {
                a.attack_slot = static_cast<int8_t>(k);
                break;
            }
        }
    }
    return a;
}

// ─── Evaluator ─────────────────────────────────────────────────────────────

float HpDeltaEvaluator::evaluate(const World& world, int heroId) const {
    const Agent* hero = world.findById(heroId);
    if (!hero) return 0.0f;

    float hero_frac = hero->unit().alive()
        ? hero->unit().hp / std::max(1e-6f, hero->unit().maxHp)
        : 0.0f;

    float enemy_frac_sum = 0.0f;
    int   enemy_count    = 0;
    bool  any_enemy_alive = false;
    for (Agent* a : world.agents()) {
        if (a == hero) continue;
        if (a->unit().teamId == hero->unit().teamId) continue;
        enemy_count++;
        if (a->unit().alive()) any_enemy_alive = true;
        enemy_frac_sum += a->unit().alive()
            ? a->unit().hp / std::max(1e-6f, a->unit().maxHp)
            : 0.0f;
    }

    // Terminal shortcuts so win/loss always dominate heuristic.
    if (!hero->unit().alive()) return -1.0f;
    if (!any_enemy_alive)      return  1.0f;

    float enemy_avg = enemy_count > 0
        ? enemy_frac_sum / static_cast<float>(enemy_count)
        : 0.0f;
    float delta = hero_frac - enemy_avg;        // in [-1, 1]
    if (delta < -1.0f) delta = -1.0f;
    if (delta >  1.0f) delta =  1.0f;
    return delta;
}

// ─── Rollout policy ────────────────────────────────────────────────────────

CombatAction RandomRollout::choose(Agent& self, World& world) const {
    auto acts = legal_actions(self, world);
    if (acts.empty()) return {};
    // Uniform pick via the world's own PRNG so rollouts are reproducible
    // alongside the sim's other random draws under a common seed.
    int i = world.randInt(0, static_cast<int>(acts.size()) - 1);
    return acts[static_cast<size_t>(i)];
}

CombatAction AggressiveRollout::choose(Agent& self, World& world) const {
    if (!self.unit().alive()) return {};
    return policy_aggressive(self, world);
}

// ─── Priors ────────────────────────────────────────────────────────────────

std::vector<float> UniformPrior::score(
    const Agent& /*self*/, const World& /*world*/,
    const std::vector<CombatAction>& actions) const
{
    return std::vector<float>(actions.size(), 1.0f);
}

std::vector<float> AttackBiasPrior::score(
    const Agent& /*self*/, const World& /*world*/,
    const std::vector<CombatAction>& actions) const
{
    std::vector<float> w(actions.size(), 1.0f);
    for (size_t i = 0; i < actions.size(); i++) {
        if (actions[i].attack_slot  >= 0) w[i] = 4.0f;
        else if (actions[i].ability_slot >= 0) w[i] = 2.0f;
        // else move-only: keep 1.0
    }
    return w;
}

namespace {

/// Compute normalized priors for `actions` via `prior` (or uniform if null).
/// Output is a probability distribution summing to 1; if all weights are
/// zero or input is empty, returns uniform.
std::vector<float> compute_priors(const IPrior* prior,
                                   const Agent& self, const World& world,
                                   const std::vector<CombatAction>& actions) {
    const size_t n = actions.size();
    if (n == 0) return {};
    std::vector<float> w = prior ? prior->score(self, world, actions)
                                  : std::vector<float>(n, 1.0f);
    if (w.size() != n) w.assign(n, 1.0f);
    float sum = 0.0f;
    for (float v : w) { if (v < 0.0f) v = 0.0f; sum += v; }
    if (sum <= 0.0f) {
        std::fill(w.begin(), w.end(), 1.0f / static_cast<float>(n));
        return w;
    }
    for (float& v : w) v = (v < 0.0f ? 0.0f : v) / sum;
    return w;
}

} // namespace

// ─── Mcts engine ───────────────────────────────────────────────────────────

namespace {

Node* best_uct_child(Node* node, float c) {
    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float ln_n = std::log(static_cast<float>(std::max(1, node->visits)));
    for (auto& up : node->children) {
        Node* ch = up.get();
        if (ch->visits == 0) return ch;   // always explore unvisited first
        float exploit = ch->mean();
        float explore = c * std::sqrt(ln_n / static_cast<float>(ch->visits));
        float score = exploit + explore;
        if (score > best_score) { best_score = score; best = ch; }
    }
    return best;
}

/// PUCT selection: score = Q + c * P * √N_parent / (1 + n_child). Used when
/// cfg_.prior_c > 0. Does NOT short-circuit on visits==0 because prior
/// probability already distinguishes between unvisited children.
Node* best_puct_child(Node* node, float c) {
    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float sqrt_N = std::sqrt(static_cast<float>(std::max(1, node->visits)));
    for (auto& up : node->children) {
        Node* ch = up.get();
        float q = ch->mean();
        float u = c * ch->prior_p * sqrt_N / (1.0f + static_cast<float>(ch->visits));
        float score = q + u;
        if (score > best_score) { best_score = score; best = ch; }
    }
    return best;
}

int count_tree(const Node* n) {
    if (!n) return 0;
    int total = 1;
    for (const auto& ch : n->children) total += count_tree(ch.get());
    return total;
}

} // namespace

void Mcts::step_decision_(World& world, Agent& hero, const CombatAction& hero_action) {
    // One opponent decision per window, held across sub-ticks (matches the
    // hero's action-persistence model — keeps both sides on the same
    // decision cadence).
    std::vector<std::pair<Agent*, CombatAction>> others;
    others.reserve(world.agents().size());
    for (Agent* a : world.agents()) {
        if (a == &hero) continue;
        CombatAction act{};
        if (a->unit().alive() && opponent_) act = opponent_(*a, world);
        others.emplace_back(a, act);
    }

    const float dt = cfg_.sim_dt;
    for (int t = 0; t < cfg_.action_repeat; t++) {
        mcts::apply(hero, world, hero_action, dt);
        for (auto& [a, act] : others) {
            mcts::apply(*a, world, act, dt);
        }
        world.stepProjectiles(dt);
        world.cullProjectiles();
        if (is_terminal_for(hero, world)) break;
    }
}

Node* Mcts::select_(Node* node, World& world, Agent& hero) {
    // Descend until we reach a node where we should expand a new child.
    // Expansion is triggered by:
    //   - leaf (no children yet), OR
    //   - PW disabled and an untried action exists, OR
    //   - PW enabled and children.size() < ceil(visits^pw_alpha) with untried
    //     actions available.
    // Otherwise we descend via UCT into the existing children.
    auto wants_expansion = [&](const Node* n) {
        if (n->children.empty()) return true;
        if (n->untried.empty())  return false;
        if (cfg_.pw_alpha <= 0.0f) return true;
        float threshold = std::ceil(std::pow(
            static_cast<float>(std::max(1, n->visits)),
            cfg_.pw_alpha));
        return static_cast<float>(n->children.size()) < threshold;
    };

    while (!wants_expansion(node)) {
        if (is_terminal_for(hero, world)) return node;
        Node* next = cfg_.prior_c > 0.0f
            ? best_puct_child(node, cfg_.prior_c)
            : best_uct_child(node, cfg_.uct_c);
        if (!next) return node;
        step_decision_(world, hero, next->action);
        node = next;
    }
    return node;
}

Node* Mcts::expand_(Node* node, World& world, Agent& hero) {
    if (is_terminal_for(hero, world) || node->untried.empty()) return node;

    // Pop from the back for O(1). Under PUCT, each untried slot carries its
    // normalized prior probability; pop both aligned.
    CombatAction a = node->untried.back();
    node->untried.pop_back();
    float child_p = 1.0f;
    if (!node->untried_priors.empty()) {
        child_p = node->untried_priors.back();
        node->untried_priors.pop_back();
    }

    step_decision_(world, hero, a);

    auto child = std::make_unique<Node>();
    child->action  = a;
    child->parent  = node;
    child->prior_p = child_p;
    child->untried = legal_actions(hero, world);
    child->untried_priors = compute_priors(
        prior_.get(), hero, world, child->untried);
    Node* raw = child.get();
    node->children.push_back(std::move(child));
    return raw;
}

float Mcts::rollout_(World& world, Agent& hero) {
    int steps = 0;
    while (steps < cfg_.rollout_horizon && !is_terminal_for(hero, world)) {
        CombatAction a = rollout_policy_
            ? rollout_policy_->choose(hero, world)
            : RandomRollout{}.choose(hero, world);
        step_decision_(world, hero, a);
        steps++;
    }
    return evaluator_
        ? evaluator_->evaluate(world, hero.unit().id)
        : HpDeltaEvaluator{}.evaluate(world, hero.unit().id);
}

void Mcts::backprop_(Node* node, float value) {
    while (node) {
        node->visits++;
        node->total_value += value;
        node = node->parent;
    }
}

CombatAction Mcts::search(World& world, Agent& hero) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    // Save caller's state so search is side-effect free on the live world.
    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_) {
        reused = true;
        // When resuming, we trust the prior tree's legal-action list. The
        // caller is responsible for calling reset_tree() if the world has
        // changed beyond what advance_root can track.
    } else {
        root_ = std::make_unique<Node>();
        root_->untried = legal_actions(hero, world);
        root_->untried_priors = compute_priors(
            prior_.get(), hero, world, root_->untried);
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
        // Per-iteration seed so rollouts diverge while staying reproducible
        // under a fixed cfg_.seed. Offset by prior root visits so resumed
        // searches don't re-run identical rollouts from iteration 0.
        world.seed(cfg_.seed
                   + static_cast<uint64_t>(it)
                   + static_cast<uint64_t>(root_->visits) * 7919ull);

        Node* leaf  = select_(root_.get(), world, hero);
        Node* child = expand_(leaf, world, hero);
        float value = rollout_(world, hero);
        backprop_(child, value);
    }

    world.restore(saved);

    // Pick by visit count — more robust than mean under UCT because mean has
    // high variance on low-visit children.
    Node* best = nullptr;
    int   best_visits = -1;
    for (auto& up : root_->children) {
        if (up->visits > best_visits) { best_visits = up->visits; best = up.get(); }
    }

    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = count_tree(root_.get());
    stats_.best_mean     = best ? best->mean() : 0.0f;
    stats_.best_visits   = best ? best->visits : 0;
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;

    return best ? best->action : CombatAction{};
}

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
    return evaluator_
        ? evaluator_->evaluate(world, hero.unit().id)
        : HpDeltaEvaluator{}.evaluate(world, hero.unit().id);
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

void Mcts::advance_root(const CombatAction& committed) {
    if (!root_) return;
    std::unique_ptr<Node> promoted;
    for (auto& up : root_->children) {
        if (up->action == committed) {
            promoted = std::move(up);
            break;
        }
    }
    if (promoted) {
        promoted->parent = nullptr;
        root_ = std::move(promoted);
    } else {
        // Committed action wasn't explored deeply enough to have a child —
        // drop the tree and let the next search build fresh.
        root_.reset();
    }
}

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

} // namespace

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
            if (slot >= 0) {
                a.attack_slot = static_cast<int8_t>(slot);
                a.move_dir = MoveDir::Hold;   // in range — stop and shoot
            } else {
                a.move_dir = MoveDir::N;      // out of range — close the gap
                // Aim handled inside mcts::apply via nearest-enemy fallback;
                // to be truly robust we'd aim at `target` explicitly. That
                // requires threading the enemy_id through apply — leaving
                // that for a future tactic-targeting refactor.
            }
            break;
        }
        case TacticKind::Scatter: {
            // Attack nearest in-range (slot 0), close otherwise.
            float mask[action_mask::TOTAL];
            int   enemy_ids[action_mask::N_ENEMY_SLOTS];
            action_mask::build(hero, world, mask, enemy_ids);
            if (mask[0] > 0.0f) {
                a.attack_slot = 0;
            } else if (enemy_ids[0] >= 0) {
                a.move_dir = MoveDir::N;
            }
            break;
        }
        case TacticKind::Retreat: {
            // mcts::apply aims the agent at the nearest enemy, so local -Z
            // (N) is toward them and +Z (S) is away. No attacks.
            a.move_dir = MoveDir::S;
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
    int steps = 0;
    while (steps < cfg_.rollout_horizon && !team_terminal(world, team_id)) {
        auto tactics = legal_tactics(heroes, world);
        if (tactics.empty()) break;
        int i = world.randInt(0, static_cast<int>(tactics.size()) - 1);
        step_tactic_(world, heroes, tactics[static_cast<size_t>(i)]);
        steps++;
    }
    return evaluator_
        ? evaluator_->evaluate(world, team_id)
        : TeamHpDeltaEvaluator{}.evaluate(world, team_id);
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

// ─── TacticPrior ───────────────────────────────────────────────────────────

std::vector<float> TacticPrior::score(
    const Agent& self, const World& world,
    const std::vector<CombatAction>& actions) const
{
    std::vector<float> w(actions.size(), other_weight_);
    if (!self.unit().alive() || actions.empty()) return w;
    CombatAction target = tactic_to_action(tactic_, self, world);
    for (size_t i = 0; i < actions.size(); i++) {
        if (actions[i] == target) w[i] = match_weight_;
    }
    return w;
}


// ─── LayeredPlanner ────────────────────────────────────────────────────────

void LayeredPlanner::reset() {
    tactic_mcts_.reset_tree();
    fine_mcts_.reset_tree();
    committed_tactic_ = Tactic{};
    windows_left_ = 0;
    stats_ = Stats{};
}

TeamMcts::JointAction LayeredPlanner::decide(
    World& world, const std::vector<Agent*>& heroes)
{
    stats_.replanned_this_call = false;

    // Configure the tactic layer lazily on first use (or on any config change
    // — cheap to re-apply every call since setters just move shared_ptrs).
    tactic_mcts_.set_config(cfg_.tactic_cfg);
    if (evaluator_)      tactic_mcts_.set_evaluator(evaluator_);
    if (opponent_)       tactic_mcts_.set_opponent_policy(opponent_);

    fine_mcts_.set_config(cfg_.fine_cfg);
    if (evaluator_)      fine_mcts_.set_evaluator(evaluator_);
    if (rollout_policy_) fine_mcts_.set_rollout_policy(rollout_policy_);
    if (opponent_)       fine_mcts_.set_opponent_policy(opponent_);

    // (Re)plan the tactic when the current commitment has expired. Tree is
    // reset each time because world state has moved on between calls and the
    // prior tree's stats were collected for a stale snapshot.
    if (windows_left_ <= 0) {
        tactic_mcts_.reset_tree();
        committed_tactic_ = tactic_mcts_.search(world, heroes);
        stats_.tactic_stats = tactic_mcts_.last_stats();
        windows_left_ = std::max(1, cfg_.tactic_cfg.tactic_window_decisions);
        stats_.replanned_this_call = true;
    }

    // Bias the fine search toward the committed tactic via TacticPrior.
    if (!prior_) prior_ = std::make_shared<TacticPrior>();
    prior_->set_tactic(committed_tactic_);
    fine_mcts_.set_prior(prior_);

    fine_mcts_.reset_tree();
    TeamMcts::JointAction joint = fine_mcts_.search(world, heroes);
    stats_.fine_stats = fine_mcts_.last_stats();

    windows_left_--;
    stats_.committed_tactic      = committed_tactic_;
    stats_.windows_until_replan  = windows_left_;
    return joint;
}

} // namespace brogameagent::mcts
