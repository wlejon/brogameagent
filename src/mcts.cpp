#include "brogameagent/mcts.h"
#include "mcts_internal.h"
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

constexpr MoveVec MOVE_VECS[9] = {
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

// Returns the agent-local stick vector for a direct-steering MoveDir.
// Pathfinding MoveDirs return {0,0} — apply() dispatches those via setTarget
// before the moveX/moveZ path is taken.
MoveVec move_vec(MoveDir d) {
    int i = static_cast<int>(d);
    if (i < 0 || i >= 9) return {0, 0};
    return MOVE_VECS[i];
}

bool is_path_dir(MoveDir d) {
    return d == MoveDir::PathToTarget || d == MoveDir::PathAway;
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

} // namespace

bool is_terminal_for(const Agent& hero, const World& world) {
    if (!hero.unit().alive()) return true;
    for (Agent* a : world.agents()) {
        if (a == &hero) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != hero.unit().teamId) return false;
    }
    return true;
}

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

    if (is_path_dir(action.move_dir)) {
        // Pathfinding-aware motion. Target an aim/flee reference ~4 units
        // away; Agent::setTarget runs A* through the nav grid on significant
        // target changes and Agent::update steers along the current path.
        // The 4u lead matches the open-arena decision window (agent speed
        // ~5u/s × ~0.75s before a new plan overrides).
        int ref_id = attack_target_id;
        if (ref_id < 0) {
            if (Agent* near = find_nearest_enemy(agent, world); near) {
                ref_id = near->unit().id;
            }
        }
        bool moved = false;
        if (ref_id >= 0) {
            if (Agent* t = world.findById(ref_id); t) {
                float dx = t->x() - agent.x();
                float dz = t->z() - agent.z();
                float d  = std::sqrt(dx*dx + dz*dz);
                float sign = (action.move_dir == MoveDir::PathAway) ? -1.0f : 1.0f;
                const float LEAD = 4.0f;
                if (d > 1e-3f) {
                    float wx = agent.x() + sign * (dx / d) * LEAD;
                    float wz = agent.z() + sign * (dz / d) * LEAD;
                    agent.setTarget(wx, wz);
                    agent.update(dt);
                    moved = true;
                }
            }
        }
        // Agent::update() doesn't tick cooldowns (unlike applyAction), so
        // we mirror that side effect here regardless of whether motion
        // actually happened — otherwise attack_slot / ability_slot below
        // fire on stale cooldowns.
        agent.unit().tickCooldowns(dt);
        (void)moved;
    } else {
        MoveVec mv = move_vec(action.move_dir);
        AgentAction aa;
        aa.moveX = mv.x;
        aa.moveZ = mv.z;
        aa.aimYaw = aim_yaw;
        aa.aimPitch = 0.0f;
        aa.attackTargetId = -1;      // resolved separately so ability targeting
        aa.useAbilityId   = -1;      // is independent of attack routing
        agent.applyAction(aa, dt);
    }

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

CombatAction policy_scripted(Agent& self, const World& world) {
    CombatAction a;
    if (!self.unit().alive()) return a;
    Agent* enemy = find_nearest_enemy(self, world);
    if (!enemy) return a;

    float mask[action_mask::TOTAL];
    int   enemy_ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, enemy_ids);

    int nearest_slot = -1;
    for (int k = 0; k < action_mask::N_ENEMY_SLOTS; k++) {
        if (enemy_ids[k] == enemy->unit().id) { nearest_slot = k; break; }
    }

    float dx = enemy->x() - self.x();
    float dz = enemy->z() - self.z();
    float d  = std::sqrt(dx * dx + dz * dz);
    float r  = self.unit().attackRange;
    float hp_frac = self.unit().hp / std::max(1e-6f, self.unit().maxHp);
    bool  cd_ready = self.unit().attackCooldown <= 0.0f;
    bool  in_range = d <= r;

    // Auto-attack when we can — same slot logic as policy_aggressive, but
    // layered on top of the movement decision below.
    if (cd_ready && in_range && nearest_slot >= 0 && mask[nearest_slot] > 0.0f) {
        a.attack_slot = static_cast<int8_t>(nearest_slot);
    }

    // Pick the highest-impact ready ability that's gated in the mask. Slot
    // order here mirrors ai-arena's ability layout (heal=0, fireball=1,
    // beam=2, grenade=3). Prefer offensive casts when we have a target in
    // range; fall back to heal if wounded.
    for (int s = 0; s < action_mask::N_ABILITY_SLOTS; s++) {
        if (mask[action_mask::N_ENEMY_SLOTS + s] <= 0.0f) continue;
        // Heal ourselves when wounded and no in-range threat exists yet.
        if (s == 0 && hp_frac < 0.5f) {
            a.ability_slot = static_cast<int8_t>(s);
            break;
        }
        if (s != 0 && in_range) {
            a.ability_slot = static_cast<int8_t>(s);
            break;
        }
    }

    // Movement: flee if wounded, kite if too close, path in if too far,
    // hold if comfortably in-range and firing.
    if (hp_frac < 0.3f) {
        a.move_dir = MoveDir::PathAway;
    } else if (!in_range) {
        a.move_dir = MoveDir::PathToTarget;
    } else if (d < r * 0.55f) {
        // Strafe perpendicular to close enemy. Pick E or W by agent id parity
        // so opposite flanks emerge naturally instead of everyone mirroring.
        a.move_dir = (self.unit().id & 1) ? MoveDir::E : MoveDir::W;
    } else {
        a.move_dir = a.attack_slot >= 0 ? MoveDir::Hold : MoveDir::PathToTarget;
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

CombatAction ScriptedRollout::choose(Agent& self, World& world) const {
    if (!self.unit().alive()) return {};
    return policy_scripted(self, world);
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
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, hero.unit().id)
            : HpDeltaEvaluator{}.evaluate(world, hero.unit().id);
    };
    if (cfg_.use_leaf_value) return eval();

    int steps = 0;
    while (steps < cfg_.rollout_horizon && !is_terminal_for(hero, world)) {
        CombatAction a = rollout_policy_
            ? rollout_policy_->choose(hero, world)
            : RandomRollout{}.choose(hero, world);
        step_decision_(world, hero, a);
        steps++;
    }
    return eval();
}

void Mcts::backprop_(Node* node, float value) {
    while (node) {
        node->visits++;
        node->total_value += value;
        node = node->parent;
    }
}

void Mcts::ensure_root(World& world, Agent& hero) {
    if (root_) return;
    root_ = std::make_unique<Node>();
    root_->untried = legal_actions(hero, world);
    root_->untried_priors = compute_priors(
        prior_.get(), hero, world, root_->untried);
}

void Mcts::run_iteration(World& world, Agent& hero) {
    ensure_root(world, hero);
    Node* leaf  = select_(root_.get(), world, hero);
    Node* child = expand_(leaf, world, hero);
    float value = rollout_(world, hero);
    backprop_(child, value);
}

CombatAction Mcts::best_action() const {
    if (!root_) return {};
    const Node* best = nullptr;
    int best_visits = -1;
    for (const auto& up : root_->children) {
        if (up->visits > best_visits) {
            best_visits = up->visits;
            best = up.get();
        }
    }
    return best ? best->action : CombatAction{};
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

} // namespace brogameagent::mcts
