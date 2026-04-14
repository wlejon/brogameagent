#include "brogameagent/mcts.h"

#include "brogameagent/action_mask.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
    while (node->fully_expanded() && !node->is_leaf()) {
        if (is_terminal_for(hero, world)) return node;
        Node* next = best_uct_child(node, cfg_.uct_c);
        if (!next) return node;
        step_decision_(world, hero, next->action);
        node = next;
    }
    return node;
}

Node* Mcts::expand_(Node* node, World& world, Agent& hero) {
    if (is_terminal_for(hero, world) || node->untried.empty()) return node;

    // Pop from the back for O(1); order doesn't matter for correctness since
    // UCT's unvisited-first rule gives every untried action equal priority.
    CombatAction a = node->untried.back();
    node->untried.pop_back();

    step_decision_(world, hero, a);

    auto child = std::make_unique<Node>();
    child->action  = a;
    child->parent  = node;
    child->untried = legal_actions(hero, world);
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
    // Save caller's state so search is side-effect free on the live world.
    WorldSnapshot saved = world.snapshot();

    root_ = std::make_unique<Node>();
    root_->untried = legal_actions(hero, world);

    for (int it = 0; it < cfg_.iterations; it++) {
        world.restore(saved);
        // Per-iteration seed so rollouts diverge while staying reproducible
        // under a fixed cfg_.seed.
        world.seed(cfg_.seed + static_cast<uint64_t>(it));

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

    stats_.iterations    = cfg_.iterations;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = count_tree(root_.get());
    stats_.best_mean     = best ? best->mean() : 0.0f;
    stats_.best_visits   = best ? best->visits : 0;

    return best ? best->action : CombatAction{};
}

} // namespace brogameagent::mcts
