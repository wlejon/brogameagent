#pragma once

#include "types.h"
#include "unit.h"
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

namespace brogameagent {

class Agent;
class World;
struct AgentAction;

/// A damage event. Appended to World's event log whenever damage is dealt
/// through World::dealDamage (or indirectly through resolveAttack).
/// Event log is monotonically appended; use your consumed-index cursor
/// to read new events, or call World::clearEvents() between ticks if you
/// are the only consumer.
struct DamageEvent {
    int attackerId;  // Unit::id; may be -1 for world/environmental damage
    int targetId;
    float amount;    // actual HP lost (post-reduction)
    DamageKind kind;
    bool killed;     // true if this damage brought the target from alive to dead
};

/// Game-defined ability. Registered on the World and bound to a Unit slot
/// by storing the ability id in Unit::abilitySlot[slot]. The fn is invoked
/// when the owning agent casts that slot via applyAction/resolveAbility.
///
/// fn signature: (caster, world, targetId) — targetId is whatever the
/// policy/caller passed in (typically an enemy Unit::id, or -1 for self-cast).
/// Library does NOT interpret targetId beyond passing it through.
struct AbilitySpec {
    float cooldown = 1.0f;   // seconds — set on Unit::abilityCooldowns[slot] on cast
    float manaCost = 0.0f;
    float range    = 0.0f;   // 0 = unranged/skipped; else max target distance
    std::function<void(Agent& caster, World& world, int targetId)> fn;
};

/// Container for a set of Agents and shared obstacles. The substrate for
/// perception and observation queries: "who can I see", "nearest enemy", etc.
///
/// The World does not own its agents — register raw pointers and keep them
/// alive externally (typically in an ECS or an array on the game side).
/// Agents must have distinct Unit::id values for attack-target bookkeeping.
class World {
public:
    void addAgent(Agent* agent);
    void removeAgent(const Agent* agent);
    void addObstacle(const AABB& box);

    /// Advance all agents by dt. Calls Agent::update() on each (scripted
    /// path-following). Policy-driven agents should NOT be tick()'d —
    /// call applyAction() per agent from your own loop.
    void tick(float dt);

    const std::vector<Agent*>& agents() const { return agents_; }
    const std::vector<AABB>& obstacles() const { return obstacles_; }

    /// Returns all living agents on a different team.
    std::vector<Agent*> enemiesOf(const Agent& self) const;

    /// Returns all living agents on the same team (excluding self).
    std::vector<Agent*> alliesOf(const Agent& self) const;

    /// Nearest living enemy, or nullptr. Ignores LOS.
    Agent* nearestEnemy(const Agent& self) const;

    /// Enemies within `range` of self (euclidean 2D). Sorted nearest-first.
    std::vector<Agent*> enemiesInRange(const Agent& self, float range) const;

    /// Lookup by Unit::id. Linear scan — fine for small rosters.
    Agent* findById(int id) const;

    // --- Abilities / combat ---

    /// Register a game-defined ability under a numeric id. Bind it to a unit
    /// slot via Unit::abilitySlot[slot] = id.
    void registerAbility(int abilityId, AbilitySpec spec);

    /// True if an ability id has been registered.
    bool hasAbility(int abilityId) const;

    /// Pointer to the registered ability, or nullptr. Intended for mask
    /// builders and debugging; do not mutate the returned spec.
    const AbilitySpec* abilitySpec(int abilityId) const;

    /// Auto-attack resolution. Returns true iff the attack landed.
    /// Fails on: invalid target, dead target, same team, out of range,
    /// attack cooldown not ready, or attacker dead.
    /// On success: deals Unit::damage as Unit::attackKind, starts cooldown.
    bool resolveAttack(Agent& attacker, int targetId);

    /// Ability cast resolution. `slot` is 0..Unit::MAX_ABILITIES-1.
    /// Fails on: invalid slot, empty slot, unknown ability, cooldown not
    /// ready, insufficient mana, out of range (if AbilitySpec::range > 0),
    /// or dead caster. On success: invokes fn, deducts mana, starts cooldown.
    bool resolveAbility(Agent& caster, int slot, int targetId);

    /// Convenience step: movement/aim via Agent::applyAction, then resolve
    /// attack (if action.attackTargetId >= 0) and ability (if useAbilityId
    /// >= 0). Failures on attack/ability are silent — inspect return values
    /// of the lower-level resolve* methods if you need that feedback.
    void applyAction(Agent& agent, const AgentAction& action, float dt);

    // --- Damage logging ---

    /// Deal damage through the world so it gets logged. Ability fns should
    /// use this rather than calling Unit::takeDamage directly if they want
    /// the hit to show up in reward tracking / training telemetry.
    /// Returns the actual HP lost.
    float dealDamage(Agent& attacker, Agent& target, float amount, DamageKind kind);

    /// Same as dealDamage but for environmental / untracked sources (no attacker).
    float dealEnvDamage(Agent& target, float amount, DamageKind kind);

    /// Monotonic event log (appended, never reordered). Held until
    /// clearEvents() is called.
    const std::vector<DamageEvent>& events() const { return events_; }
    void clearEvents();

    // --- Deterministic RNG ---

    /// Seed the per-world PRNG. Use the same seed to reproduce a rollout.
    void seed(uint64_t s);

    /// Access the underlying engine for library code / custom distributions.
    std::mt19937_64& rng() { return engine_; }

    /// Portable [0,1) float draw (does not use std::uniform_real_distribution,
    /// which is implementation-defined across stdlibs).
    float randFloat01();

    /// Uniform float in [lo, hi).
    float randRange(float lo, float hi);

    /// Uniform int in [lo, hi] inclusive.
    int randInt(int lo, int hi);

    /// True with probability p.
    bool chance(float p);

private:
    std::vector<Agent*> agents_;
    std::vector<AABB> obstacles_;
    std::unordered_map<int, AbilitySpec> abilities_;

    std::vector<DamageEvent> events_;

    std::mt19937_64 engine_{0xC0FFEEULL};
};

} // namespace brogameagent
