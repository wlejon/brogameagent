#include "brogameagent/world.h"
#include "brogameagent/agent.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {

void World::addAgent(Agent* a) {
    agents_.push_back(a);
}

void World::removeAgent(const Agent* a) {
    agents_.erase(std::remove(agents_.begin(), agents_.end(), a), agents_.end());
}

void World::addObstacle(const AABB& box) {
    obstacles_.push_back(box);
}

void World::tick(float dt) {
    for (Agent* a : agents_) {
        if (a->unit().alive()) a->update(dt);
        a->unit().tickCooldowns(dt);
    }
}

std::vector<Agent*> World::enemiesOf(const Agent& self) const {
    std::vector<Agent*> out;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != self.unit().teamId) out.push_back(a);
    }
    return out;
}

std::vector<Agent*> World::alliesOf(const Agent& self) const {
    std::vector<Agent*> out;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) out.push_back(a);
    }
    return out;
}

static float distSq2D(const Agent& a, const Agent& b) {
    float dx = a.x() - b.x();
    float dz = a.z() - b.z();
    return dx * dx + dz * dz;
}

Agent* World::nearestEnemy(const Agent& self) const {
    Agent* best = nullptr;
    float bestD2 = 1e30f;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float d2 = distSq2D(self, *a);
        if (d2 < bestD2) { bestD2 = d2; best = a; }
    }
    return best;
}

std::vector<Agent*> World::enemiesInRange(const Agent& self, float range) const {
    float r2 = range * range;
    std::vector<std::pair<float, Agent*>> scored;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float d2 = distSq2D(self, *a);
        if (d2 <= r2) scored.push_back({d2, a});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& p, const auto& q) { return p.first < q.first; });
    std::vector<Agent*> out;
    out.reserve(scored.size());
    for (auto& p : scored) out.push_back(p.second);
    return out;
}

Agent* World::findById(int id) const {
    for (Agent* a : agents_) {
        if (a->unit().id == id) return a;
    }
    return nullptr;
}

void World::registerAbility(int abilityId, AbilitySpec spec) {
    abilities_[abilityId] = std::move(spec);
}

bool World::hasAbility(int abilityId) const {
    return abilities_.find(abilityId) != abilities_.end();
}

bool World::resolveAttack(Agent& attacker, int targetId) {
    if (!attacker.unit().alive()) return false;
    if (attacker.unit().attackCooldown > 0.0f) return false;
    Agent* target = findById(targetId);
    if (!target || !target->unit().alive()) return false;
    if (target->unit().teamId == attacker.unit().teamId) return false;

    float dx = target->x() - attacker.x();
    float dz = target->z() - attacker.z();
    float dist2 = dx * dx + dz * dz;
    float r = attacker.unit().attackRange;
    if (dist2 > r * r) return false;

    target->unit().takeDamage(attacker.unit().damage, attacker.unit().attackKind);
    // attacksPerSec == 0 ⇒ no auto-attack (cooldown stays 0, but this resolve still
    // deals damage; guard against divide-by-zero).
    float aps = attacker.unit().attacksPerSec;
    attacker.unit().attackCooldown = (aps > 0.0f) ? (1.0f / aps) : 0.0f;
    return true;
}

bool World::resolveAbility(Agent& caster, int slot, int targetId) {
    if (!caster.unit().alive()) return false;
    if (slot < 0 || slot >= Unit::MAX_ABILITIES) return false;
    int abilityId = caster.unit().abilitySlot[slot];
    if (abilityId < 0) return false;
    auto it = abilities_.find(abilityId);
    if (it == abilities_.end()) return false;
    const AbilitySpec& spec = it->second;

    if (caster.unit().abilityCooldowns[slot] > 0.0f) return false;
    if (caster.unit().mana < spec.manaCost) return false;

    if (spec.range > 0.0f && targetId >= 0) {
        Agent* target = findById(targetId);
        if (!target) return false;
        float dx = target->x() - caster.x();
        float dz = target->z() - caster.z();
        if (dx * dx + dz * dz > spec.range * spec.range) return false;
    }

    caster.unit().mana -= spec.manaCost;
    caster.unit().abilityCooldowns[slot] = spec.cooldown;
    if (spec.fn) spec.fn(caster, *this, targetId);
    return true;
}

void World::applyAction(Agent& agent, const AgentAction& action, float dt) {
    agent.applyAction(action, dt);
    if (action.attackTargetId >= 0) {
        resolveAttack(agent, action.attackTargetId);
    }
    if (action.useAbilityId >= 0) {
        resolveAbility(agent, action.useAbilityId, action.attackTargetId);
    }
}

} // namespace brogameagent
