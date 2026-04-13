#include "brogameagent/world.h"
#include "brogameagent/agent.h"

#include <algorithm>
#include <cmath>
#include <sstream>

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
        applyDotHot(*a, dt);
    }
    stepProjectiles(dt);
    cullProjectiles();
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

const AbilitySpec* World::abilitySpec(int abilityId) const {
    auto it = abilities_.find(abilityId);
    return (it == abilities_.end()) ? nullptr : &it->second;
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

    // Effective attacks-per-sec drives cooldown regardless of hit/miss.
    float aps = attacker.unit().effectiveAttacksPerSec();
    attacker.unit().attackCooldown = (aps > 0.0f) ? (1.0f / aps) : 0.0f;

    // Stealth dodge: roll once per attack against the target's stealth chance.
    float dodge = target->unit().stealthChance;
    if (dodge > 0.0f && randFloat01() < dodge) {
        return false;  // attack swung but missed; cooldown still consumed
    }

    dealDamage(attacker, *target,
               attacker.unit().effectiveDamage(),
               attacker.unit().attackKind);
    return true;
}

float World::dealDamage(Agent& attacker, Agent& target, float amount, DamageKind kind) {
    bool wasAlive = target.unit().alive();
    float dealt = target.unit().takeDamage(amount, kind);
    if (dealt > 0.0f) {
        events_.push_back(DamageEvent{
            attacker.unit().id,
            target.unit().id,
            dealt,
            kind,
            wasAlive && !target.unit().alive()
        });
    }
    return dealt;
}

void World::applyDotHot(Agent& target, float dt) {
    Unit& u = target.unit();
    // DoT first (can kill); HoT heals only living units.
    if (u.alive() && u.dotRemaining > 0.0f && u.dotDps > 0.0f) {
        float chunk = dt;
        if (chunk > u.dotRemaining) chunk = u.dotRemaining;
        float amount = u.dotDps * chunk;
        bool wasAlive = u.alive();
        float dealt = u.takeDamage(amount, u.dotKind);
        if (dealt > 0.0f) {
            events_.push_back(DamageEvent{
                u.dotSourceId,
                u.id,
                dealt,
                u.dotKind,
                wasAlive && !u.alive()
            });
        }
        u.dotRemaining -= dt;
        if (u.dotRemaining <= 0.0f) {
            u.dotRemaining = 0.0f;
            u.dotDps = 0.0f;
            u.dotSourceId = -1;
        }
    }
    if (u.alive() && u.hotRemaining > 0.0f && u.hotRate > 0.0f) {
        float chunk = dt;
        if (chunk > u.hotRemaining) chunk = u.hotRemaining;
        u.hp += u.hotRate * chunk;
        if (u.hp > u.maxHp) u.hp = u.maxHp;
        u.hotRemaining -= dt;
        if (u.hotRemaining <= 0.0f) {
            u.hotRemaining = 0.0f;
            u.hotRate = 0.0f;
        }
    }
}

float World::dealEnvDamage(Agent& target, float amount, DamageKind kind) {
    bool wasAlive = target.unit().alive();
    float dealt = target.unit().takeDamage(amount, kind);
    if (dealt > 0.0f) {
        events_.push_back(DamageEvent{
            -1,
            target.unit().id,
            dealt,
            kind,
            wasAlive && !target.unit().alive()
        });
    }
    return dealt;
}

void World::clearEvents() {
    events_.clear();
}

void World::seed(uint64_t s) {
    engine_.seed(s);
}

float World::randFloat01() {
    // Top 24 bits of the 64-bit engine output mapped to [0,1).
    uint64_t r = engine_();
    return static_cast<float>(r >> 40) * (1.0f / 16777216.0f);
}

float World::randRange(float lo, float hi) {
    return lo + (hi - lo) * randFloat01();
}

int World::randInt(int lo, int hi) {
    if (hi <= lo) return lo;
    uint64_t span = static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(engine_() % span);
}

bool World::chance(float p) {
    return randFloat01() < p;
}

int World::spawnProjectile(const Projectile& proto) {
    Projectile p = proto;
    p.id = nextProjectileId_++;
    p.alive = true;
    projectiles_.push_back(p);
    return p.id;
}

void World::stepProjectiles(float dt) {
    for (Projectile& p : projectiles_) {
        if (!p.alive) continue;

        // Homing: re-steer toward target if it's still alive.
        if (p.targetId >= 0) {
            Agent* t = findById(p.targetId);
            if (t && t->unit().alive()) {
                float dx = t->x() - p.x;
                float dz = t->z() - p.z;
                float d = std::sqrt(dx * dx + dz * dz);
                if (d > 1e-4f) {
                    p.vx = (dx / d) * p.speed;
                    p.vz = (dz / d) * p.speed;
                }
            } else {
                // Lost target — continue as skillshot on last velocity.
                p.targetId = -1;
            }
        }

        p.x += p.vx * dt;
        p.z += p.vz * dt;
        p.remainingLife -= dt;
        if (p.remainingLife <= 0.0f) {
            p.alive = false;
            continue;
        }

        // Resolve collisions based on mode.
        switch (p.mode) {
            case ProjectileMode::Single: {
                for (Agent* a : agents_) {
                    if (!a->unit().alive()) continue;
                    if (a->unit().teamId == p.teamId) continue;
                    float dx = a->x() - p.x;
                    float dz = a->z() - p.z;
                    float hitR = p.radius + a->unit().radius;
                    if (dx * dx + dz * dz <= hitR * hitR) {
                        applyProjectileHit_(p, *a);
                        p.alive = false;
                        break;
                    }
                }
                break;
            }
            case ProjectileMode::Pierce: {
                for (Agent* a : agents_) {
                    if (!a->unit().alive()) continue;
                    if (a->unit().teamId == p.teamId) continue;
                    if (pierceAlreadyHit_(p, a->unit().id)) continue;
                    float dx = a->x() - p.x;
                    float dz = a->z() - p.z;
                    float hitR = p.radius + a->unit().radius;
                    if (dx * dx + dz * dz > hitR * hitR) continue;

                    applyProjectileHit_(p, *a);
                    pierceRemember_(p, a->unit().id);

                    if (p.maxHits > 0 && p.hitCount >= p.maxHits) {
                        p.alive = false;
                        break;
                    }
                }
                break;
            }
            case ProjectileMode::AoE: {
                Agent* impact = nullptr;
                for (Agent* a : agents_) {
                    if (!a->unit().alive()) continue;
                    if (a->unit().teamId == p.teamId) continue;
                    float dx = a->x() - p.x;
                    float dz = a->z() - p.z;
                    float hitR = p.radius + a->unit().radius;
                    if (dx * dx + dz * dz <= hitR * hitR) { impact = a; break; }
                }
                if (impact) {
                    float ex = impact->x();
                    float ez = impact->z();
                    float r2 = p.splashRadius * p.splashRadius;
                    for (Agent* a : agents_) {
                        if (!a->unit().alive()) continue;
                        if (a->unit().teamId == p.teamId) continue;
                        float dx = a->x() - ex;
                        float dz = a->z() - ez;
                        if (dx * dx + dz * dz <= r2) {
                            applyProjectileHit_(p, *a);
                        }
                    }
                    p.alive = false;
                }
                break;
            }
        }
    }
}

bool World::pierceAlreadyHit_(const Projectile& p, int unitId) {
    for (int i = 0; i < p.hitCount; i++) {
        if (p.hitIds[i] == unitId) return true;
    }
    return false;
}

void World::pierceRemember_(Projectile& p, int unitId) {
    if (p.hitCount < Projectile::MAX_PIERCE_MEMORY) {
        p.hitIds[p.hitCount++] = unitId;
    } else {
        for (int i = 1; i < Projectile::MAX_PIERCE_MEMORY; i++)
            p.hitIds[i - 1] = p.hitIds[i];
        p.hitIds[Projectile::MAX_PIERCE_MEMORY - 1] = unitId;
    }
}

WorldSnapshot World::snapshot() const {
    WorldSnapshot s;
    s.agents.reserve(agents_.size());
    for (Agent* a : agents_) s.agents.push_back(a->captureSnapshot());
    s.projectiles = projectiles_;
    s.nextProjectileId = nextProjectileId_;
    s.events = events_;
    std::ostringstream os;
    os << engine_;
    s.rngState = os.str();
    return s;
}

void World::restore(const WorldSnapshot& snap) {
    // Index live agents by Unit::id for matching.
    std::unordered_map<int, Agent*> byId;
    byId.reserve(agents_.size());
    for (Agent* a : agents_) byId[a->unit().id] = a;

    for (const AgentSnapshot& as : snap.agents) {
        auto it = byId.find(as.id);
        if (it != byId.end()) it->second->applySnapshot(as);
    }

    projectiles_ = snap.projectiles;
    nextProjectileId_ = snap.nextProjectileId;
    events_ = snap.events;
    if (!snap.rngState.empty()) {
        std::istringstream is(snap.rngState);
        is >> engine_;
    }
}

void World::applyProjectileHit_(const Projectile& p, Agent& target) {
    bool wasAlive = target.unit().alive();
    float dealt = target.unit().takeDamage(p.damage, p.kind);
    if (dealt > 0.0f) {
        events_.push_back(DamageEvent{
            p.ownerId,
            target.unit().id,
            dealt,
            p.kind,
            wasAlive && !target.unit().alive()
        });
    }
}

void World::cullProjectiles() {
    projectiles_.erase(
        std::remove_if(projectiles_.begin(), projectiles_.end(),
                       [](const Projectile& p) { return !p.alive; }),
        projectiles_.end());
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
