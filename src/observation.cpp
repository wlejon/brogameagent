#include "brogameagent/observation.h"
#include "brogameagent/agent.h"
#include "brogameagent/world.h"
#include "brogameagent/unit.h"
#include "brogameagent/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace brogameagent::observation {

// Cooldown normalization: 1.0 if >= this many seconds away, 0 if ready.
static constexpr float COOLDOWN_NORM_SEC = 10.0f;

static float norm01(float v, float max) {
    if (max <= 0) return 0.0f;
    float x = v / max;
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

struct LocalRel {
    float lx, lz, dist;
};

// World-relative (dx,dz) rotated into self's local frame where -Z = forward.
static LocalRel toLocal(float dx, float dz, float selfYaw) {
    // Inverse of the applyAction rotation: local = R(-yaw) * world.
    float c = std::cos(selfYaw);
    float s = std::sin(selfYaw);
    // World (dx,dz) where +X=east, +Z=south. Local (+X=right, -Z=forward).
    // If yaw rotates local->world via (wx,wz) = R*local, then local = R^T*world.
    // From agent.cpp integrate logic: worldDx = lx*c - lz*s; worldDz = lx*s + lz*c.
    // So lx =  worldDx*c + worldDz*s;  lz = -worldDx*s + worldDz*c.
    float lx =  dx * c + dz * s;
    float lz = -dx * s + dz * c;
    float dist = std::sqrt(dx * dx + dz * dz);
    return {lx, lz, dist};
}

void build(const Agent& self, const World& world, float* out) {
    std::memset(out, 0, sizeof(float) * TOTAL);

    const Unit& u = self.unit();
    int i = 0;

    // --- Self block ---
    out[i++] = norm01(u.hp, u.maxHp);
    out[i++] = norm01(u.mana, u.maxMana);
    out[i++] = norm01(u.attackCooldown, COOLDOWN_NORM_SEC);
    out[i++] = norm01(u.abilityCooldowns[0], COOLDOWN_NORM_SEC);
    out[i++] = norm01(u.abilityCooldowns[1], COOLDOWN_NORM_SEC);

    Vec2 v = self.velocity();
    float speed = std::sqrt(v.x * v.x + v.z * v.z);
    out[i++] = norm01(speed, u.moveSpeed);

    float aimOff = angleDelta(self.yaw(), self.aimYaw());
    out[i++] = std::sin(aimOff);
    out[i++] = std::cos(aimOff);

    // --- Neighbor extraction: sort all enemies/allies once by distance ---
    struct Scored { float d2; Agent* a; };
    std::vector<Scored> enemies, allies;
    enemies.reserve(world.agents().size());
    allies.reserve(world.agents().size());

    for (Agent* a : world.agents()) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        float dx = a->x() - self.x();
        float dz = a->z() - self.z();
        float d2 = dx * dx + dz * dz;
        if (a->unit().teamId == u.teamId) allies.push_back({d2, a});
        else                              enemies.push_back({d2, a});
    }
    std::sort(enemies.begin(), enemies.end(),
              [](const Scored& p, const Scored& q) { return p.d2 < q.d2; });
    std::sort(allies.begin(), allies.end(),
              [](const Scored& p, const Scored& q) { return p.d2 < q.d2; });

    // --- Enemy block ---
    for (int k = 0; k < K_ENEMIES; k++) {
        int base = SELF_FEATURES + k * ENEMY_FEATURES;
        if (k >= static_cast<int>(enemies.size())) {
            // Already zeroed by memset; leave valid=0.
            continue;
        }
        Agent* e = enemies[k].a;
        float dx = e->x() - self.x();
        float dz = e->z() - self.z();
        LocalRel r = toLocal(dx, dz, self.yaw());

        out[base + 0] = 1.0f;
        out[base + 1] = std::clamp(r.lx / OBS_RANGE, -1.0f, 1.0f);
        out[base + 2] = std::clamp(r.lz / OBS_RANGE, -1.0f, 1.0f);
        out[base + 3] = std::clamp(r.dist / OBS_RANGE, 0.0f, 1.0f);
        out[base + 4] = norm01(e->unit().hp, e->unit().maxHp);
        out[base + 5] = (r.dist <= u.attackRange) ? 1.0f : 0.0f;
    }

    // --- Ally block ---
    int allyOffset = SELF_FEATURES + K_ENEMIES * ENEMY_FEATURES;
    for (int k = 0; k < K_ALLIES; k++) {
        int base = allyOffset + k * ALLY_FEATURES;
        if (k >= static_cast<int>(allies.size())) continue;
        Agent* al = allies[k].a;
        float dx = al->x() - self.x();
        float dz = al->z() - self.z();
        LocalRel r = toLocal(dx, dz, self.yaw());

        out[base + 0] = 1.0f;
        out[base + 1] = std::clamp(r.lx / OBS_RANGE, -1.0f, 1.0f);
        out[base + 2] = std::clamp(r.lz / OBS_RANGE, -1.0f, 1.0f);
        out[base + 3] = std::clamp(r.dist / OBS_RANGE, 0.0f, 1.0f);
        out[base + 4] = norm01(al->unit().hp, al->unit().maxHp);
    }
}

} // namespace brogameagent::observation
