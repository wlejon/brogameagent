#pragma once

#include "types.h"
#include "unit.h"

namespace brogameagent {

/// An in-flight projectile tracked by the World.
///
/// Two modes:
///   - Skillshot:  targetId == -1. Travels in a fixed direction until it hits
///                 the first living enemy whose radius overlaps the projectile
///                 radius, or its lifetime expires.
///   - Homing:     targetId >= 0. Re-steers toward the target each step
///                 (capped by `speed`). Hits the target on proximity; if the
///                 target dies mid-flight it falls back to skillshot behavior
///                 using its last velocity.
///
/// Fire via World::spawnProjectile. The World updates projectiles during
/// tick() and logs a DamageEvent on impact.
struct Projectile {
    int   id       = 0;     // assigned by World
    int   ownerId  = -1;    // Unit::id of the attacker (for DamageEvent attribution)
    int   teamId   = 0;     // projectile is hostile to units with a different teamId
    int   targetId = -1;    // >=0 = homing, -1 = skillshot

    float x = 0, z = 0;
    float vx = 0, vz = 0;   // velocity (already scaled to speed)
    float speed = 20.0f;
    float radius = 0.3f;

    float damage = 0.0f;
    DamageKind kind = DamageKind::Physical;

    float remainingLife = 2.0f; // seconds before auto-expiry

    bool  alive = true;         // cleared on hit / expiry / cull
};

} // namespace brogameagent
