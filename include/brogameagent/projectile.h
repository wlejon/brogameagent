#pragma once

#include "types.h"
#include "unit.h"

namespace brogameagent {

enum class ProjectileMode {
    Single, // first overlap deals damage and dies (default)
    Pierce, // every overlap deals damage; dies on lifetime or maxHits
    AoE     // first overlap deals damage, splashes within splashRadius, dies
};

/// An in-flight projectile tracked by the World.
///
/// Two homing modes (skillshot vs homing via targetId) combined with three
/// collision modes (Single / Pierce / AoE):
///   - Skillshot + Single: classic fire-and-forget shot (stops at first hit)
///   - Skillshot + Pierce: line AoE (e.g. a laser), tracks ids already hit
///   - Skillshot + AoE:    grenade / rocket (detonates on first contact)
///   - Homing + Single:    guided missile
///   - Homing + Pierce/AoE: valid but unusual — homing tracks targetId, the
///                         splash or pierce occurs at the impact point.
///
/// Fire via World::spawnProjectile. The World updates projectiles during
/// tick() and logs a DamageEvent on each hit.
struct Projectile {
    static constexpr int MAX_PIERCE_MEMORY = 8;

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

    float remainingLife = 2.0f;

    ProjectileMode mode = ProjectileMode::Single;
    float splashRadius = 0.0f; // AoE only
    int   maxHits = 0;         // Pierce cap; 0 = unlimited (until lifetime ends)

    // Pierce bookkeeping: which target ids have already been damaged by this
    // shot, so each enemy is hit at most once.
    int   hitIds[MAX_PIERCE_MEMORY] = {0};
    int   hitCount = 0;

    bool  alive = true;
};

} // namespace brogameagent
