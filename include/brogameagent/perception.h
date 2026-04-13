#pragma once

#include "types.h"
#include <vector>

namespace brogameagent {

/// 2D line-of-sight check: returns true if line from→to is not blocked by any AABB.
bool hasLineOfSight(Vec2 from, Vec2 to, const AABB* obstacles, int count);

/// Combined visibility check: range + field-of-view + line-of-sight.
/// @param facingYaw  Observer's forward yaw (same convention as AimResult::yaw).
/// @param fovRadians Full cone angle; target must be within ±fov/2 of facing.
/// @param maxRange   Ignored if <= 0.
bool canSee(Vec2 from, Vec2 to,
            float facingYaw, float fovRadians, float maxRange,
            const AABB* obstacles, int count);

/// Compute aim angles from a position to a 3D target point.
/// Uses -Z forward convention (yaw=0 faces -Z).
AimResult computeAim(float fromX, float fromY, float fromZ,
                     float toX, float toY, float toZ);

struct LeadAimResult {
    AimResult aim;
    bool valid;      // false if no real intercept exists (target outrunning projectile)
    float timeToHit; // seconds; 0 when !valid
};

/// Compute aim angles that lead a moving target for a constant-speed projectile.
/// Solves for the smallest positive intercept time under constant target velocity.
/// If the target outruns the projectile, `valid` is false and `aim` falls back
/// to direct aim at the current position.
LeadAimResult computeLeadAim(float fromX, float fromY, float fromZ,
                             float tX, float tY, float tZ,
                             float tVX, float tVY, float tVZ,
                             float projectileSpeed);

} // namespace brogameagent
