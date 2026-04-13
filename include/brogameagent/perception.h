#pragma once

#include "types.h"
#include <vector>

namespace brogameagent {

/// 2D line-of-sight check: returns true if line from→to is not blocked by any AABB.
bool hasLineOfSight(Vec2 from, Vec2 to, const AABB* obstacles, int count);

/// Compute aim angles from a position to a 3D target point.
/// Uses -Z forward convention (yaw=0 faces -Z).
AimResult computeAim(float fromX, float fromY, float fromZ,
                     float toX, float toY, float toZ);

} // namespace brogameagent
