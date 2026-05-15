#pragma once

// Domain types unique to brogameagent. General-purpose math (Vec2, angle
// wrapping, scalar constants) lives in bromath — include "bromath/bromath.h"
// and use bromath:: types directly. brogameagent never re-exports those.

#include <bromath/bromath.h>

namespace brogameagent {

/// Axis-aligned bounding box on the ground plane, in CENTER + HALF-EXTENTS form.
/// Kept local (not bromath::AABB2 which is min/max) because the perception
/// ray-vs-AABB and the navgrid obstacle marker both consume centre+half-extents
/// directly — converting would just push the same math one layer up.
struct AABB {
    float cx, cz;   // center
    float hw, hd;    // half-width (x), half-depth (z)
};

/// Yaw/pitch aim solution (radians, FPS convention: yaw=0 faces -Z/-Y).
struct AimResult {
    float yaw;
    float pitch;
};

} // namespace brogameagent
