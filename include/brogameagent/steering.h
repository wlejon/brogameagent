#pragma once

#include "types.h"
#include <vector>

namespace brogameagent {

struct SteeringOutput {
    float fx = 0, fz = 0; // desired velocity direction (not normalized)
};

/// Seek: move directly toward target at max speed.
SteeringOutput seek(Vec2 pos, Vec2 target);

/// Arrive: move toward target, decelerating within slowRadius.
SteeringOutput arrive(Vec2 pos, Vec2 target, float slowRadius);

/// Flee: move directly away from threat.
SteeringOutput flee(Vec2 pos, Vec2 threat);

/// Follow a path: seek toward the next waypoint, advancing when close enough.
/// @param waypointIndex  In/out — current waypoint being pursued.
/// @param advanceRadius  How close before advancing to next waypoint.
SteeringOutput followPath(Vec2 pos, const std::vector<Vec2>& path,
                          int& waypointIndex, float advanceRadius);

} // namespace brogameagent
