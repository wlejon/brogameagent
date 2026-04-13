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

/// Pursue: seek toward where `target` will be, assuming constant velocity.
/// @param selfSpeed  The pursuer's speed (used to estimate lookahead time).
SteeringOutput pursue(Vec2 pos, Vec2 target, Vec2 targetVel, float selfSpeed);

/// Evade: flee from where `threat` will be, assuming constant velocity.
/// @param selfSpeed  The evader's speed (used to estimate lookahead time).
SteeringOutput evade(Vec2 pos, Vec2 threat, Vec2 threatVel, float selfSpeed);

/// Follow a path: seek toward the next waypoint, advancing when close enough.
/// @param waypointIndex  In/out — current waypoint being pursued.
/// @param advanceRadius  How close before advancing to next waypoint.
SteeringOutput followPath(Vec2 pos, const std::vector<Vec2>& path,
                          int& waypointIndex, float advanceRadius);

} // namespace brogameagent
