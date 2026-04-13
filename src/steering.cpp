#include "brogameagent/steering.h"

#include <cmath>

namespace brogameagent {

SteeringOutput seek(Vec2 pos, Vec2 target) {
    Vec2 dir = target - pos;
    float len = dir.length();
    if (len < 0.001f) return {0, 0};
    return {dir.x / len, dir.z / len};
}

SteeringOutput arrive(Vec2 pos, Vec2 target, float slowRadius) {
    Vec2 dir = target - pos;
    float dist = dir.length();
    if (dist < 0.001f) return {0, 0};

    float scale = (dist < slowRadius) ? (dist / slowRadius) : 1.0f;
    return {(dir.x / dist) * scale, (dir.z / dist) * scale};
}

SteeringOutput flee(Vec2 pos, Vec2 threat) {
    Vec2 dir = pos - threat;
    float len = dir.length();
    if (len < 0.001f) return {0, 0};
    return {dir.x / len, dir.z / len};
}

static Vec2 predictPosition(Vec2 pos, Vec2 target, Vec2 targetVel, float selfSpeed) {
    // Simple lookahead: time proportional to distance / closing speed.
    Vec2 toTarget = target - pos;
    float dist = toTarget.length();
    float combined = selfSpeed + targetVel.length();
    float lookahead = (combined > 0.001f) ? (dist / combined) : 0.0f;
    return target + targetVel * lookahead;
}

SteeringOutput pursue(Vec2 pos, Vec2 target, Vec2 targetVel, float selfSpeed) {
    return seek(pos, predictPosition(pos, target, targetVel, selfSpeed));
}

SteeringOutput evade(Vec2 pos, Vec2 threat, Vec2 threatVel, float selfSpeed) {
    return flee(pos, predictPosition(pos, threat, threatVel, selfSpeed));
}

SteeringOutput followPath(Vec2 pos, const std::vector<Vec2>& path,
                          int& waypointIndex, float advanceRadius)
{
    if (path.empty() || waypointIndex >= static_cast<int>(path.size()))
        return {0, 0};

    Vec2 wp = path[waypointIndex];
    Vec2 diff = wp - pos;
    float dist = diff.length();

    // Advance to next waypoint if close enough (but not for the last one)
    if (dist < advanceRadius && waypointIndex < static_cast<int>(path.size()) - 1) {
        waypointIndex++;
        wp = path[waypointIndex];
        diff = wp - pos;
        dist = diff.length();
    }

    // Use arrive for the last waypoint, seek for intermediate ones
    if (waypointIndex >= static_cast<int>(path.size()) - 1) {
        return arrive(pos, wp, advanceRadius * 2.0f);
    }
    return seek(pos, wp);
}

} // namespace brogameagent
