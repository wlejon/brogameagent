#include "brogameagent/steering.h"

#include <bromath/bromath.h>

#include <cmath>

namespace brogameagent {

SteeringOutput seek(bromath::Vec2 pos, bromath::Vec2 target) {
    bromath::Vec2 dir = target - pos;
    float len = bromath::vlen(dir);
    if (len < 0.001f) return {0, 0};
    return {dir.x / len, dir.y / len};
}

SteeringOutput arrive(bromath::Vec2 pos, bromath::Vec2 target, float slowRadius) {
    bromath::Vec2 dir = target - pos;
    float dist = bromath::vlen(dir);
    if (dist < 0.001f) return {0, 0};

    float scale = (dist < slowRadius) ? (dist / slowRadius) : 1.0f;
    return {(dir.x / dist) * scale, (dir.y / dist) * scale};
}

SteeringOutput flee(bromath::Vec2 pos, bromath::Vec2 threat) {
    bromath::Vec2 dir = pos - threat;
    float len = bromath::vlen(dir);
    if (len < 0.001f) return {0, 0};
    return {dir.x / len, dir.y / len};
}

static bromath::Vec2 predictPosition(bromath::Vec2 pos, bromath::Vec2 target, bromath::Vec2 targetVel, float selfSpeed) {
    // Simple lookahead: time proportional to distance / closing speed.
    bromath::Vec2 toTarget = target - pos;
    float dist = bromath::vlen(toTarget);
    float combined = selfSpeed + bromath::vlen(targetVel);
    float lookahead = (combined > 0.001f) ? (dist / combined) : 0.0f;
    return target + targetVel * lookahead;
}

SteeringOutput pursue(bromath::Vec2 pos, bromath::Vec2 target, bromath::Vec2 targetVel, float selfSpeed) {
    return seek(pos, predictPosition(pos, target, targetVel, selfSpeed));
}

SteeringOutput evade(bromath::Vec2 pos, bromath::Vec2 threat, bromath::Vec2 threatVel, float selfSpeed) {
    return flee(pos, predictPosition(pos, threat, threatVel, selfSpeed));
}

SteeringOutput followPath(bromath::Vec2 pos, const std::vector<bromath::Vec2>& path,
                          int& waypointIndex, float advanceRadius)
{
    if (path.empty() || waypointIndex >= static_cast<int>(path.size()))
        return {0, 0};

    bromath::Vec2 wp = path[waypointIndex];
    bromath::Vec2 diff = wp - pos;
    float dist = bromath::vlen(diff);

    // Advance to next waypoint if close enough (but not for the last one)
    if (dist < advanceRadius && waypointIndex < static_cast<int>(path.size()) - 1) {
        waypointIndex++;
        wp = path[waypointIndex];
        diff = wp - pos;
        dist = bromath::vlen(diff);
    }

    // Use arrive for the last waypoint, seek for intermediate ones
    if (waypointIndex >= static_cast<int>(path.size()) - 1) {
        return arrive(pos, wp, advanceRadius * 2.0f);
    }
    return seek(pos, wp);
}

} // namespace brogameagent
