#include "brogameagent/perception.h"

#include <cmath>
#include <algorithm>

namespace brogameagent {

// 2D ray vs AABB (XZ plane). Returns fraction [0,1] of hit, or -1.
static float rayVsAABB2D(Vec2 origin, Vec2 dir, float len, const AABB& box) {
    float invLen = 1.0f / len;
    float dx = dir.x * invLen;
    float dz = dir.z * invLen;

    float x0 = box.cx - box.hw, x1 = box.cx + box.hw;
    float z0 = box.cz - box.hd, z1 = box.cz + box.hd;

    float tmin = 0, tmax = len;

    if (std::abs(dx) < 1e-9f) {
        if (origin.x < x0 || origin.x > x1) return -1;
    } else {
        float t1 = (x0 - origin.x) / dx;
        float t2 = (x1 - origin.x) / dx;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax) return -1;
    }

    if (std::abs(dz) < 1e-9f) {
        if (origin.z < z0 || origin.z > z1) return -1;
    } else {
        float t1 = (z0 - origin.z) / dz;
        float t2 = (z1 - origin.z) / dz;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax) return -1;
    }

    return tmin;
}

bool hasLineOfSight(Vec2 from, Vec2 to, const AABB* obstacles, int count) {
    Vec2 diff = to - from;
    float len = diff.length();
    if (len < 0.001f) return true;

    for (int i = 0; i < count; i++) {
        float t = rayVsAABB2D(from, diff, len, obstacles[i]);
        if (t >= 0 && t <= len) return false;
    }
    return true;
}

AimResult computeAim(float fromX, float fromY, float fromZ,
                     float toX, float toY, float toZ)
{
    float dx = toX - fromX;
    float dy = toY - fromY;
    float dz = toZ - fromZ;

    // Yaw: angle in XZ plane. 0 = facing -Z, positive = clockwise.
    // atan2(dx, -dz) gives: dx=0,dz=-1 → 0, dx=1,dz=0 → π/2
    float yaw = std::atan2(dx, -dz);

    // Pitch: angle above/below horizontal
    float horizDist = std::sqrt(dx * dx + dz * dz);
    float pitch = std::atan2(dy, horizDist);

    return {yaw, pitch};
}

} // namespace brogameagent
