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

bool canSee(Vec2 from, Vec2 to,
            float facingYaw, float fovRadians, float maxRange,
            const AABB* obstacles, int count)
{
    Vec2 diff = to - from;
    float dist = diff.length();
    if (dist < 0.001f) return true;
    if (maxRange > 0 && dist > maxRange) return false;

    // Yaw to target (same convention: atan2(dx, -dz), 0 = -Z)
    float targetYaw = std::atan2(diff.x, -diff.z);
    float delta = std::abs(wrapAngle(targetYaw - facingYaw));
    if (delta > fovRadians * 0.5f) return false;

    return hasLineOfSight(from, to, obstacles, count);
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

LeadAimResult computeLeadAim(float fromX, float fromY, float fromZ,
                             float tX, float tY, float tZ,
                             float tVX, float tVY, float tVZ,
                             float projectileSpeed)
{
    // Solve |T + V*t - S|^2 = (p*t)^2 for smallest positive t.
    float dx = tX - fromX, dy = tY - fromY, dz = tZ - fromZ;
    float dd = dx*dx + dy*dy + dz*dz;
    float vv = tVX*tVX + tVY*tVY + tVZ*tVZ;
    float dv = dx*tVX + dy*tVY + dz*tVZ;
    float p2 = projectileSpeed * projectileSpeed;

    float a = vv - p2;
    float b = 2.0f * dv;
    float c = dd;

    float t = -1.0f;
    if (std::abs(a) < 1e-6f) {
        // Linear case: target speed == projectile speed.
        if (std::abs(b) > 1e-6f) {
            float candidate = -c / b;
            if (candidate > 0) t = candidate;
        }
    } else {
        float disc = b*b - 4.0f*a*c;
        if (disc >= 0) {
            float sq = std::sqrt(disc);
            float t1 = (-b - sq) / (2.0f * a);
            float t2 = (-b + sq) / (2.0f * a);
            // Pick smallest positive
            if (t1 > 0 && t2 > 0)      t = std::min(t1, t2);
            else if (t1 > 0)           t = t1;
            else if (t2 > 0)           t = t2;
        }
    }

    if (t <= 0) {
        return {computeAim(fromX, fromY, fromZ, tX, tY, tZ), false, 0.0f};
    }

    float ix = tX + tVX * t;
    float iy = tY + tVY * t;
    float iz = tZ + tVZ * t;
    return {computeAim(fromX, fromY, fromZ, ix, iy, iz), true, t};
}

} // namespace brogameagent
