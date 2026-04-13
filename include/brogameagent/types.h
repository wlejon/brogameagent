#pragma once

#include <cmath>

namespace brogameagent {

struct Vec2 {
    float x = 0, z = 0;

    Vec2() = default;
    Vec2(float x, float z) : x(x), z(z) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, z + o.z}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, z - o.z}; }
    Vec2 operator*(float s) const { return {x * s, z * s}; }

    float dot(const Vec2& o) const { return x * o.x + z * o.z; }
    float lengthSq() const { return x * x + z * z; }
    float length() const { return std::sqrt(x * x + z * z); }
    Vec2 normalized() const {
        float len = length();
        if (len < 0.0001f) return {0, 0};
        return {x / len, z / len};
    }
};

struct AABB {
    float cx, cz;   // center
    float hw, hd;    // half-width (x), half-depth (z)
};

struct AimResult {
    float yaw;
    float pitch;
};

} // namespace brogameagent
