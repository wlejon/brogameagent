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

/// Wrap an angle (radians) into [-pi, pi].
inline float wrapAngle(float a) {
    constexpr float TWO_PI = 6.28318530717958647692f;
    constexpr float PI     = 3.14159265358979323846f;
    a = std::fmod(a + PI, TWO_PI);
    if (a < 0) a += TWO_PI;
    return a - PI;
}

/// Shortest signed delta from `from` to `to`, in [-pi, pi].
inline float angleDelta(float from, float to) {
    return wrapAngle(to - from);
}

} // namespace brogameagent
