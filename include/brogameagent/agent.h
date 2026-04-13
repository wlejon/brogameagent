#pragma once

#include "types.h"
#include <vector>

namespace brogameagent {

class NavGrid;

/// A game agent that combines pathfinding with steering to navigate
/// toward a target position. Call update(dt) each tick to advance.
class Agent {
public:
    Agent();

    void setNavGrid(const NavGrid* grid);
    void setPosition(float x, float z);
    void setSpeed(float speed);
    void setRadius(float radius);

    /// Set the movement target. Recomputes path if target changed significantly.
    void setTarget(float x, float z);

    /// Clear the current target. Agent stops moving.
    void clearTarget();

    /// Advance the agent by dt seconds. Moves along the current path.
    void update(float dt);

    float x() const { return x_; }
    float z() const { return z_; }

    /// Direction the agent is currently facing (from movement), in radians.
    /// 0 = facing -Z, positive = clockwise (matches FPS yaw convention).
    float yaw() const { return yaw_; }

    /// Compute aim yaw/pitch from agent's position to a 3D world point.
    AimResult aimAt(float tx, float ty, float tz, float eyeHeight) const;

    /// Whether the agent is currently moving toward a target.
    bool hasTarget() const { return hasTarget_; }

    /// Whether the agent has reached its current target.
    bool atTarget() const;

private:
    void recomputePath();

    const NavGrid* navGrid_ = nullptr;
    float x_ = 0, z_ = 0;
    float speed_ = 6.0f;
    float radius_ = 0.4f;
    float yaw_ = 0;

    bool hasTarget_ = false;
    float targetX_ = 0, targetZ_ = 0;

    std::vector<Vec2> path_;
    int waypointIdx_ = 0;

    // Recompute path when target moves significantly
    float lastPathTargetX_ = 0, lastPathTargetZ_ = 0;
    static constexpr float REPATH_DIST_SQ = 4.0f; // repath if target moved >2 units
    static constexpr float ARRIVE_DIST = 0.5f;
};

} // namespace brogameagent
