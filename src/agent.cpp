#include "brogameagent/agent.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/steering.h"
#include "brogameagent/perception.h"

#include <cmath>

namespace brogameagent {

Agent::Agent() = default;

void Agent::setNavGrid(const NavGrid* grid) { navGrid_ = grid; }
void Agent::setPosition(float x, float z) { x_ = x; z_ = z; }
void Agent::setSpeed(float speed) { speed_ = speed; }
void Agent::setRadius(float radius) { radius_ = radius; }

void Agent::setTarget(float x, float z) {
    hasTarget_ = true;
    targetX_ = x;
    targetZ_ = z;

    // Recompute path if target moved significantly or we have no path
    float dx = targetX_ - lastPathTargetX_;
    float dz = targetZ_ - lastPathTargetZ_;
    if (path_.empty() || (dx * dx + dz * dz) > REPATH_DIST_SQ) {
        recomputePath();
    }
}

void Agent::clearTarget() {
    hasTarget_ = false;
    path_.clear();
    waypointIdx_ = 0;
}

bool Agent::atTarget() const {
    if (!hasTarget_) return false;
    float dx = x_ - targetX_;
    float dz = z_ - targetZ_;
    return (dx * dx + dz * dz) < ARRIVE_DIST * ARRIVE_DIST;
}

void Agent::update(float dt) {
    if (!hasTarget_ || path_.empty()) return;

    Vec2 pos{x_, z_};
    SteeringOutput steer = followPath(pos, path_, waypointIdx_, radius_ * 2.0f);

    float len = std::sqrt(steer.fx * steer.fx + steer.fz * steer.fz);
    if (len > 0.001f) {
        x_ += (steer.fx / len) * speed_ * dt * len; // len acts as speed scale (arrive)
        z_ += (steer.fz / len) * speed_ * dt * len;

        // Update yaw from movement direction
        yaw_ = std::atan2(steer.fx, -steer.fz);
    }

    // Repath periodically if target is moving
    float dx = targetX_ - lastPathTargetX_;
    float dz = targetZ_ - lastPathTargetZ_;
    if ((dx * dx + dz * dz) > REPATH_DIST_SQ) {
        recomputePath();
    }
}

AimResult Agent::aimAt(float tx, float ty, float tz, float eyeHeight) const {
    return computeAim(x_, eyeHeight, z_, tx, ty, tz);
}

void Agent::recomputePath() {
    if (!navGrid_) {
        // No navgrid — direct path
        path_ = {Vec2{targetX_, targetZ_}};
        waypointIdx_ = 0;
        lastPathTargetX_ = targetX_;
        lastPathTargetZ_ = targetZ_;
        return;
    }

    path_ = navGrid_->findPath({x_, z_}, {targetX_, targetZ_});
    waypointIdx_ = 0;
    lastPathTargetX_ = targetX_;
    lastPathTargetZ_ = targetZ_;
}

} // namespace brogameagent
