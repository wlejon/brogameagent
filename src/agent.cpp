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
    vx_ = 0;
    vz_ = 0;
    if (!hasTarget_ || path_.empty()) return;

    Vec2 pos{x_, z_};
    SteeringOutput steer = followPath(pos, path_, waypointIdx_, radius_ * 2.0f);

    // followPath returns a vector whose magnitude already encodes the arrive
    // slowdown (0..1), so we apply it directly as a velocity scale.
    if (steer.fx * steer.fx + steer.fz * steer.fz > 1e-6f) {
        vx_ = steer.fx * speed_;
        vz_ = steer.fz * speed_;
        x_ += vx_ * dt;
        z_ += vz_ * dt;
        yaw_ = std::atan2(steer.fx, -steer.fz);
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
