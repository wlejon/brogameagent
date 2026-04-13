#include "brogameagent/agent.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/steering.h"
#include "brogameagent/perception.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {

Agent::Agent() {
    // Mirror legacy default so Unit and the scripted speed agree.
    speed_ = unit_.moveSpeed;
    radius_ = unit_.radius;
}

void Agent::setNavGrid(const NavGrid* grid) { navGrid_ = grid; }
void Agent::setPosition(float x, float z) { x_ = x; z_ = z; }
void Agent::setSpeed(float speed) { speed_ = speed; unit_.moveSpeed = speed; }
void Agent::setRadius(float radius) { radius_ = radius; unit_.radius = radius; }
void Agent::setMaxAccel(float a) { maxAccel_ = a; }
void Agent::setMaxTurnRate(float r) { maxTurnRate_ = r; }

void Agent::setTarget(float x, float z) {
    hasTarget_ = true;
    targetX_ = x;
    targetZ_ = z;

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

void Agent::integrate_(float desiredVx, float desiredVz, float dt) {
    // Accel clamp: limit change in velocity per step.
    float dvx = desiredVx - vx_;
    float dvz = desiredVz - vz_;
    if (maxAccel_ > 0.0f) {
        float maxDv = maxAccel_ * dt;
        float dvLen = std::sqrt(dvx * dvx + dvz * dvz);
        if (dvLen > maxDv && dvLen > 1e-6f) {
            float s = maxDv / dvLen;
            dvx *= s;
            dvz *= s;
        }
    }
    vx_ += dvx;
    vz_ += dvz;

    // Turn-rate clamp on movement facing (only when moving).
    float speedSq = vx_ * vx_ + vz_ * vz_;
    if (speedSq > 1e-6f) {
        float desiredYaw = std::atan2(vx_, -vz_);
        if (maxTurnRate_ > 0.0f) {
            float delta = angleDelta(yaw_, desiredYaw);
            float maxStep = maxTurnRate_ * dt;
            if (delta > maxStep)       delta = maxStep;
            else if (delta < -maxStep) delta = -maxStep;
            yaw_ = wrapAngle(yaw_ + delta);
        } else {
            yaw_ = desiredYaw;
        }
    }

    x_ += vx_ * dt;
    z_ += vz_ * dt;
}

void Agent::update(float dt) {
    if (!hasTarget_ || path_.empty()) {
        // Decelerate to rest under the accel clamp.
        integrate_(0.0f, 0.0f, dt);
        return;
    }

    Vec2 pos{x_, z_};
    SteeringOutput steer = followPath(pos, path_, waypointIdx_, radius_ * 2.0f);
    integrate_(steer.fx * speed_, steer.fz * speed_, dt);
}

void Agent::applyAction(const AgentAction& action, float dt) {
    // Clamp stick magnitude to [0,1].
    float mx = action.moveX;
    float mz = action.moveZ;
    float m2 = mx * mx + mz * mz;
    if (m2 > 1.0f) {
        float s = 1.0f / std::sqrt(m2);
        mx *= s;
        mz *= s;
    }

    // Interpret (mx, mz) in agent-local frame (+X=right, -Z=forward),
    // rotate into world space using current yaw. This lets the policy
    // output "go forward" regardless of which direction that is in world space.
    float c = std::cos(yaw_);
    float s = std::sin(yaw_);
    // World forward corresponding to yaw: (sin(yaw), -cos(yaw)).
    // Local (+X,+Z) maps to world: +X_local = right = (cos(yaw), sin(yaw)),
    //                              +Z_local = back  = (-sin(yaw), cos(yaw))
    float worldDx = mx * c + mz * (-s);
    float worldDz = mx * s + mz * ( c);

    float ms = unit_.effectiveMoveSpeed();
    integrate_(worldDx * ms, worldDz * ms, dt);

    aimYaw_ = wrapAngle(action.aimYaw);
    aimPitch_ = action.aimPitch;

    unit_.tickCooldowns(dt);
}

AimResult Agent::aimAt(float tx, float ty, float tz, float eyeHeight) const {
    return computeAim(x_, eyeHeight, z_, tx, ty, tz);
}

AgentSnapshot Agent::captureSnapshot() const {
    AgentSnapshot s;
    s.id = unit_.id;
    s.x = x_; s.z = z_;
    s.vx = vx_; s.vz = vz_;
    s.yaw = yaw_; s.aimYaw = aimYaw_; s.aimPitch = aimPitch_;
    s.speed = speed_; s.radius = radius_;
    s.maxAccel = maxAccel_; s.maxTurnRate = maxTurnRate_;
    s.unit = unit_;
    s.hasTarget = hasTarget_;
    s.targetX = targetX_; s.targetZ = targetZ_;
    return s;
}

void Agent::applySnapshot(const AgentSnapshot& s) {
    x_ = s.x; z_ = s.z;
    vx_ = s.vx; vz_ = s.vz;
    yaw_ = s.yaw; aimYaw_ = s.aimYaw; aimPitch_ = s.aimPitch;
    speed_ = s.speed; radius_ = s.radius;
    maxAccel_ = s.maxAccel; maxTurnRate_ = s.maxTurnRate;
    unit_ = s.unit;
    hasTarget_ = s.hasTarget;
    targetX_ = s.targetX; targetZ_ = s.targetZ;
    path_.clear();
    waypointIdx_ = 0;
    lastPathTargetX_ = 0;
    lastPathTargetZ_ = 0;
    if (hasTarget_) recomputePath();
}

void Agent::recomputePath() {
    if (!navGrid_) {
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
