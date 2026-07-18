#include "brogameagent/agent.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/steering.h"
#include "brogameagent/perception.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {

Agent::Agent() {
    // Mirror legacy default so Unit and the scripted speed agree.
    speed_ = unit_.moveSpeed;
    radius_ = unit_.radius;
}

Agent::~Agent() {
    // Auto-deregister so a stale Agent* can't survive in World::agents_ and
    // crash the next tick. World::~World clears registeredWorld_ on each
    // registered agent first, so destruction order doesn't matter.
    if (registeredWorld_) registeredWorld_->removeAgent(this);
}

// Copy/move: clone gameplay state but NEVER inherit the World registration.
// A copy/moved-to Agent is unregistered; the source keeps any prior World
// registration (the World still points at the source's address, not at us).

Agent::Agent(const Agent& other)
    : navGrid_(other.navGrid_),
      registeredWorld_(nullptr),  // copy is unregistered
      unit_(other.unit_),
      x_(other.x_), z_(other.z_),
      elevation_(other.elevation_),
      vx_(other.vx_), vz_(other.vz_),
      yaw_(other.yaw_),
      aimYaw_(other.aimYaw_), aimPitch_(other.aimPitch_),
      speed_(other.speed_), radius_(other.radius_),
      maxAccel_(other.maxAccel_), maxTurnRate_(other.maxTurnRate_),
      avoidance_(other.avoidance_),
      hasTarget_(other.hasTarget_),
      targetX_(other.targetX_), targetZ_(other.targetZ_),
      path_(other.path_),
      waypointIdx_(other.waypointIdx_),
      lastPathTargetX_(other.lastPathTargetX_),
      lastPathTargetZ_(other.lastPathTargetZ_) {}

Agent::Agent(Agent&& other) noexcept
    : navGrid_(other.navGrid_),
      registeredWorld_(nullptr),  // moved-to is unregistered
      unit_(std::move(other.unit_)),
      x_(other.x_), z_(other.z_),
      elevation_(other.elevation_),
      vx_(other.vx_), vz_(other.vz_),
      yaw_(other.yaw_),
      aimYaw_(other.aimYaw_), aimPitch_(other.aimPitch_),
      speed_(other.speed_), radius_(other.radius_),
      maxAccel_(other.maxAccel_), maxTurnRate_(other.maxTurnRate_),
      avoidance_(other.avoidance_),
      hasTarget_(other.hasTarget_),
      targetX_(other.targetX_), targetZ_(other.targetZ_),
      path_(std::move(other.path_)),
      waypointIdx_(other.waypointIdx_),
      lastPathTargetX_(other.lastPathTargetX_),
      lastPathTargetZ_(other.lastPathTargetZ_) {
    // Source keeps its registration (if any) — the World still points at &other,
    // not at *this. Source's ~Agent will deregister normally.
}

Agent& Agent::operator=(const Agent& other) {
    if (this == &other) return *this;
    // Preserve our own registration; copy gameplay state.
    navGrid_ = other.navGrid_;
    unit_ = other.unit_;
    x_ = other.x_; z_ = other.z_;
    elevation_ = other.elevation_;
    vx_ = other.vx_; vz_ = other.vz_;
    yaw_ = other.yaw_;
    aimYaw_ = other.aimYaw_; aimPitch_ = other.aimPitch_;
    speed_ = other.speed_; radius_ = other.radius_;
    maxAccel_ = other.maxAccel_; maxTurnRate_ = other.maxTurnRate_;
    avoidance_ = other.avoidance_;
    hasTarget_ = other.hasTarget_;
    targetX_ = other.targetX_; targetZ_ = other.targetZ_;
    path_ = other.path_;
    waypointIdx_ = other.waypointIdx_;
    lastPathTargetX_ = other.lastPathTargetX_;
    lastPathTargetZ_ = other.lastPathTargetZ_;
    return *this;
}

Agent& Agent::operator=(Agent&& other) noexcept {
    if (this == &other) return *this;
    navGrid_ = other.navGrid_;
    unit_ = std::move(other.unit_);
    x_ = other.x_; z_ = other.z_;
    elevation_ = other.elevation_;
    vx_ = other.vx_; vz_ = other.vz_;
    yaw_ = other.yaw_;
    aimYaw_ = other.aimYaw_; aimPitch_ = other.aimPitch_;
    speed_ = other.speed_; radius_ = other.radius_;
    maxAccel_ = other.maxAccel_; maxTurnRate_ = other.maxTurnRate_;
    avoidance_ = other.avoidance_;
    hasTarget_ = other.hasTarget_;
    targetX_ = other.targetX_; targetZ_ = other.targetZ_;
    path_ = std::move(other.path_);
    waypointIdx_ = other.waypointIdx_;
    lastPathTargetX_ = other.lastPathTargetX_;
    lastPathTargetZ_ = other.lastPathTargetZ_;
    return *this;
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

    // With no nav grid the "path" is just {target} — rebuild it on every
    // change (it's free, and a stale point would pin the agent at the old
    // target when successive targets are closer than REPATH_DIST; navmesh
    // funnel waypoints routinely are). With a grid, keep the threshold so a
    // chased moving target doesn't re-run A* every tick.
    float dx = targetX_ - lastPathTargetX_;
    float dz = targetZ_ - lastPathTargetZ_;
    if (path_.empty() || !navGrid_ || (dx * dx + dz * dz) > REPATH_DIST_SQ) {
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
            float delta = bromath::angleDelta(yaw_, desiredYaw);
            float maxStep = maxTurnRate_ * dt;
            if (delta > maxStep)       delta = maxStep;
            else if (delta < -maxStep) delta = -maxStep;
            yaw_ = bromath::wrapAngle(yaw_ + delta);
        } else {
            yaw_ = desiredYaw;
        }
    }

    // Candidate next position under current velocity.
    float nx = x_ + vx_ * dt;
    float nz = z_ + vz_ * dt;

    // Nav-grid aware clamping: axis-wise slide so an agent grazing an
    // obstacle wall keeps the tangential component instead of stalling.
    // Also clamps to the grid bounds so policy-driven agents (MCTS, NN)
    // stay inside the arena without needing external supervision —
    // matches what the scripted A* path would produce. Shrink by a small
    // epsilon to keep samples clear of cell boundaries.
    if (navGrid_) {
        constexpr float EPS = 1e-3f;
        float lo_x = navGrid_->minX() + EPS, hi_x = navGrid_->maxX() - EPS;
        float lo_z = navGrid_->minZ() + EPS, hi_z = navGrid_->maxZ() - EPS;
        if (nx < lo_x) { nx = lo_x; vx_ = 0.0f; }
        if (nx > hi_x) { nx = hi_x; vx_ = 0.0f; }
        if (nz < lo_z) { nz = lo_z; vz_ = 0.0f; }
        if (nz > hi_z) { nz = hi_z; vz_ = 0.0f; }
        if (!navGrid_->isWalkable(nx, nz)) {
            // Try X-only move, then Z-only, then give up (stay put).
            if (navGrid_->isWalkable(nx, z_)) {
                nz = z_; vz_ = 0.0f;
            } else if (navGrid_->isWalkable(x_, nz)) {
                nx = x_; vx_ = 0.0f;
            } else {
                nx = x_; nz = z_; vx_ = 0.0f; vz_ = 0.0f;
            }
        }
    }

    x_ = nx;
    z_ = nz;
}

bromath::Vec2 Agent::preferredVelocity_() {
    if (!hasTarget_ || path_.empty()) {
        // Idle: prefer rest (integrate_ decelerates under the accel clamp).
        return {0.0f, 0.0f};
    }
    bromath::Vec2 pos{x_, z_};
    SteeringOutput steer = followPath(pos, path_, waypointIdx_, radius_ * 2.0f);
    return {steer.fx * speed_, steer.fz * speed_};
}

void Agent::update(float dt) {
    bromath::Vec2 pref = preferredVelocity_();
    integrate_(pref.x, pref.y, dt);
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

    aimYaw_ = bromath::wrapAngle(action.aimYaw);
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
        path_ = {bromath::Vec2{targetX_, targetZ_}};
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
