#pragma once

#include "types.h"
#include "unit.h"
#include "snapshot.h"
#include <vector>

namespace brogameagent {

class NavGrid;
class World;

/// Continuous-control input for a NN policy (or any decision layer).
/// moveX/moveZ are in the agent's local frame: +X = right, -Z = forward.
/// Magnitude is clamped to [0, 1] — think of it as a joystick stick.
/// aimYaw / aimPitch are in world frame (FPS convention: yaw=0 faces -Z).
/// attackTargetId / useAbilityId default to -1 (no action).
struct AgentAction {
    float moveX = 0.0f;
    float moveZ = 0.0f;
    float aimYaw = 0.0f;
    float aimPitch = 0.0f;
    int   attackTargetId = -1;
    int   useAbilityId = -1;
};

/// Per-agent ORCA local-avoidance configuration, consumed by World::tick()
/// when the world's avoidance pass is enabled (World::setAvoidanceEnabled).
/// radius / maxSpeed <= 0 mean "derive from the agent" (its radius() and
/// speed() at tick time), so the defaults just work.
struct AgentAvoidance {
    /// When false the agent keeps its legacy scripted movement (unfiltered)
    /// but still appears to others as a non-reciprocating body they steer
    /// around at full effort.
    bool  enabled = true;
    float radius = -1.0f;          // <=0: use Agent::radius()
    float maxSpeed = -1.0f;        // <=0: use Agent::speed()
    float neighborDist = 10.0f;
    int   maxNeighbors = 10;
    float timeHorizon = 2.0f;      // seconds of mutual lookahead vs agents
    float timeHorizonObst = 1.0f;  // seconds of lookahead vs obstacles
    /// Vertical extent for the ORCA elevation filter: agents whose vertical
    /// spans [elevation - height/2, elevation + height/2] don't overlap are
    /// on different levels and ignore each other. Agent elevations default
    /// to 0 (see Agent::setElevation), so single-level worlds are unaffected.
    float height = 2.0f;
};

/// A game agent. Owns a Unit (combat stats), a 2D position, a velocity,
/// a facing yaw (movement direction) and a separate aim yaw/pitch
/// (where the bot is "looking" — decoupled from movement for FPS-style
/// strafe-and-aim).
///
/// There are two ways to drive an Agent:
///   1. Scripted: setTarget(x,z) + update(dt) — A*-pathed seek-and-arrive.
///   2. Policy:   applyAction(action, dt) — continuous control, used by NN.
/// Both routes pass through the same dynamics integrator (max accel + max
/// turn rate) so behavior is physically consistent.
class Agent {
public:
    Agent();

    /// Auto-deregisters from any World it was added to so a stale pointer
    /// doesn't survive in World::agents_ and crash the next tick().
    ~Agent();

    // Copy and move are supported but DO NOT propagate the World registration
    // back-pointer: the produced/assigned-to Agent is not registered with any
    // World. Re-call World::addAgent on the new instance if you need it.
    // (Implementations live in agent.cpp so we don't have to inline-include
    // world.h here.)
    Agent(const Agent& other);
    Agent(Agent&& other) noexcept;
    Agent& operator=(const Agent& other);
    Agent& operator=(Agent&& other) noexcept;

    void setNavGrid(const NavGrid* grid);
    void setPosition(float x, float z);

    /// Kinematic limits. maxAccel and maxTurnRate are applied inside
    /// applyAction() and the path-following update(); they are a no-op when
    /// <= 0 (treat as unlimited).
    void setMaxAccel(float unitsPerSecSq);
    void setMaxTurnRate(float radPerSec);

    // Back-compat scripted API
    void setSpeed(float speed);
    void setRadius(float radius);
    float speed() const { return speed_; }
    float radius() const { return radius_; }

    /// ORCA local-avoidance participation — applied by World::tick() when
    /// the world's avoidance pass is enabled. Defaults derive radius and
    /// maxSpeed from the agent itself.
    void setAvoidance(const AgentAvoidance& av) { avoidance_ = av; }
    const AgentAvoidance& avoidance() const { return avoidance_; }

    /// Combat / stat payload. Direct access is intentional — callers routinely
    /// mutate HP, cooldowns, etc. and we don't want accessor boilerplate.
    Unit& unit() { return unit_; }
    const Unit& unit() const { return unit_; }

    /// Set the movement target. Recomputes path if target changed significantly.
    void setTarget(float x, float z);

    /// Clear the current target. Agent stops moving.
    void clearTarget();

    /// Scripted update: follow the A* path toward setTarget().
    /// Does nothing if no target is set.
    void update(float dt);

    /// Policy update: integrate a continuous-control action. Does not touch
    /// the scripted path/target state. Use this from a NN inference loop.
    void applyAction(const AgentAction& action, float dt);

    float x() const { return x_; }
    float z() const { return z_; }

    /// Vertical position (Y) for multi-level worlds. The agent's own movement
    /// stays 2D (XZ) — elevation is embedder-driven state (set it from the
    /// navmesh route height, a ground probe, or floor index each tick) and is
    /// consumed by World's avoidance pass so agents on different levels don't
    /// steer around each other (see AgentAvoidance::height). Not part of
    /// snapshots: re-derive it after applySnapshot the same way it was set.
    void setElevation(float y) { elevation_ = y; }
    float elevation() const { return elevation_; }

    /// Movement facing (radians, FPS convention: 0 = -Z).
    float yaw() const { return yaw_; }

    /// Directly set the movement-facing yaw. Normally yaw is updated from
    /// velocity direction inside the dynamics integrator; use this when an
    /// external planner (e.g. MCTS) needs to align movement facing with an
    /// aim target before applying an action, so that "local forward" in the
    /// action's frame points at the intended aim direction.
    void setYaw(float yaw) { yaw_ = yaw; }

    /// Aim direction — decoupled from movement, set by applyAction or aimAt.
    float aimYaw() const { return aimYaw_; }
    float aimPitch() const { return aimPitch_; }

    /// Compute aim yaw/pitch from agent's position to a 3D world point.
    /// Does NOT mutate the agent's aim state — callers that want it latched
    /// should write the result into applyAction.aimYaw/aimPitch.
    AimResult aimAt(float tx, float ty, float tz, float eyeHeight) const;

    bool hasTarget() const { return hasTarget_; }
    bool atTarget() const;

    const std::vector<bromath::Vec2>& path() const { return path_; }
    int currentWaypoint() const { return waypointIdx_; }
    bromath::Vec2 velocity() const { return {vx_, vz_}; }

    /// Capture full resettable agent state into a snapshot.
    AgentSnapshot captureSnapshot() const;

    /// Restore agent state from a snapshot. If hasTarget, repaths against
    /// the current navgrid binding (not captured in the snapshot).
    void applySnapshot(const AgentSnapshot& s);

private:
    void recomputePath();

    /// Shared dynamics step: clamp to maxAccel, clamp facing rotation to
    /// maxTurnRate, integrate position.
    void integrate_(float desiredVx, float desiredVz, float dt);

    /// What the scripted path follower wants this tick ({0,0} when idle).
    /// Advances the waypoint cursor. update() is preferredVelocity_ +
    /// integrate_; World's avoidance pass inserts the ORCA filter between
    /// the two.
    bromath::Vec2 preferredVelocity_();

    const NavGrid* navGrid_ = nullptr;
    // Set by World::addAgent / cleared by World::removeAgent (and by ~World).
    // Used by ~Agent() to auto-deregister so a stale pointer can't be ticked.
    World* registeredWorld_ = nullptr;
    friend class World;
    Unit unit_{};

    float x_ = 0, z_ = 0;
    float elevation_ = 0;  // Y, avoidance elevation filter only
    float vx_ = 0, vz_ = 0;
    float yaw_ = 0;
    float aimYaw_ = 0, aimPitch_ = 0;

    // Legacy scripted-path speed; mirrors unit_.moveSpeed by default.
    float speed_ = 6.0f;
    float radius_ = 0.4f;

    float maxAccel_    = 0.0f; // <=0 = unlimited
    float maxTurnRate_ = 0.0f; // <=0 = unlimited

    AgentAvoidance avoidance_{};

    bool hasTarget_ = false;
    float targetX_ = 0, targetZ_ = 0;

    std::vector<bromath::Vec2> path_;
    int waypointIdx_ = 0;

    float lastPathTargetX_ = 0, lastPathTargetZ_ = 0;
    static constexpr float REPATH_DIST_SQ = 4.0f;
    static constexpr float ARRIVE_DIST = 0.5f;
};

} // namespace brogameagent
