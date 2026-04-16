#pragma once

#include "types.h"
#include "unit.h"
#include "snapshot.h"
#include <vector>

namespace brogameagent {

class NavGrid;

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

    const std::vector<Vec2>& path() const { return path_; }
    int currentWaypoint() const { return waypointIdx_; }
    Vec2 velocity() const { return {vx_, vz_}; }

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

    const NavGrid* navGrid_ = nullptr;
    Unit unit_{};

    float x_ = 0, z_ = 0;
    float vx_ = 0, vz_ = 0;
    float yaw_ = 0;
    float aimYaw_ = 0, aimPitch_ = 0;

    // Legacy scripted-path speed; mirrors unit_.moveSpeed by default.
    float speed_ = 6.0f;
    float radius_ = 0.4f;

    float maxAccel_    = 0.0f; // <=0 = unlimited
    float maxTurnRate_ = 0.0f; // <=0 = unlimited

    bool hasTarget_ = false;
    float targetX_ = 0, targetZ_ = 0;

    std::vector<Vec2> path_;
    int waypointIdx_ = 0;

    float lastPathTargetX_ = 0, lastPathTargetZ_ = 0;
    static constexpr float REPATH_DIST_SQ = 4.0f;
    static constexpr float ARRIVE_DIST = 0.5f;
};

} // namespace brogameagent
