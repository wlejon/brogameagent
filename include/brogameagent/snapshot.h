#pragma once

#include "projectile.h"
#include "types.h"
#include "unit.h"

#include <string>
#include <vector>

namespace brogameagent {

/// Full resettable state for a single Agent. Path_/waypointIdx_ are not
/// captured because the path is regenerated from (position, target, navgrid)
/// on restore. NavGrid pointer is not captured — the caller's existing
/// navgrid binding is preserved across restore.
struct AgentSnapshot {
    int id = 0;
    float x = 0, z = 0;
    float vx = 0, vz = 0;
    float yaw = 0, aimYaw = 0, aimPitch = 0;
    float speed = 0, radius = 0;
    float maxAccel = 0, maxTurnRate = 0;
    Unit  unit{};
    bool  hasTarget = false;
    float targetX = 0, targetZ = 0;
};

/// Full resettable World state. Re-apply via World::restore to rewind the
/// simulation exactly (given the same agent roster & registered abilities).
///
/// Agents are matched by Unit::id between snapshot and live world; agents
/// present in the live world but missing from the snapshot are untouched,
/// and vice versa. Obstacles and registered abilities are NOT captured —
/// they are assumed static for the lifetime of a training run.
struct WorldSnapshot {
    std::vector<AgentSnapshot> agents;
    std::vector<Projectile>    projectiles;
    int  nextProjectileId = 1;
    std::vector<DamageEvent>   events;
    /// Serialized std::mt19937_64 state (via operator<<). Portable across
    /// runs of the same stdlib version; not necessarily across compilers.
    std::string rngState;
};

} // namespace brogameagent
