#pragma once

#include "types.h"

#include <vector>

namespace brogameagent {

class World;

namespace obs {

/// Snapshot of one agent's state from a team's point of view. For visible
/// agents these fields mirror ground truth; for previously-seen but currently
/// hidden enemies, they reflect the last direct sighting (time in
/// `last_seen_elapsed`).
struct AgentObservation {
    int   id = 0;
    int   team_id = 0;
    Vec2  pos{};
    Vec2  vel{};
    float hp = 0.0f;
    float max_hp = 0.0f;
    float heading = 0.0f;
    bool  alive = false;
    bool  visible = false;          // true iff any ally has LOS+FOV+range right now
    float last_seen_elapsed = 0.0f; // sim seconds since last direct sighting; 0 when visible
};

/// A team's observation of the world. Allies are fully known; enemies are a
/// mix of currently-visible entries (ground truth) and stale entries carried
/// forward from prior sightings. Caller is responsible for timekeeping via
/// the `now` argument — typically the simulation's accumulated elapsed time.
struct TeamObservation {
    int                           team_id = 0;
    float                         timestamp = 0.0f;
    std::vector<AgentObservation> allies;
    std::vector<AgentObservation> enemies;
};

/// Visibility parameters. FOV is measured as a full cone angle (a unit with
/// `fov_radians = 2*pi` sees in every direction); `max_range` <= 0 disables
/// the range check. `check_los` gates obstacles-based occlusion.
struct VisibilityConfig {
    float fov_radians = 6.28318530717958647692f;  // full circle by default
    float max_range   = 0.0f;                      // 0 = unlimited
    bool  check_los   = true;
};

/// Build a fresh team observation against ground truth. Allies are populated
/// in registration order; enemies are populated in ground-truth registration
/// order, marked visible iff at least one living ally on the team satisfies
/// FOV + LOS + range.
TeamObservation observe(const World& world, int team_id,
                         const VisibilityConfig& cfg,
                         float now);

/// Carry state forward: fold a fresh `fresh` observation into the prior
/// belief-compatible view `prior`, preserving stale enemy entries for enemies
/// that were seen before and are still tracked. Output matches `prior`'s
/// roster — visible enemies overwrite stale entries, hidden enemies retain
/// their last-seen values but update `last_seen_elapsed = now - last_ts`.
///
/// Use this when the caller wants a continuously-updated TeamObservation
/// across ticks without re-registering enemy ids every call.
TeamObservation merge(const TeamObservation& prior,
                       const TeamObservation& fresh,
                       float now);

} // namespace obs
} // namespace brogameagent
