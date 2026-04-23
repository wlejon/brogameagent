#pragma once

#include "observability.h"
#include "types.h"

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace brogameagent {

class NavGrid;
class World;

namespace belief {

/// One hypothesis about a hidden enemy's concrete state. Weights are reserved
/// for later particle-resampling work; v1 keeps them uniform.
struct EnemyParticle {
    Vec2  pos{};
    Vec2  vel{};
    float hp = 0.0f;
    float heading = 0.0f;
    float weight = 1.0f;
};

/// Per-enemy particle cloud. `last_observation` is a cached copy of the most
/// recent observation seen for this enemy — used to collapse particles on a
/// fresh sighting and to cheaply answer "is this enemy currently visible?"
struct EnemyBelief {
    int                        enemy_id = 0;
    float                      max_hp = 0.0f;
    std::vector<EnemyParticle> particles;
    bool                       ever_seen = false;
    bool                       visible   = false;      // as of last update()
    float                      last_seen_elapsed = 0.0f;
};

/// Simple motion-model parameters for particle propagation. Particles drift
/// along their cached velocity with random-walk perturbation; motion is
/// clamped to the nav grid (when supplied).
struct MotionParams {
    float max_speed = 6.0f;      // hard cap on propagated speed
    float accel_std = 4.0f;      // stddev of per-second random accel (units/s²)
    float spread_on_loss = 3.0f; // initial positional spread when contact is lost
};

/// Team-scoped belief over hidden enemy state. Thread-compatible, not thread-
/// safe: a single TeamBelief instance should be driven from one planner.
class TeamBelief {
public:
    TeamBelief() = default;
    TeamBelief(int team_id, int num_particles,
                const NavGrid* nav,
                MotionParams motion = {},
                uint64_t rng_seed = 0xBE11EFCAFEULL);

    /// Reset everything; keeps configuration (team, nav, motion).
    void clear();

    /// Register an enemy id we want to track. If a prior `initial_pos` is
    /// known, particles are seeded around it with `spread_on_loss`; otherwise
    /// they uniform-sample over the nav grid. Ignored if the id already
    /// exists. `max_hp` is used only to clamp/normalise later estimates.
    void register_enemy(int enemy_id, float max_hp,
                         const Vec2* initial_pos_prior = nullptr);

    /// Advance all particles by `dt` under the motion model and negative-info
    /// prune them against `visibility_source_world` from the perspective of
    /// `team_id`. A particle at a pose that *would* be visible to this team
    /// but isn't reflected in the latest update() is discarded and resampled
    /// from its remaining siblings.
    ///
    /// Callers typically invoke this right before update() each tick, so the
    /// negative-information step operates on the already-propagated cloud.
    void propagate(const World& world_for_geometry,
                    const obs::VisibilityConfig& vis,
                    float dt);

    /// Fold a fresh observation into the belief. Visible enemies collapse to
    /// a delta at the observed state (all particles snap there); hidden-but-
    /// previously-seen enemies keep their propagated cloud; newly-invisible
    /// enemies (just lost contact) get reseeded with `spread_on_loss` around
    /// the last-seen position.
    void update(const obs::TeamObservation& o);

    /// Sample a concrete {enemy_id → particle} assignment for one
    /// determinization. Per-enemy samples are independent (no joint
    /// correlations in v1).
    std::unordered_map<int, EnemyParticle> sample(std::mt19937_64& rng) const;

    /// Belief-mean particle per enemy. Used by planners to build a "canonical"
    /// world for observation-slot ordering and legal-action enumeration so
    /// that tree structure is stable across determinizations.
    std::unordered_map<int, EnemyParticle> mean() const;

    /// Effective sample size (Kish): sum(w)^2 / sum(w^2), summed across
    /// enemies and normalised by total particle count. 1.0 means maximally
    /// spread (flat weights); approaches 0 as one particle dominates.
    float effective_sample_size() const;

    const std::vector<EnemyBelief>& enemies() const { return enemies_; }
    int    num_particles() const { return num_particles_; }
    int    team_id()       const { return team_id_; }
    std::mt19937_64& rng() { return rng_; }

private:
    EnemyBelief* find_(int enemy_id);
    const EnemyBelief* find_(int enemy_id) const;

    void seed_uniform_(EnemyBelief& b);
    void seed_around_(EnemyBelief& b, Vec2 center, float spread);
    void clamp_to_nav_(Vec2& p) const;

    int                       team_id_ = 0;
    int                       num_particles_ = 32;
    const NavGrid*            nav_ = nullptr;
    MotionParams              motion_{};
    std::vector<EnemyBelief>  enemies_;
    std::mt19937_64           rng_{0xBE11EFCAFEULL};
};

} // namespace belief
} // namespace brogameagent
