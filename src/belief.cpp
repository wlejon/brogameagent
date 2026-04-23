#include "brogameagent/belief.h"

#include "brogameagent/agent.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/perception.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <cmath>

namespace brogameagent::belief {

namespace {
constexpr float TWO_PI = 6.28318530717958647692f;

float gaussian(std::mt19937_64& rng, float mean, float stddev) {
    // Box–Muller. std::normal_distribution is implementation-defined across
    // stdlibs and we want cross-platform determinism under a fixed seed.
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u1 = std::max(1e-7f, u01(rng));
    float u2 = u01(rng);
    float mag = std::sqrt(-2.0f * std::log(u1));
    return mean + stddev * mag * std::cos(TWO_PI * u2);
}

} // namespace

TeamBelief::TeamBelief(int team_id, int num_particles,
                        const NavGrid* nav,
                        MotionParams motion,
                        uint64_t rng_seed)
    : team_id_(team_id)
    , num_particles_(num_particles > 0 ? num_particles : 1)
    , nav_(nav)
    , motion_(motion)
    , rng_(rng_seed) {}

void TeamBelief::clear() { enemies_.clear(); }

EnemyBelief* TeamBelief::find_(int id) {
    for (auto& e : enemies_) if (e.enemy_id == id) return &e;
    return nullptr;
}

const EnemyBelief* TeamBelief::find_(int id) const {
    for (auto& e : enemies_) if (e.enemy_id == id) return &e;
    return nullptr;
}

void TeamBelief::clamp_to_nav_(Vec2& p) const {
    if (!nav_) return;
    // First clamp to bounds.
    p.x = std::clamp(p.x, nav_->minX() + 0.1f, nav_->maxX() - 0.1f);
    p.z = std::clamp(p.z, nav_->minZ() + 0.1f, nav_->maxZ() - 0.1f);
    // If we landed inside an obstacle, nudge outward radially by up to a few
    // cell widths. Cheap fallback; good enough for particle clouds.
    if (!nav_->isWalkable(p.x, p.z)) {
        const float cs = nav_->cellSize();
        for (int k = 1; k <= 6; k++) {
            const float r = cs * static_cast<float>(k);
            for (int ang = 0; ang < 8; ang++) {
                float a = static_cast<float>(ang) * (TWO_PI / 8.0f);
                Vec2 cand{ p.x + r * std::cos(a), p.z + r * std::sin(a) };
                if (nav_->isWalkable(cand.x, cand.z)) { p = cand; return; }
            }
        }
    }
}

void TeamBelief::seed_uniform_(EnemyBelief& b) {
    b.particles.clear();
    b.particles.reserve(num_particles_);
    if (nav_) {
        std::uniform_real_distribution<float> ux(nav_->minX(), nav_->maxX());
        std::uniform_real_distribution<float> uz(nav_->minZ(), nav_->maxZ());
        int tries = 0;
        while (static_cast<int>(b.particles.size()) < num_particles_ && tries < num_particles_ * 20) {
            Vec2 p{ ux(rng_), uz(rng_) };
            tries++;
            if (!nav_->isWalkable(p.x, p.z)) continue;
            b.particles.push_back({p, {0,0}, b.max_hp, 0.0f, 1.0f});
        }
        if (b.particles.empty()) {
            b.particles.push_back({{0,0}, {0,0}, b.max_hp, 0.0f, 1.0f});
        }
        while (static_cast<int>(b.particles.size()) < num_particles_) {
            b.particles.push_back(b.particles.back());
        }
    } else {
        for (int i = 0; i < num_particles_; i++) {
            b.particles.push_back({{0,0}, {0,0}, b.max_hp, 0.0f, 1.0f});
        }
    }
}

void TeamBelief::seed_around_(EnemyBelief& b, Vec2 center, float spread) {
    b.particles.clear();
    b.particles.reserve(num_particles_);
    for (int i = 0; i < num_particles_; i++) {
        Vec2 p{ center.x + gaussian(rng_, 0.0f, spread),
                center.z + gaussian(rng_, 0.0f, spread) };
        clamp_to_nav_(p);
        b.particles.push_back({p, {0,0}, b.max_hp, 0.0f, 1.0f});
    }
}

void TeamBelief::register_enemy(int enemy_id, float max_hp,
                                  const Vec2* initial_pos_prior) {
    if (find_(enemy_id)) return;
    EnemyBelief b;
    b.enemy_id = enemy_id;
    b.max_hp = max_hp;
    if (initial_pos_prior) {
        seed_around_(b, *initial_pos_prior, motion_.spread_on_loss);
        b.ever_seen = true;  // prior counts as a weak sighting
    } else {
        seed_uniform_(b);
        b.ever_seen = false;
    }
    enemies_.push_back(std::move(b));
}

void TeamBelief::propagate(const World& world_for_geometry,
                             const obs::VisibilityConfig& vis,
                             float dt) {
    if (dt <= 0.0f) return;

    // Build a list of "observer poses" for this team — the set of (pos, yaw)
    // from which FOV/LOS is computed. Particles that would be visible from
    // any of these but aren't in the *current* observation get pruned.
    struct Observer { Vec2 pos; float yaw; };
    std::vector<Observer> observers;
    observers.reserve(8);
    for (Agent* a : world_for_geometry.agents()) {
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != team_id_) continue;
        observers.push_back({{a->x(), a->z()}, a->yaw()});
    }
    const auto& obstacles = world_for_geometry.obstacles();
    const AABB* obs_ptr = obstacles.empty() ? nullptr : obstacles.data();
    const int   obs_n   = static_cast<int>(obstacles.size());

    auto visible_from_team = [&](Vec2 p) {
        for (auto& o : observers) {
            if (vis.max_range > 0.0f) {
                float dx = p.x - o.pos.x, dz = p.z - o.pos.z;
                if (dx*dx + dz*dz > vis.max_range * vis.max_range) continue;
            }
            if (vis.fov_radians < TWO_PI - 1e-4f) {
                float dx = p.x - o.pos.x, dz = p.z - o.pos.z;
                float angle = std::atan2(dx, -dz);
                if (std::fabs(wrapAngle(angle - o.yaw)) > vis.fov_radians * 0.5f) continue;
            }
            if (vis.check_los && !hasLineOfSight(o.pos, p, obs_ptr, obs_n)) continue;
            return true;
        }
        return false;
    };

    for (auto& b : enemies_) {
        if (b.visible) continue;  // belief pinned to the live observation
        if (!b.ever_seen) continue;  // uniform prior; skip drift

        for (auto& p : b.particles) {
            // Random-walk acceleration, integrate velocity + position.
            float ax = gaussian(rng_, 0.0f, motion_.accel_std);
            float az = gaussian(rng_, 0.0f, motion_.accel_std);
            p.vel.x += ax * dt;
            p.vel.z += az * dt;
            float vmag = std::sqrt(p.vel.x*p.vel.x + p.vel.z*p.vel.z);
            if (vmag > motion_.max_speed) {
                float s = motion_.max_speed / vmag;
                p.vel.x *= s; p.vel.z *= s;
            }
            p.pos.x += p.vel.x * dt;
            p.pos.z += p.vel.z * dt;
            clamp_to_nav_(p.pos);
        }

        // Negative-info pruning: zero out particles that should have been seen.
        std::vector<EnemyParticle> kept;
        kept.reserve(b.particles.size());
        for (auto& p : b.particles) {
            if (!visible_from_team(p.pos)) kept.push_back(p);
        }
        if (kept.empty()) {
            // Planner sees nothing consistent — fall back to a uniform prior
            // over the nav grid so the belief doesn't collapse to "nowhere."
            seed_uniform_(b);
        } else {
            // Resample with replacement to keep particle count fixed.
            std::uniform_int_distribution<size_t> pick(0, kept.size() - 1);
            b.particles.clear();
            b.particles.reserve(num_particles_);
            for (int i = 0; i < num_particles_; i++) {
                b.particles.push_back(kept[pick(rng_)]);
            }
        }

        b.last_seen_elapsed += dt;
    }
}

void TeamBelief::update(const obs::TeamObservation& o) {
    for (const auto& eo : o.enemies) {
        EnemyBelief* b = find_(eo.id);
        if (!b) continue;  // caller hasn't registered this enemy
        if (eo.visible && eo.alive) {
            b->visible = true;
            b->ever_seen = true;
            b->last_seen_elapsed = 0.0f;
            // Collapse to a delta: all particles at observed state.
            b->particles.clear();
            b->particles.reserve(num_particles_);
            EnemyParticle obs_p{ eo.pos, eo.vel, eo.hp, eo.heading, 1.0f };
            for (int i = 0; i < num_particles_; i++) b->particles.push_back(obs_p);
        } else if (!eo.alive && eo.visible) {
            // Confirmed dead: pin particles to a dead state so rollouts score correctly.
            b->visible = true;
            b->ever_seen = true;
            b->particles.clear();
            EnemyParticle dead{ eo.pos, {0,0}, 0.0f, 0.0f, 1.0f };
            for (int i = 0; i < num_particles_; i++) b->particles.push_back(dead);
        } else {
            // Hidden. If we *just* lost contact (was visible last call),
            // reseed cloud around the last-seen pose with some spread.
            if (b->visible) {
                // Transitioning from visible to hidden: spread.
                Vec2 center = b->particles.empty()
                    ? Vec2{0,0}
                    : b->particles.front().pos;
                seed_around_(*b, center, motion_.spread_on_loss);
            }
            b->visible = false;
            b->last_seen_elapsed = eo.last_seen_elapsed;
        }
    }
}

std::unordered_map<int, EnemyParticle> TeamBelief::sample(std::mt19937_64& rng) const {
    std::unordered_map<int, EnemyParticle> out;
    out.reserve(enemies_.size());
    for (const auto& b : enemies_) {
        if (b.particles.empty()) continue;
        std::uniform_int_distribution<size_t> pick(0, b.particles.size() - 1);
        out.emplace(b.enemy_id, b.particles[pick(rng)]);
    }
    return out;
}

std::unordered_map<int, EnemyParticle> TeamBelief::mean() const {
    std::unordered_map<int, EnemyParticle> out;
    out.reserve(enemies_.size());
    for (const auto& b : enemies_) {
        if (b.particles.empty()) continue;
        double sx = 0, sz = 0, svx = 0, svz = 0, shp = 0, sh = 0;
        double sw = 0;
        for (const auto& p : b.particles) {
            sx += p.pos.x * p.weight;
            sz += p.pos.z * p.weight;
            svx += p.vel.x * p.weight;
            svz += p.vel.z * p.weight;
            shp += p.hp * p.weight;
            sh += p.heading * p.weight;
            sw += p.weight;
        }
        if (sw <= 0) sw = 1.0;
        EnemyParticle m;
        m.pos = { static_cast<float>(sx / sw), static_cast<float>(sz / sw) };
        m.vel = { static_cast<float>(svx / sw), static_cast<float>(svz / sw) };
        m.hp  = static_cast<float>(shp / sw);
        m.heading = static_cast<float>(sh / sw);
        m.weight = 1.0f;
        out.emplace(b.enemy_id, m);
    }
    return out;
}

float TeamBelief::effective_sample_size() const {
    if (enemies_.empty()) return 0.0f;
    double total = 0.0;
    int    count = 0;
    for (const auto& b : enemies_) {
        if (b.particles.empty()) continue;
        double sum = 0, sum_sq = 0;
        for (const auto& p : b.particles) { sum += p.weight; sum_sq += p.weight * p.weight; }
        if (sum_sq <= 0) continue;
        double ess = (sum * sum) / sum_sq;
        total += ess / static_cast<double>(b.particles.size());
        count++;
    }
    return count > 0 ? static_cast<float>(total / count) : 0.0f;
}

} // namespace brogameagent::belief
