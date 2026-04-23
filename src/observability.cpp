#include "brogameagent/observability.h"

#include "brogameagent/agent.h"
#include "brogameagent/perception.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace brogameagent::obs {

namespace {

AgentObservation snapshot_of(const Agent& a, bool visible, float now) {
    AgentObservation o;
    o.id       = a.unit().id;
    o.team_id  = a.unit().teamId;
    o.pos      = { a.x(), a.z() };
    o.vel      = a.velocity();
    o.hp       = a.unit().hp;
    o.max_hp   = a.unit().maxHp;
    o.heading  = a.yaw();
    o.alive    = a.unit().alive();
    o.visible  = visible;
    o.last_seen_elapsed = visible ? 0.0f : 0.0f;
    (void)now;
    return o;
}

// Does any living ally on `team` see `target`?
bool team_can_see(const World& world, int team, const Agent& target,
                  const VisibilityConfig& cfg) {
    const auto& obstacles = world.obstacles();
    const AABB* obs_ptr = obstacles.empty() ? nullptr : obstacles.data();
    const int   obs_n   = static_cast<int>(obstacles.size());

    for (Agent* a : world.agents()) {
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != team) continue;
        Vec2 from{ a->x(), a->z() };
        Vec2 to  { target.x(), target.z() };

        if (cfg.max_range > 0.0f) {
            float dx = to.x - from.x;
            float dz = to.z - from.z;
            if (dx * dx + dz * dz > cfg.max_range * cfg.max_range) continue;
        }
        // FOV: treat >= 2π as "omniscient cone" and skip the angular test.
        const float TWO_PI = 6.28318530717958647692f;
        if (cfg.fov_radians < TWO_PI - 1e-4f) {
            float dx = to.x - from.x;
            float dz = to.z - from.z;
            float angle_to = std::atan2(dx, -dz);
            float delta = std::fabs(wrapAngle(angle_to - a->yaw()));
            if (delta > cfg.fov_radians * 0.5f) continue;
        }
        if (cfg.check_los && !hasLineOfSight(from, to, obs_ptr, obs_n)) continue;
        return true;
    }
    return false;
}

} // namespace

TeamObservation observe(const World& world, int team_id,
                         const VisibilityConfig& cfg,
                         float now) {
    TeamObservation out;
    out.team_id   = team_id;
    out.timestamp = now;

    for (Agent* a : world.agents()) {
        if (a->unit().teamId == team_id) {
            out.allies.push_back(snapshot_of(*a, /*visible=*/true, now));
        } else {
            bool vis = a->unit().alive() && team_can_see(world, team_id, *a, cfg);
            AgentObservation o = snapshot_of(*a, vis, now);
            // For invisible enemies we clear current-position/velocity fields
            // to force the belief layer to deal with the lack of information.
            // Caller may overwrite via merge() using prior entries.
            if (!vis) {
                o.pos = {0, 0};
                o.vel = {0, 0};
                o.hp  = 0.0f;
                o.last_seen_elapsed = 0.0f;  // merge() fills this from prior
            }
            out.enemies.push_back(o);
        }
    }
    return out;
}

TeamObservation merge(const TeamObservation& prior,
                       const TeamObservation& fresh,
                       float now) {
    TeamObservation out = fresh;

    std::unordered_map<int, const AgentObservation*> prior_by_id;
    prior_by_id.reserve(prior.enemies.size());
    for (const auto& e : prior.enemies) prior_by_id.emplace(e.id, &e);

    const float dt = now - prior.timestamp;
    for (auto& e : out.enemies) {
        if (e.visible) { e.last_seen_elapsed = 0.0f; continue; }
        auto it = prior_by_id.find(e.id);
        if (it == prior_by_id.end()) {
            // Never seen; keep zeroes and mark "unknown" by large elapsed.
            e.last_seen_elapsed = std::numeric_limits<float>::infinity();
            continue;
        }
        const AgentObservation& p = *it->second;
        if (p.visible) {
            // Was visible last tick; carry its values forward with one dt.
            e.pos = p.pos;
            e.vel = p.vel;
            e.hp = p.hp;
            e.max_hp = p.max_hp;
            e.heading = p.heading;
            e.alive = p.alive;
            e.last_seen_elapsed = std::max(0.0f, dt);
        } else {
            e.pos = p.pos;
            e.vel = p.vel;
            e.hp = p.hp;
            e.max_hp = p.max_hp;
            e.heading = p.heading;
            e.alive = p.alive;
            e.last_seen_elapsed = p.last_seen_elapsed + std::max(0.0f, dt);
        }
    }
    return out;
}

} // namespace brogameagent::obs
