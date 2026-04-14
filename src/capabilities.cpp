// Built-in capabilities. Each is a small class implementing the
// Capability interface. Kept in one TU to avoid six near-identical files.
//
// Convention for the Action struct:
//   MoveTo       — fx,fz = world target
//   LaneWalk     — i0 = chosen waypoint index at start (informational)
//   BasicAttack  — i0 = target unit id; dur = swing time in seconds
//   CastAbility  — i0 = slot index; i1 = target unit id; dur = cast time
//   Flee         — fx,fz = computed retreat point
//   Hold         — dur = hold duration in seconds

#include "brogameagent/capability.h"
#include "brogameagent/agent.h"
#include "brogameagent/unit.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {
namespace {

// Fixed cast time for ability casts. Could later be moved onto AbilitySpec.
constexpr float kDefaultCastTime = 0.25f;

// If a waypoint is within this radius, consider it reached and advance.
constexpr float kLaneWaypointReachRadius = 1.2f;

// How far from the threat direction to project a flee target.
constexpr float kFleeDistance = 6.0f;

// ---------------------------------------------------------------------------

class MoveToCap : public Capability {
public:
    int id() const override { return kCapMoveTo; }
    const char* name() const override { return "move_to"; }

    void start(const CapContext& ctx, Action& a) override {
        if (ctx.self) ctx.self->setTarget(a.fx, a.fz);
        a.elapsed = 0;
        a.done = true; // non-blocking; movement continues in world.tick
    }
    void advance(const CapContext&, Action& a, float) override {
        a.done = true;
    }
};

// ---------------------------------------------------------------------------

class LaneWalkCap : public Capability {
public:
    int id() const override { return kCapLaneWalk; }
    const char* name() const override { return "lane_walk"; }

    bool gate(const CapContext&) const override {
        // Gated at the set level — the CapabilitySet carries the waypoints,
        // but the Capability itself doesn't see them. We trust the caller to
        // only add this cap when the set has waypoints.
        return true;
    }

    // LaneWalk is set-and-forget: advance the waypoint index if we're close
    // to the current one, then target the (possibly new) current waypoint.
    // Done immediately; the Agent keeps walking via world.tick().
    static void step(Agent& self, CapabilitySet& set) {
        const auto& wps = set.laneWaypoints();
        if (wps.empty()) return;
        int idx = set.laneIndex();
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(wps.size())) idx = static_cast<int>(wps.size()) - 1;

        const Vec2 wp = wps[idx];
        const float dx = self.x() - wp.x;
        const float dz = self.z() - wp.z;
        if (std::sqrt(dx*dx + dz*dz) <= kLaneWaypointReachRadius
            && idx + 1 < static_cast<int>(wps.size())) {
            idx++;
            set.setLaneIndex(idx);
        }
        self.setTarget(wps[idx].x, wps[idx].z);
    }

    void start(const CapContext& ctx, Action& a) override {
        if (ctx.self && ctx.caps) {
            step(*ctx.self, *ctx.caps);
            a.i0 = ctx.caps->laneIndex();
        }
        a.elapsed = 0;
        a.done = true;
    }
    void advance(const CapContext&, Action& a, float) override { a.done = true; }
};

// ---------------------------------------------------------------------------

class BasicAttackCap : public Capability {
public:
    int id() const override { return kCapBasicAttack; }
    const char* name() const override { return "basic_attack"; }

    bool gate(const CapContext& ctx) const override {
        return ctx.unit && ctx.unit->alive() && ctx.unit->attackCooldown <= 0.0f;
    }

    void start(const CapContext& ctx, Action& a) override {
        a.elapsed = 0;
        a.done = false;
        // Duration = one swing time at current effective APS (minimum 0.1s).
        float aps = ctx.unit ? ctx.unit->effectiveAttacksPerSec() : 1.0f;
        if (aps < 0.1f) aps = 0.1f;
        a.dur = 1.0f / aps;
        if (a.dur > 3.0f) a.dur = 3.0f;

        // Resolve the swing immediately at the start of the windup.
        // (Alternative: resolve at midpoint. We pick start for simplicity
        // and because the cooldown on the Unit already models recovery.)
        if (ctx.self && ctx.world && a.i0 >= 0) {
            ctx.world->resolveAttack(*ctx.self, a.i0);
        }
    }
    void advance(const CapContext&, Action& a, float dt) override {
        a.elapsed += dt;
        if (a.elapsed >= a.dur) a.done = true;
    }
};

// ---------------------------------------------------------------------------

class CastAbilityCap : public Capability {
public:
    int id() const override { return kCapCastAbility; }
    const char* name() const override { return "cast_ability"; }

    bool gate(const CapContext& ctx) const override {
        // Can't cheaply check which slot the policy plans to use. Require
        // at least one ability slot with cooldown ready.
        if (!ctx.unit || !ctx.unit->alive()) return false;
        for (int s = 0; s < Unit::MAX_ABILITIES; s++) {
            if (ctx.unit->abilitySlot[s] >= 0
                && ctx.unit->abilityCooldowns[s] <= 0.0f) return true;
        }
        return false;
    }

    void start(const CapContext& ctx, Action& a) override {
        a.elapsed = 0;
        a.done = false;
        a.dur = kDefaultCastTime;
        if (ctx.self && ctx.world && a.i0 >= 0 && a.i0 < Unit::MAX_ABILITIES) {
            ctx.world->resolveAbility(*ctx.self, a.i0, a.i1);
        }
    }
    void advance(const CapContext&, Action& a, float dt) override {
        a.elapsed += dt;
        if (a.elapsed >= a.dur) a.done = true;
    }
};

// ---------------------------------------------------------------------------

class FleeCap : public Capability {
public:
    int id() const override { return kCapFlee; }
    const char* name() const override { return "flee"; }

    void start(const CapContext& ctx, Action& a) override {
        a.elapsed = 0;
        a.done = true;
        if (!ctx.self) return;

        // If caller provided explicit retreat point, use it. Otherwise take a
        // direction opposite the nearest enemy.
        float tx = a.fx, tz = a.fz;
        const bool explicitPoint = (tx != 0.0f || tz != 0.0f);
        if (!explicitPoint && ctx.world) {
            if (Agent* e = ctx.world->nearestEnemy(*ctx.self)) {
                float dx = ctx.self->x() - e->x();
                float dz = ctx.self->z() - e->z();
                float len = std::sqrt(dx*dx + dz*dz);
                if (len > 0.001f) {
                    tx = ctx.self->x() + (dx / len) * kFleeDistance;
                    tz = ctx.self->z() + (dz / len) * kFleeDistance;
                } else {
                    tx = ctx.self->x() + kFleeDistance;
                    tz = ctx.self->z();
                }
            }
        }
        ctx.self->setTarget(tx, tz);
        a.fx = tx; a.fz = tz;
    }
    void advance(const CapContext&, Action& a, float) override { a.done = true; }
};

// ---------------------------------------------------------------------------

class HoldCap : public Capability {
public:
    int id() const override { return kCapHold; }
    const char* name() const override { return "hold"; }

    void start(const CapContext&, Action& a) override {
        a.elapsed = 0;
        // dur=0 → immediately done; think-rate accumulator provides the gap.
        a.done = (a.dur <= 0.0f);
    }
    void advance(const CapContext&, Action& a, float dt) override {
        a.elapsed += dt;
        if (a.elapsed >= a.dur) a.done = true;
    }
};

} // namespace

// ---------- factories -------------------------------------------------------

std::unique_ptr<Capability> makeMoveToCapability()      { return std::make_unique<MoveToCap>(); }
std::unique_ptr<Capability> makeLaneWalkCapability()    { return std::make_unique<LaneWalkCap>(); }
std::unique_ptr<Capability> makeBasicAttackCapability() { return std::make_unique<BasicAttackCap>(); }
std::unique_ptr<Capability> makeCastAbilityCapability() { return std::make_unique<CastAbilityCap>(); }
std::unique_ptr<Capability> makeFleeCapability()        { return std::make_unique<FleeCap>(); }
std::unique_ptr<Capability> makeHoldCapability()        { return std::make_unique<HoldCap>(); }

} // namespace brogameagent
