#pragma once

#include "types.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brogameagent {

class Agent;
class World;
struct Unit;

/// Stable ids for the built-in capabilities. JS-registered caps use ids
/// starting at kJsCapFirst, allocated by the JS binding layer.
enum CapabilityId : int {
    kCapNone        = -1,
    kCapMoveTo      = 0,
    kCapLaneWalk    = 1,
    kCapBasicAttack = 2,
    kCapCastAbility = 3,
    kCapFlee        = 4,
    kCapHold        = 5,
    kCapBuiltinCount = 6,
    kJsCapFirst     = 100,
};

/// An in-flight action picked by the policy/JS think(). One per binding at a
/// time. Members beyond the first three are cap-specific scratch.
struct Action {
    int   capId   = kCapNone;
    bool  done    = true;
    float elapsed = 0.0f;

    // Generic payload. Each cap documents how it reads these.
    float fx = 0, fz = 0;  // target coords, flee point
    int   i0 = -1;         // target id, ability slot, waypoint index
    int   i1 = -1;
    float dur = 0.0f;      // duration for blocking actions (0 = immediate)

    // Opaque state for JS-authored caps. The core library never dereferences
    // this; the JS binding owns the lifetime.
    void* jsState = nullptr;
};

class CapabilitySet;

/// Context passed to gate/start/advance. Bundled so sig stays stable.
struct CapContext {
    Agent*         self  = nullptr;
    Unit*          unit  = nullptr;
    World*         world = nullptr;
    CapabilitySet* caps  = nullptr; // enclosing set (for LaneWalk waypoints etc.)
    float          now   = 0.0f;
};

/// One capability = one "tool" an agent can invoke. Capabilities are
/// stateless w.r.t. a specific in-flight action — they mutate the Action
/// struct passed in. Long-lived per-binding state (e.g. lane waypoints)
/// lives on CapabilitySet.
class Capability {
public:
    virtual ~Capability() = default;

    virtual int         id()   const = 0;
    virtual const char* name() const = 0;

    /// Cheap pre-flight check: can this cap be chosen right now? Used to
    /// build a mask exposed to scripted policies and as a guard before
    /// start(). JS callers typically skip this and just call start() —
    /// start() must degrade gracefully when gate would fail.
    virtual bool gate(const CapContext& /*ctx*/) const { return true; }

    /// Begin executing. Reads action.fx/fz/i0/i1/dur that the caller filled
    /// in, then may overwrite them with scratch state.
    virtual void start(const CapContext& ctx, Action& action) = 0;

    /// Called every frame while the action is running. Must set action.done
    /// when complete. Cheap caps can just set done=true in start() and
    /// return immediately here.
    virtual void advance(const CapContext& ctx, Action& action, float dt) = 0;

    /// Called if the binding is torn down mid-action.
    virtual void cancel(const CapContext& /*ctx*/, Action& /*action*/) {}
};

/// A per-binding bag of capabilities. Owns its entries.
///
/// IDs are stable handles; a set never has two entries with the same id.
/// add() replaces on collision.
class CapabilitySet {
public:
    CapabilitySet() = default;
    CapabilitySet(CapabilitySet&&) = default;
    CapabilitySet& operator=(CapabilitySet&&) = default;
    CapabilitySet(const CapabilitySet&) = delete;
    CapabilitySet& operator=(const CapabilitySet&) = delete;

    /// Add or replace a capability.
    void add(std::unique_ptr<Capability> cap);

    /// Remove a capability by id. Returns true if something was removed.
    bool remove(int id);

    /// Lookup (nullptr if not present).
    Capability*       get(int id);
    const Capability* get(int id) const;
    bool              has(int id) const { return get(id) != nullptr; }

    /// Range over all entries.
    const std::vector<std::unique_ptr<Capability>>& entries() const { return caps_; }

    /// Bitmask of built-in-cap ids whose gate() currently passes. Bit N is
    /// set iff a capability with id==N is present AND its gate() is true.
    /// Only the first 32 ids are representable — built-ins fit comfortably.
    uint32_t buildBuiltinMask(const CapContext& ctx) const;

    /// Per-binding waypoint list used by LaneWalkCapability. Kept on the
    /// set because it's configuration specific to this binding, not to the
    /// capability class itself.
    void               setLaneWaypoints(std::vector<Vec2> wps) { laneWaypoints_ = std::move(wps); laneIdx_ = 0; }
    const std::vector<Vec2>& laneWaypoints() const { return laneWaypoints_; }
    int                laneIndex() const { return laneIdx_; }
    void               setLaneIndex(int idx) { laneIdx_ = idx; }

    /// Optional fallback point used by FleeCapability when no explicit
    /// direction is given (e.g. home base).
    void setFallbackPoint(Vec2 p) { fallback_ = p; hasFallback_ = true; }
    bool hasFallbackPoint() const { return hasFallback_; }
    Vec2 fallbackPoint() const { return fallback_; }

private:
    std::vector<std::unique_ptr<Capability>> caps_;
    std::vector<Vec2> laneWaypoints_;
    int  laneIdx_     = 0;
    Vec2 fallback_    {0, 0};
    bool hasFallback_ = false;
};

// ---------- Built-in capability factories -----------------------------------
//
// Each returns a fresh instance. Callers typically do
// capSet.add(makeBasicAttackCapability());
std::unique_ptr<Capability> makeMoveToCapability();
std::unique_ptr<Capability> makeLaneWalkCapability();
std::unique_ptr<Capability> makeBasicAttackCapability();
std::unique_ptr<Capability> makeCastAbilityCapability();
std::unique_ptr<Capability> makeFleeCapability();
std::unique_ptr<Capability> makeHoldCapability();

/// Convenience: add all six built-ins to a set.
void addAllBuiltinCapabilities(CapabilitySet& set);

} // namespace brogameagent
