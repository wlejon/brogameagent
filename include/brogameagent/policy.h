#pragma once

#include "capability.h"
#include <memory>

namespace brogameagent {

/// A policy picks one capability to start, given the current state.
/// The JS-primary path bypasses this entirely: a JS think(self, world)
/// function writes directly into the binding's pending Action. Policies
/// are the C++ escape hatch — useful for bulk minion behaviour where
/// the per-tick JS call overhead would be wasteful.
class Policy {
public:
    virtual ~Policy() = default;

    /// Fill `out` with a chosen capability id + its start args. Return true
    /// if a choice was made, false to fall through to a Hold.
    ///
    /// The returned action is passed to CapabilitySet::get(out.capId)->start()
    /// by the binding — so the policy only needs to set capId and the
    /// caller-filled fields (fx/fz/i0/i1), not scratch state.
    virtual bool decide(const CapContext& ctx, const CapabilitySet& caps,
                        Action& out) = 0;
};

/// A sensible default for "dumb" minions:
///   1. If any enemy is in attack range, start BasicAttack on the nearest.
///   2. Else if LaneWalk is available, advance along waypoints.
///   3. Else Hold.
///
/// Uses the world's perception queries (nearestEnemy, enemiesInRange).
/// Does not read the Observation or ActionMask.
class ScriptedMinionPolicy : public Policy {
public:
    bool decide(const CapContext& ctx, const CapabilitySet& caps,
                Action& out) override;
};

std::unique_ptr<Policy> makeScriptedMinionPolicy();

} // namespace brogameagent
