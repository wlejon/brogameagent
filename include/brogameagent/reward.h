#pragma once

namespace brogameagent {

class Agent;
class World;

/// Per-agent reward-delta accumulator. Attach once (`reset()` captures the
/// starting state), then call `consume()` every training tick to retrieve
/// the deltas since the last call. Deltas accumulate from:
///   - HP damage dealt (sum of damage events where attackerId matches)
///   - HP damage taken (sum of damage events where targetId matches)
///   - Kills credited (damage events with killed==true and attackerId match)
///   - Deaths (own "alive → dead" transition)
///   - Distance travelled on the XZ plane
///
/// Reads the World's event log — make sure you don't call World::clearEvents()
/// in between a tracker's `consume()` calls, or the deltas for the intervening
/// window will be lost. A typical loop: all trackers consume, then clear.
class RewardTracker {
public:
    struct Delta {
        float damageDealt = 0.0f;
        float damageTaken = 0.0f;
        int   kills = 0;
        int   deaths = 0;          // 0 or 1 per tick typically
        float distanceTravelled = 0.0f;
    };

    /// Capture starting state. Must be called before consume().
    void reset(const Agent& self, const World& world);

    /// Return the delta since the previous consume() (or reset()) and
    /// latch the new baseline.
    Delta consume(const Agent& self, const World& world);

private:
    int   agentId_     = 0;
    bool  wasAlive_    = true;
    float lastHp_      = 0.0f;
    float lastX_       = 0.0f;
    float lastZ_       = 0.0f;
    int   lastEventIdx_ = 0; // index into World::events() already accounted for
};

} // namespace brogameagent
