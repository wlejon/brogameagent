#pragma once

#include "types.h"
#include <vector>

namespace brogameagent {

class Agent;

/// Container for a set of Agents and shared obstacles. The substrate for
/// perception and observation queries: "who can I see", "nearest enemy", etc.
///
/// The World does not own its agents — register raw pointers and keep them
/// alive externally (typically in an ECS or an array on the game side).
/// Agents must have distinct Unit::id values for attack-target bookkeeping.
class World {
public:
    void addAgent(Agent* agent);
    void removeAgent(const Agent* agent);
    void addObstacle(const AABB& box);

    /// Advance all agents by dt. Calls Agent::update() on each (scripted
    /// path-following). Policy-driven agents should NOT be tick()'d —
    /// call applyAction() per agent from your own loop.
    void tick(float dt);

    const std::vector<Agent*>& agents() const { return agents_; }
    const std::vector<AABB>& obstacles() const { return obstacles_; }

    /// Returns all living agents on a different team.
    std::vector<Agent*> enemiesOf(const Agent& self) const;

    /// Returns all living agents on the same team (excluding self).
    std::vector<Agent*> alliesOf(const Agent& self) const;

    /// Nearest living enemy, or nullptr. Ignores LOS.
    Agent* nearestEnemy(const Agent& self) const;

    /// Enemies within `range` of self (euclidean 2D). Sorted nearest-first.
    std::vector<Agent*> enemiesInRange(const Agent& self, float range) const;

    /// Lookup by Unit::id. Linear scan — fine for small rosters.
    Agent* findById(int id) const;

private:
    std::vector<Agent*> agents_;
    std::vector<AABB> obstacles_;
};

} // namespace brogameagent
