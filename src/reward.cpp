#include "brogameagent/reward.h"
#include "brogameagent/agent.h"
#include "brogameagent/world.h"
#include "brogameagent/unit.h"

#include <cmath>

namespace brogameagent {

void RewardTracker::reset(const Agent& self, const World& world) {
    agentId_      = self.unit().id;
    wasAlive_     = self.unit().alive();
    lastHp_       = self.unit().hp;
    lastX_        = self.x();
    lastZ_        = self.z();
    lastEventIdx_ = static_cast<int>(world.events().size());
}

RewardTracker::Delta RewardTracker::consume(const Agent& self, const World& world) {
    Delta d;

    // Events since our last cursor.
    const auto& events = world.events();
    int end = static_cast<int>(events.size());
    for (int i = lastEventIdx_; i < end; i++) {
        const DamageEvent& e = events[i];
        if (e.attackerId == agentId_) {
            d.damageDealt += e.amount;
            if (e.killed) d.kills += 1;
        }
        if (e.targetId == agentId_) {
            d.damageTaken += e.amount;
        }
    }
    lastEventIdx_ = end;

    // Own death transition.
    bool alive = self.unit().alive();
    if (wasAlive_ && !alive) d.deaths = 1;
    wasAlive_ = alive;
    lastHp_   = self.unit().hp;

    // Distance travelled.
    float dx = self.x() - lastX_;
    float dz = self.z() - lastZ_;
    d.distanceTravelled = std::sqrt(dx * dx + dz * dz);
    lastX_ = self.x();
    lastZ_ = self.z();

    return d;
}

} // namespace brogameagent
