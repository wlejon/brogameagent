#include "brogameagent/policy.h"
#include "brogameagent/agent.h"
#include "brogameagent/unit.h"
#include "brogameagent/world.h"

namespace brogameagent {

bool ScriptedMinionPolicy::decide(const CapContext& ctx,
                                  const CapabilitySet& caps,
                                  Action& out) {
    if (!ctx.self || !ctx.unit || !ctx.world || !ctx.unit->alive()) return false;

    // 1) In-range enemy -> basic attack (if cap present and cooldown ready).
    if (caps.has(kCapBasicAttack) && ctx.unit->attackCooldown <= 0.0f) {
        auto inRange = ctx.world->enemiesInRange(*ctx.self, ctx.unit->attackRange);
        if (!inRange.empty()) {
            out.capId = kCapBasicAttack;
            out.i0 = inRange.front()->unit().id;
            return true;
        }
    }

    // 2) Lane walk if waypoints configured.
    if (caps.has(kCapLaneWalk) && !caps.laneWaypoints().empty()) {
        out.capId = kCapLaneWalk;
        return true;
    }

    // 3) Hold.
    if (caps.has(kCapHold)) {
        out.capId = kCapHold;
        return true;
    }
    return false;
}

std::unique_ptr<Policy> makeScriptedMinionPolicy() {
    return std::make_unique<ScriptedMinionPolicy>();
}

} // namespace brogameagent
