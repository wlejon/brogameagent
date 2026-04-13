#include "brogameagent/simulation.h"

namespace brogameagent {

void Simulation::addPolicy(int agentId, PolicyFn fn) {
    policies_[agentId] = std::move(fn);
}

void Simulation::removePolicy(int agentId) {
    policies_.erase(agentId);
}

void Simulation::step(float dt) {
    // Policy agents get their action chosen and applied; scripted agents
    // run their pathing update. We loop agents directly rather than calling
    // World::tick, so a policy-driven agent isn't double-integrated.
    for (Agent* a : world_.agents()) {
        auto it = policies_.find(a->unit().id);
        if (it != policies_.end()) {
            if (!a->unit().alive()) {
                a->unit().tickCooldowns(dt);
                continue;
            }
            AgentAction act = it->second(*a, world_);
            world_.applyAction(*a, act, dt);
        } else {
            if (a->unit().alive()) a->update(dt);
            a->unit().tickCooldowns(dt);
        }
    }

    world_.stepProjectiles(dt);
    world_.cullProjectiles();

    steps_++;
    elapsed_ += dt;
}

void Simulation::runSteps(float dt, int n) {
    for (int i = 0; i < n; i++) step(dt);
}

void Simulation::resetCounters() {
    steps_ = 0;
    elapsed_ = 0.0f;
}

} // namespace brogameagent
