#include "brogameagent/simulation.h"

namespace brogameagent {

void Simulation::addPolicy(int agentId, PolicyFn fn) {
    if (!fn) { removePolicy(agentId); return; }
    if (stepDepth_ > 0) {
        pending_.push_back({ agentId, std::move(fn) });
        return;
    }
    policies_[agentId] = std::move(fn);
}

void Simulation::removePolicy(int agentId) {
    if (stepDepth_ > 0) {
        pending_.push_back({ agentId, PolicyFn{} });
        return;
    }
    policies_.erase(agentId);
}

bool Simulation::hasPolicy(int agentId) const {
    for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
        if (it->agentId == agentId) return static_cast<bool>(it->fn);
    }
    return policies_.count(agentId) != 0;
}

void Simulation::applyPending_() {
    // Swap out first: a policy destroyed here may own state whose destructor
    // calls back into add/removePolicy, which then applies directly.
    std::vector<PendingChange> changes;
    changes.swap(pending_);
    for (auto& c : changes) {
        if (c.fn) policies_[c.agentId] = std::move(c.fn);
        else      policies_.erase(c.agentId);
    }
}

void Simulation::step(float dt) {
    // Policies run while the map is iterated (and while their own function
    // object executes), so add/removePolicy queue until the outermost step
    // ends — including when a policy throws.
    struct DepthGuard {
        Simulation& s;
        explicit DepthGuard(Simulation& sim) : s(sim) { ++s.stepDepth_; }
        ~DepthGuard() {
            if (--s.stepDepth_ == 0 && !s.pending_.empty()) s.applyPending_();
        }
    } guard(*this);

    // Policy agents get their action chosen and applied; scripted agents
    // run their pathing update. We loop agents directly rather than calling
    // World::tick, so a policy-driven agent isn't double-integrated. Indexed,
    // re-reading the vector each time: a policy may add agents to the world.
    const auto& agents = world_.agents();
    for (size_t i = 0; i < agents.size(); ++i) {
        Agent* a = agents[i];
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
