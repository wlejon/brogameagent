#pragma once

#include "agent.h"
#include "world.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace brogameagent {

/// Fixed-dt rollout driver for headless training.
///
/// Register one policy per agent that should be controlled by a policy
/// (typically your NN). On each step(dt):
///   1. Every registered policy is invoked to produce an AgentAction.
///   2. World::applyAction runs movement, aim, attack, ability for that agent.
///   3. World::tick advances scripted (non-policy) agents and projectiles.
///   4. The step counter / elapsed time advances.
///
/// Non-policy agents still run their scripted A*/steering update via
/// World::tick — register them as agents with the world but do not call
/// addPolicy for them.
///
/// The harness does not own the world or agents; lifetime is the caller's.
class Simulation {
public:
    using PolicyFn = std::function<AgentAction(Agent& self, const World& world)>;

    explicit Simulation(World& world) : world_(world) {}

    /// Register / replace a policy for the agent with this Unit::id.
    /// Policy agents are skipped by World::tick's scripted update — the
    /// Simulation drives them via applyAction instead.
    ///
    /// Called from inside a policy while step() runs, the change is queued
    /// and applied, in call order, when the outermost step() returns: the
    /// step finishes with the policy set it started with (a policy may
    /// replace or remove itself without destroying the function running).
    void addPolicy(int agentId, PolicyFn fn);

    /// Remove a policy (the agent will fall back to scripted World::tick).
    /// Deferred like addPolicy when called during step().
    void removePolicy(int agentId);

    /// Whether `agentId` has a policy, counting changes queued by a step in
    /// progress.
    bool hasPolicy(int agentId) const;

    /// One fixed-dt step. Order: policies → applyAction → world.tick.
    void step(float dt);

    /// Convenience: call step(dt) n times.
    void runSteps(float dt, int n);

    int   steps() const { return steps_; }
    float elapsed() const { return elapsed_; }

    /// Reset counters. Does NOT reset world state.
    void resetCounters();

private:
    struct PendingChange {
        int agentId;
        PolicyFn fn;   // empty = remove
    };

    void applyPending_();

    World& world_;
    std::unordered_map<int, PolicyFn> policies_;
    std::vector<PendingChange> pending_;
    int stepDepth_ = 0;   // > 0 while step() runs (re-entrant steps nest)
    int   steps_ = 0;
    float elapsed_ = 0.0f;
};

} // namespace brogameagent
