// 05_simulation — the Simulation harness with per-agent policies.
//
// Demonstrates:
//   - Registering a PolicyFn with Simulation::addPolicy.
//   - Simulation drives policies → applyAction; scripted agents still tick
//     via World::tick.
//   - Deterministic seed: same seed → identical outcome.

#include "brogameagent/agent.h"
#include "brogameagent/simulation.h"
#include "brogameagent/world.h"

#include <cmath>
#include <cstdio>

using namespace brogameagent;

namespace {
// Scripted turret policy: face the nearest enemy and auto-attack. No
// movement — keeps the example focused on the harness itself. World::
// resolveAttack handles cooldown and range gating internally.
AgentAction turret_policy(Agent& self, const World& world) {
    AgentAction a;
    Agent* ne = world.nearestEnemy(self);
    if (!ne) return a;
    float dx = ne->x() - self.x();
    float dz = ne->z() - self.z();
    a.aimYaw = std::atan2(dx, -dz);
    a.attackTargetId = ne->unit().id;
    return a;
}
} // namespace

int main() {
    World world;
    world.seed(0xBEEF);

    // Policy agent (a long-range shooter) vs a slow scripted dummy.
    Agent a; a.unit().id = 1; a.unit().teamId = 0;
    a.unit().hp = 100; a.unit().damage = 15; a.unit().attackRange = 8;
    a.unit().attacksPerSec = 2;
    a.setPosition(-5, 0); a.setMaxAccel(30); a.setMaxTurnRate(10);
    world.addAgent(&a);

    Agent b; b.unit().id = 2; b.unit().teamId = 1;
    b.unit().hp = 60; b.unit().damage = 6; b.unit().attackRange = 3;
    b.unit().attacksPerSec = 1;
    b.setPosition(3, 0); b.setMaxAccel(30); b.setMaxTurnRate(10);
    world.addAgent(&b);

    Simulation sim(world);
    sim.addPolicy(a.unit().id, turret_policy);
    // Agent b has no policy → World::tick drives it (scripted; no target → idle).

    const float dt = 0.05f;
    while (a.unit().alive() && b.unit().alive() && sim.steps() < 400) {
        sim.step(dt);
    }
    std::printf("events logged: %zu\n", world.events().size());
    std::printf("ended after %d steps (%.2fs)\n", sim.steps(), sim.elapsed());
    std::printf("agent 1: hp=%.1f %s\n", a.unit().hp, a.unit().alive() ? "alive" : "dead");
    std::printf("agent 2: hp=%.1f %s\n", b.unit().hp, b.unit().alive() ? "alive" : "dead");
    return 0;
}
