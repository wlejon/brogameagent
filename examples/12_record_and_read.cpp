// 12_record_and_read — write a .bgar replay, then read it back.
//
// Demonstrates:
//   - Recorder: open / writeRoster / recordFrame-per-step / close.
//   - ReplayReader: header, roster, frame-random-access, trajectory query.
//
// The on-disk schema (replay_format.h) is a POD, little-endian binary
// format with a footer index for O(1) frame seek.

#include "brogameagent/agent.h"
#include "brogameagent/recorder.h"
#include "brogameagent/replay_reader.h"
#include "brogameagent/simulation.h"
#include "brogameagent/world.h"

#include <cmath>
#include <cstdio>

using namespace brogameagent;

namespace {
AgentAction aggressive_policy(Agent& self, const World& world) {
    AgentAction a;
    Agent* ne = world.nearestEnemy(self);
    if (!ne) return a;
    float dx = ne->x() - self.x();
    float dz = ne->z() - self.z();
    a.aimYaw = std::atan2(dx, -dz);
    a.moveZ = -1.0f;
    float r = self.unit().attackRange;
    if (self.unit().attackCooldown <= 0 && dx*dx + dz*dz <= r*r) {
        a.attackTargetId = ne->unit().id;
    }
    return a;
}
} // namespace

int main() {
    const char* path = "example_episode.bgar";

    // ---- Record a short fight. ----
    {
        World world; world.seed(0x9E7);
        Agent a; a.unit().id = 1; a.unit().teamId = 0;
        a.unit().hp = 100; a.unit().damage = 12; a.unit().attackRange = 3;
        a.unit().attacksPerSec = 2;
        a.setPosition(-2, 0); a.setMaxAccel(30); a.setMaxTurnRate(10);
        world.addAgent(&a);

        Agent b; b.unit().id = 2; b.unit().teamId = 1;
        b.unit().hp = 80; b.unit().damage = 8; b.unit().attackRange = 3;
        b.unit().attacksPerSec = 1;
        b.setPosition(2, 0); b.setMaxAccel(30); b.setMaxTurnRate(10);
        world.addAgent(&b);

        Simulation sim(world);
        sim.addPolicy(1, aggressive_policy);
        sim.addPolicy(2, aggressive_policy);

        const float dt = 0.05f;
        Recorder rec;
        if (!rec.open(path, /*episodeId*/ 42, /*seed*/ 0x9E7, dt)) {
            std::fprintf(stderr, "failed to open %s for write\n", path);
            return 1;
        }
        rec.writeRoster(world.agents());
        for (int step = 0; step < 200 && a.unit().alive() && b.unit().alive(); step++) {
            sim.step(dt);
            rec.recordFrame((uint32_t)step, sim.elapsed(), world);
        }
        size_t frames = rec.frameCount();   // query before close() resets
        rec.close();
        std::printf("wrote %s (%zu frames)\n", path, frames);
    }

    // ---- Read it back. ----
    {
        ReplayReader reader;
        if (!reader.open(path)) {
            std::fprintf(stderr, "read failed: %s\n", reader.errorMessage().c_str());
            return 1;
        }
        const auto& h = reader.header();
        std::printf("header: episode=%llu seed=%llu dt=%.3f frames=%zu\n",
            (unsigned long long)h.episodeId, (unsigned long long)h.seed,
            h.dt, reader.frameCount());
        std::printf("roster (%zu agents):\n", reader.roster().size());
        for (const auto& ag : reader.roster()) {
            std::printf("  id=%d team=%d maxHp=%.0f range=%.1f\n",
                ag.id, ag.teamId, ag.maxHp, ag.attackRange);
        }
        // Random-access: fetch the last frame.
        size_t last = reader.frameCount() - 1;
        auto fr = reader.frame(last);
        std::printf("last frame: step=%u elapsed=%.2f agents=%zu events=%zu\n",
            fr.header.stepIdx, fr.header.elapsed,
            fr.agents.size(), fr.events.size());

        // Trajectory: (x, z, hp) per step for agent 1.
        auto traj = reader.trajectory(1);
        if (!traj.empty()) {
            const auto& first = traj.front();
            const auto& lastp = traj.back();
            std::printf("agent 1 trajectory: %zu points, start=(%.2f,%.2f hp=%.1f) end=(%.2f,%.2f hp=%.1f)\n",
                traj.size(), first.x, first.z, first.hp, lastp.x, lastp.z, lastp.hp);
        }
    }
    return 0;
}
