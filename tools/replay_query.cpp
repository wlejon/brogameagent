// replay_query — inspect and slice .bgar recordings from the command line.
//
// Usage:
//   replay_query info     <file>
//   replay_query roster   <file>
//   replay_query frame    <file> <frame_index>
//   replay_query step     <file> <step_idx>
//   replay_query agent    <file> <agent_id>
//   replay_query events   <file> [attacker_id]
//   replay_query dps      <file>            (summary of damage by pair)
//   replay_query dump     <file>            (every frame, tab-separated)
//
// All numeric arguments are decimal integers.

#include "brogameagent/replay_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace brogameagent;
using namespace brogameagent::replay;

namespace {

int usage() {
    std::fprintf(stderr,
        "usage: replay_query <cmd> <file> [args...]\n"
        "  info     <file>\n"
        "  roster   <file>\n"
        "  frame    <file> <frame_index>\n"
        "  step     <file> <step_idx>\n"
        "  agent    <file> <agent_id>\n"
        "  events   <file> [attacker_id]\n"
        "  dps      <file>\n"
        "  dump     <file>\n"
    );
    return 2;
}

bool openOrDie(ReplayReader& r, const char* path) {
    if (!r.open(path)) {
        std::fprintf(stderr, "error: %s (%s)\n", r.errorMessage().c_str(), path);
        return false;
    }
    return true;
}

void printFrame(const ReplayReader::Frame& f) {
    std::printf("frame step=%u  elapsed=%.3f  live=%u  proj=%u  events=%u\n",
                f.header.stepIdx, f.header.elapsed,
                f.header.liveCount, f.header.projCount, f.header.eventCount);
    for (const auto& a : f.agents) {
        std::printf("  agent id=%d  x=%.3f  z=%.3f  vx=%.3f  vz=%.3f  "
                    "yaw=%.3f  hp=%.2f  cd=%.3f  alive=%d\n",
                    a.id, a.x, a.z, a.vx, a.vz, a.yaw, a.hp, a.attackCooldown,
                    (a.flags & AGENT_FLAG_ALIVE) ? 1 : 0);
    }
    for (const auto& p : f.projectiles) {
        std::printf("  proj  id=%d  owner=%d  team=%d  x=%.3f  z=%.3f  "
                    "vx=%.3f  vz=%.3f  mode=%u  alive=%u\n",
                    p.id, p.ownerId, p.teamId, p.x, p.z, p.vx, p.vz,
                    p.mode, p.alive);
    }
    for (const auto& e : f.events) {
        std::printf("  event attacker=%d  target=%d  amount=%.2f  "
                    "kind=%u  killed=%u\n",
                    e.attackerId, e.targetId, e.amount, e.kind, e.killed);
    }
}

int cmdInfo(ReplayReader& r) {
    const auto& h = r.header();
    std::printf("magic=0x%08X version=%u episodeId=%llu seed=%llu dt=%.6f\n",
                h.magic, h.version,
                static_cast<unsigned long long>(h.episodeId),
                static_cast<unsigned long long>(h.seed),
                h.dt);
    std::printf("frames=%zu  rosterSize=%zu\n",
                r.frameCount(), r.roster().size());
    if (r.frameCount() > 0) {
        auto first = r.frame(0);
        auto last  = r.frame(r.frameCount() - 1);
        std::printf("stepRange=[%u..%u]  elapsedRange=[%.3f..%.3f]\n",
                    first.header.stepIdx, last.header.stepIdx,
                    first.header.elapsed, last.header.elapsed);
    }
    return 0;
}

int cmdRoster(ReplayReader& r) {
    std::printf("id\tteam\tmaxHp\tmaxMana\tradius\tatkRng\n");
    for (const auto& a : r.roster()) {
        std::printf("%d\t%d\t%.2f\t%.2f\t%.3f\t%.2f\n",
                    a.id, a.teamId, a.maxHp, a.maxMana, a.radius, a.attackRange);
    }
    return 0;
}

int cmdFrame(ReplayReader& r, int idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= r.frameCount()) {
        std::fprintf(stderr, "frame index out of range (0..%zu)\n",
                     r.frameCount() ? r.frameCount() - 1 : 0);
        return 1;
    }
    printFrame(r.frame(static_cast<size_t>(idx)));
    return 0;
}

int cmdStep(ReplayReader& r, uint32_t step) {
    size_t i = r.findByStep(step);
    if (i == SIZE_MAX) {
        std::fprintf(stderr, "step %u not in recording\n", step);
        return 1;
    }
    printFrame(r.frame(i));
    return 0;
}

int cmdAgent(ReplayReader& r, int32_t agentId) {
    auto traj = r.trajectory(agentId);
    std::printf("step\telapsed\tx\tz\thp\talive\n");
    for (const auto& p : traj) {
        std::printf("%u\t%.3f\t%.3f\t%.3f\t%.2f\t%d\n",
                    p.stepIdx, p.elapsed, p.x, p.z, p.hp, p.alive ? 1 : 0);
    }
    return 0;
}

int cmdEvents(ReplayReader& r, bool filter, int32_t attackerId) {
    std::printf("step\tattacker\ttarget\tamount\tkind\tkilled\n");
    for (size_t i = 0; i < r.frameCount(); i++) {
        auto f = r.frame(i);
        for (const auto& e : f.events) {
            if (filter && e.attackerId != attackerId) continue;
            std::printf("%u\t%d\t%d\t%.2f\t%u\t%u\n",
                        f.header.stepIdx,
                        e.attackerId, e.targetId, e.amount,
                        e.kind, e.killed);
        }
    }
    return 0;
}

int cmdDps(ReplayReader& r) {
    auto rows = r.damageSummary();
    std::printf("attacker\ttarget\ttotalDmg\thits\tkills\n");
    for (const auto& s : rows) {
        std::printf("%d\t%d\t%.2f\t%u\t%u\n",
                    s.attackerId, s.targetId, s.totalDamage, s.hits, s.kills);
    }
    return 0;
}

int cmdDump(ReplayReader& r) {
    std::printf("step\telapsed\tid\tx\tz\thp\talive\n");
    for (size_t i = 0; i < r.frameCount(); i++) {
        auto f = r.frame(i);
        for (const auto& a : f.agents) {
            std::printf("%u\t%.3f\t%d\t%.3f\t%.3f\t%.2f\t%d\n",
                        f.header.stepIdx, f.header.elapsed,
                        a.id, a.x, a.z, a.hp,
                        (a.flags & AGENT_FLAG_ALIVE) ? 1 : 0);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* cmd  = argv[1];
    const char* path = argv[2];

    ReplayReader r;
    if (!openOrDie(r, path)) return 1;

    if (!std::strcmp(cmd, "info"))   return cmdInfo(r);
    if (!std::strcmp(cmd, "roster")) return cmdRoster(r);
    if (!std::strcmp(cmd, "frame")) {
        if (argc < 4) return usage();
        return cmdFrame(r, std::atoi(argv[3]));
    }
    if (!std::strcmp(cmd, "step")) {
        if (argc < 4) return usage();
        return cmdStep(r, static_cast<uint32_t>(std::atoll(argv[3])));
    }
    if (!std::strcmp(cmd, "agent")) {
        if (argc < 4) return usage();
        return cmdAgent(r, std::atoi(argv[3]));
    }
    if (!std::strcmp(cmd, "events")) {
        if (argc == 3) return cmdEvents(r, false, 0);
        return cmdEvents(r, true, std::atoi(argv[3]));
    }
    if (!std::strcmp(cmd, "dps"))  return cmdDps(r);
    if (!std::strcmp(cmd, "dump")) return cmdDump(r);

    return usage();
}
