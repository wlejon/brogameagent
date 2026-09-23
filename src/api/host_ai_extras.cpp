// Host bindings for AI game extras: steering, observation, reward tracker,
// snapshots, projectiles, vectorized simulation, and evaluator/rollout primitives.

#include "host_ai_internal.h"
#include "host_ai_mcts_shared.h"
#include <brogameagent/brogameagent.h>
#include <brogameagent/reward.h>
#include <brogameagent/vec_simulation.h>
#include <cmath>
#include <vector>

namespace brogameagent::api {

HostClass g_agentSnapshotClass;
HostClass g_worldSnapshotClass;
HostClass g_vecSimClass;
HostClass g_rewardTrackerClass;

namespace {

struct HostVecSim {
    uint32_t tag = kHostVecSimTag;
    std::unique_ptr<brogameagent::VecSimulation> sim;
};

struct HostRewardTracker {
    uint32_t tag = kHostRewardTrackerTag;
    brogameagent::RewardTracker tracker;
};

HostVecSim* unwrapVecSim(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostVecSim*>(ev::handleData(v));
    return (h && h->tag == kHostVecSimTag) ? h : nullptr;
}

HostRewardTracker* unwrapRewardTracker(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostRewardTracker*>(ev::handleData(v));
    return (h && h->tag == kHostRewardTrackerTag) ? h : nullptr;
}

Value makeProjectileObject(const brogameagent::Projectile& p) {
    ObjectBuilder obj;
    obj.set("id", ev::fromDouble(p.id));
    obj.set("ownerId", ev::fromDouble(p.ownerId));
    obj.set("teamId", ev::fromDouble(p.teamId));
    obj.set("targetId", ev::fromDouble(p.targetId));
    obj.set("x", ev::fromDouble(p.x));
    obj.set("z", ev::fromDouble(p.z));
    obj.set("vx", ev::fromDouble(p.vx));
    obj.set("vz", ev::fromDouble(p.vz));
    obj.set("speed", ev::fromDouble(p.speed));
    obj.set("radius", ev::fromDouble(p.radius));
    obj.set("damage", ev::fromDouble(p.damage));
    obj.set("kind", ev::fromUtf8(damageKindStr(p.kind)));
    obj.set("remainingLife", ev::fromDouble(p.remainingLife));
    const char* modeStr = "single";
    if (p.mode == brogameagent::ProjectileMode::Pierce) modeStr = "pierce";
    else if (p.mode == brogameagent::ProjectileMode::AoE) modeStr = "aoe";
    obj.set("mode", ev::fromUtf8(modeStr));
    obj.set("splashRadius", ev::fromDouble(p.splashRadius));
    obj.set("maxHits", ev::fromDouble(p.maxHits));
    obj.set("alive", ev::fromBool(p.alive));
    return obj.get();
}

brogameagent::Projectile parseProjectileObj(Value opts) {
    ev::Persistent root(opts);
    brogameagent::Projectile p;
    p.ownerId = getI32Property(root.get(), "ownerId", -1);
    p.teamId = getI32Property(root.get(), "teamId", 0);
    p.targetId = getI32Property(root.get(), "targetId", -1);
    p.x = static_cast<float>(getDoubleProperty(root.get(), "x", 0));
    p.z = static_cast<float>(getDoubleProperty(root.get(), "z", 0));
    p.vx = static_cast<float>(getDoubleProperty(root.get(), "vx", 0));
    p.vz = static_cast<float>(getDoubleProperty(root.get(), "vz", 0));
    p.speed = static_cast<float>(getDoubleProperty(root.get(), "speed", 20));
    p.radius = static_cast<float>(getDoubleProperty(root.get(), "radius", 0.3));
    p.damage = static_cast<float>(getDoubleProperty(root.get(), "damage", 0));
    p.remainingLife = static_cast<float>(getDoubleProperty(root.get(), "remainingLife", 2));
    p.splashRadius = static_cast<float>(getDoubleProperty(root.get(), "splashRadius", 0));
    p.maxHits = getI32Property(root.get(), "maxHits", 0, 0);

    Value kindVal = ev::getProperty(root.get(), "kind");
    if (ev::isString(kindVal)) p.kind = parseDamageKind(ev::toUtf8(kindVal).c_str());

    Value modeVal = ev::getProperty(root.get(), "mode");
    if (ev::isString(modeVal)) {
        std::string m = ev::toUtf8(modeVal);
        if (m == "pierce") p.mode = brogameagent::ProjectileMode::Pierce;
        else if (m == "aoe") p.mode = brogameagent::ProjectileMode::AoE;
    }
    return p;
}

} // namespace

void ensureAIExtrasClassesInstalled() {
    // Per thread, like ensureAIClassesInstalled.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    // AgentSnapshot
    g_agentSnapshotClass.init("AIAgentSnapshot", [](ObjectBuilder& b) {
        b.accessor("id", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromDouble(s ? s->s.id : 0);
        }, nullptr);
        b.accessor("x", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromDouble(s ? s->s.x : 0);
        }, nullptr);
        b.accessor("z", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromDouble(s ? s->s.z : 0);
        }, nullptr);
        b.accessor("yaw", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromDouble(s ? s->s.yaw : 0);
        }, nullptr);
        b.accessor("hp", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromDouble(s ? s->s.unit.hp : 0);
        }, nullptr);
        b.accessor("alive", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapAgentSnapshot(self); return ev::fromBool(s ? s->s.unit.alive() : false);
        }, nullptr);
    });

    // WorldSnapshot
    g_worldSnapshotClass.init("AIWorldSnapshot", [](ObjectBuilder& b) {
        b.accessor("agentCount", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapWorldSnapshot(self); return ev::fromDouble(s ? static_cast<double>(s->s.agents.size()) : 0.0);
        }, nullptr);
        b.accessor("projectileCount", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapWorldSnapshot(self); return ev::fromDouble(s ? static_cast<double>(s->s.projectiles.size()) : 0.0);
        }, nullptr);
        b.accessor("eventCount", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapWorldSnapshot(self); return ev::fromDouble(s ? static_cast<double>(s->s.events.size()) : 0.0);
        }, nullptr);
        b.accessor("nextProjectileId", [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapWorldSnapshot(self); return ev::fromDouble(s ? s->s.nextProjectileId : 0);
        }, nullptr);
        b.def("projectiles", 0, [](Value self, std::span<const Value>) -> Value {
            auto* s = unwrapWorldSnapshot(self);
            if (!s) return hostArrayOf(0, [](size_t) { return ev::null(); });
            return hostArrayOf(s->s.projectiles.size(), [&](size_t i) {
                return makeProjectileObject(s->s.projectiles[i]);
            });
        });
    });

    // RewardTracker
    g_rewardTrackerClass.init("AIRewardTracker", [](ObjectBuilder& b) {
        b.def("consume", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* rt = unwrapRewardTracker(self);
            if (!rt || a.size() < 2) return ev::null();
            auto* ag = unwrapAgent(a[0]);
            auto* w = unwrapWorld(a[1]);
            if (!ag || !w) return ev::null();

            auto delta = rt->tracker.consume(ag->agent, w->world);
            ObjectBuilder o;
            o.set("damageDealt", ev::fromDouble(delta.damageDealt));
            o.set("damageTaken", ev::fromDouble(delta.damageTaken));
            o.set("kills", ev::fromDouble(delta.kills));
            o.set("deaths", ev::fromDouble(delta.deaths));
            o.set("distanceTravelled", ev::fromDouble(delta.distanceTravelled));
            return o.get();
        });

        b.def("reset", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* rt = unwrapRewardTracker(self);
            if (!rt || a.size() < 2) return ev::undefined();
            auto* ag = unwrapAgent(a[0]);
            auto* w = unwrapWorld(a[1]);
            if (ag && w) rt->tracker.reset(ag->agent, w->world);
            return ev::undefined();
        });
    });

    // VecSimulation
    g_vecSimClass.init("AIVecSimulation", [](ObjectBuilder& b) {
        b.accessor("numEnvs", [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self); return ev::fromDouble(vs && vs->sim ? vs->sim->numEnvs() : 0);
        }, nullptr);

        b.def("seedAndReset", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* vs = unwrapVecSim(self);
            if (vs && vs->sim && !a.empty()) vs->sim->seedAndReset(u64At(a, 0));
            return ev::undefined();
        });

        b.def("resetDone", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (vs && vs->sim) vs->sim->resetDone();
            return ev::undefined();
        });

        b.def("resetEnv", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* vs = unwrapVecSim(self);
            if (vs && vs->sim && !a.empty()) vs->sim->resetEnv(i32At(a, 0, "resetEnv: envIdx"));
            return ev::undefined();
        });

        b.def("observe", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim || a.empty()) return ev::null();
            int agentId = i32At(a, 0, "observe: agentId");
            int N = vs->sim->numEnvs();
            int total = N * brogameagent::observation::TOTAL;
            std::vector<float> buf(total);
            vs->sim->observe(agentId, buf.data());
            return makeFloat32Array(buf.data(), total);
        });

        b.def("actionMask", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim || a.empty()) return ev::null();
            int agentId = i32At(a, 0, "actionMask: agentId");
            int N = vs->sim->numEnvs();
            std::vector<float> mask(static_cast<size_t>(N) * brogameagent::action_mask::TOTAL);
            std::vector<int> ids(static_cast<size_t>(N) * brogameagent::action_mask::N_ENEMY_SLOTS);
            vs->sim->actionMask(agentId, mask.data(), ids.data());
            ObjectBuilder o;
            o.set("mask", makeFloat32Array(mask.data(), mask.size()));
            o.set("enemyIds", makeInt32Array(ids.data(), ids.size()));
            return o.get();
        });

        b.def("applyActions", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim || a.size() < 2) return ev::undefined();
            int agentId = i32At(a, 0, "applyActions: agentId");
            // a[1] is a rooted slot and is re-read each time; a local copy
            // would be stale after the first getProperty.
            if (!ev::isObject(a[1])) return ev::throwTypeError("applyActions: actions must be an array");
            int N = vs->sim->numEnvs();
            std::vector<brogameagent::AgentAction> acts(static_cast<size_t>(N));
            Value lenV = ev::getProperty(a[1], "length");
            const int n = static_cast<int>(std::min<uint32_t>(toLength(lenV),
                                                              static_cast<uint32_t>(N)));
            for (int i = 0; i < n; i++) {
                Value e = ev::getElement(a[1], static_cast<uint32_t>(i));
                if (ev::isObject(e)) acts[i] = parseAgentAction(e);
            }
            vs->sim->applyActions(agentId, acts.data());
            return ev::undefined();
        });

        b.def("step", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (vs && vs->sim) vs->sim->step();
            return ev::undefined();
        });

        b.def("dones", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim) return ev::null();
            int N = vs->sim->numEnvs();
            std::vector<int> done(N), win(N);
            vs->sim->dones(done.data(), win.data());
            ObjectBuilder o;
            o.set("done", makeUint32Array(reinterpret_cast<const uint32_t*>(done.data()), N));
            o.set("winner", makeUint32Array(reinterpret_cast<const uint32_t*>(win.data()), N));
            return o.get();
        });

        b.def("rewards", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim) return ev::null();
            int N = vs->sim->numEnvs();
            std::vector<float> rh(N), ro(N);
            vs->sim->rewards(rh.data(), ro.data());
            ObjectBuilder o;
            o.set("hero", makeFloat32Array(rh.data(), N));
            o.set("opponent", makeFloat32Array(ro.data(), N));
            return o.get();
        });

        b.def("stepCounts", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim) return ev::null();
            int N = vs->sim->numEnvs();
            std::vector<int> v(N);
            vs->sim->stepCounts(v.data());
            return makeInt32Array(v.data(), N);
        });

        b.def("episodeCounts", 0, [](Value self, std::span<const Value>) -> Value {
            auto* vs = unwrapVecSim(self);
            if (!vs || !vs->sim) return ev::null();
            int N = vs->sim->numEnvs();
            std::vector<int> v(N);
            vs->sim->episodeCounts(v.data());
            return makeInt32Array(v.data(), N);
        });
    });
}

void installAIExtras(ObjectBuilder& game) {
    ensureAIExtrasClassesInstalled();

    // ── Steering ─────────────────────────────────────────────────────────────
    ObjectBuilder steer;
    steer.def("seek", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 4) return ev::null();
        return makeSteeringOutput(brogameagent::seek(
            {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
            {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))}));
    });

    steer.def("arrive", 5, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 5) return ev::null();
        return makeSteeringOutput(brogameagent::arrive(
            {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
            {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))},
            static_cast<float>(numAt(a, 4))));
    });

    steer.def("flee", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 4) return ev::null();
        return makeSteeringOutput(brogameagent::flee(
            {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
            {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))}));
    });

    steer.def("pursue", 7, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 7) return ev::null();
        return makeSteeringOutput(brogameagent::pursue(
            {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
            {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))},
            {static_cast<float>(numAt(a, 4)), static_cast<float>(numAt(a, 5))},
            static_cast<float>(numAt(a, 6))));
    });

    steer.def("evade", 7, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 7) return ev::null();
        return makeSteeringOutput(brogameagent::evade(
            {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
            {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))},
            {static_cast<float>(numAt(a, 4)), static_cast<float>(numAt(a, 5))},
            static_cast<float>(numAt(a, 6))));
    });
    game.set("steer", steer.get());

    // ── Observation / Mask / RewardTracker ───────────────────────────────────
    game.def("buildObservation", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("buildObservation(agent, world)");
        auto* ag = unwrapAgent(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!ag || !w) return ev::throwTypeError("buildObservation: invalid agent or world");

        float buf[brogameagent::observation::TOTAL];
        brogameagent::observation::build(ag->agent, w->world, buf);
        return makeFloat32Array(buf, brogameagent::observation::TOTAL);
    });

    game.def("buildActionMask", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("buildActionMask(agent, world)");
        auto* ag = unwrapAgent(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!ag || !w) return ev::throwTypeError("buildActionMask: invalid agent or world");

        float mask[brogameagent::action_mask::TOTAL];
        int enemyIds[brogameagent::action_mask::N_ENEMY_SLOTS];
        brogameagent::action_mask::build(ag->agent, w->world, mask, enemyIds);

        ObjectBuilder o;
        o.set("mask", makeFloat32Array(mask, brogameagent::action_mask::TOTAL));
        o.set("enemyIds", makeInt32Array(enemyIds, brogameagent::action_mask::N_ENEMY_SLOTS));
        return o.get();
    });

    game.def("createRewardTracker", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("createRewardTracker(agent, world)");
        auto* ag = unwrapAgent(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!ag || !w) return ev::throwTypeError("createRewardTracker: invalid agent or world");

        auto cell = std::make_unique<HostRewardTracker>();
        cell->tracker.reset(ag->agent, w->world);
        return g_rewardTrackerClass.createInstance(std::move(cell));
    });

    // ── Snapshots ────────────────────────────────────────────────────────────
    game.def("captureAgentSnapshot", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("captureAgentSnapshot(agent)");
        auto* ag = unwrapAgent(a[0]);
        if (!ag) return ev::throwTypeError("captureAgentSnapshot: invalid agent");
        auto cell = std::make_unique<HostAgentSnapshot>();
        cell->s = ag->agent.captureSnapshot();
        return g_agentSnapshotClass.createInstance(std::move(cell));
    });

    game.def("applyAgentSnapshot", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("applyAgentSnapshot(agent, snapshot)");
        auto* ag = unwrapAgent(a[0]);
        auto* s = unwrapAgentSnapshot(a[1]);
        if (!ag || !s) return ev::throwTypeError("applyAgentSnapshot: invalid agent or snapshot");
        ag->agent.applySnapshot(s->s);
        return ev::undefined();
    });

    game.def("captureWorldSnapshot", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("captureWorldSnapshot(world)");
        auto* w = unwrapWorld(a[0]);
        if (!w) return ev::throwTypeError("captureWorldSnapshot: invalid world");
        auto cell = std::make_unique<HostWorldSnapshot>();
        cell->s = w->world.snapshot();
        return g_worldSnapshotClass.createInstance(std::move(cell));
    });

    game.def("applyWorldSnapshot", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("applyWorldSnapshot(world, snapshot)");
        auto* w = unwrapWorld(a[0]);
        auto* s = unwrapWorldSnapshot(a[1]);
        if (!w || !s) return ev::throwTypeError("applyWorldSnapshot: invalid world or snapshot");
        w->world.restore(s->s);
        return ev::undefined();
    });

    // patchSnapshotWithParticles is installed by installAIBelief, which owns
    // the particle-map marshalling.

    // ── Free Projectile Helpers ──────────────────────────────────────────────
    game.def("spawnProjectile", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::fromDouble(-1);
        auto* w = unwrapWorld(a[0]);
        if (!w || !ev::isObject(a[1])) return ev::fromDouble(-1);
        auto p = parseProjectileObj(a[1]);
        return ev::fromDouble(w->world.spawnProjectile(p));
    });

    game.def("worldProjectiles", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto* w = unwrapWorld(a[0]);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        const auto& projs = w->world.projectiles();
        std::vector<const brogameagent::Projectile*> alive;
        for (const auto& p : projs) if (p.alive) alive.push_back(&p);
        return hostArrayOf(alive.size(), [&](size_t i) {
            return makeProjectileObject(*alive[i]);
        });
    });

    // ── VecSimulation ────────────────────────────────────────────────────────
    auto createVecSimFn = [](Value, std::span<const Value> a) -> Value {
        brogameagent::VecSimulation::Config cfg{};
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent root(a[0]);
            cfg.numEnvs            = getI32Property(root.get(), "numEnvs", cfg.numEnvs, 1, 65536);
            cfg.arenaHalfSize      = static_cast<float>(getDoubleProperty(root.get(), "arenaHalfSize", cfg.arenaHalfSize));
            cfg.minSpawnDist       = static_cast<float>(getDoubleProperty(root.get(), "minSpawnDist", cfg.minSpawnDist));
            cfg.maxSpawnDist       = static_cast<float>(getDoubleProperty(root.get(), "maxSpawnDist", cfg.maxSpawnDist));
            cfg.dt                 = static_cast<float>(getDoubleProperty(root.get(), "dt", cfg.dt));
            cfg.maxStepsPerEpisode = getI32Property(root.get(), "maxStepsPerEpisode", cfg.maxStepsPerEpisode, 0);
            cfg.hp                 = static_cast<float>(getDoubleProperty(root.get(), "hp", cfg.hp));
            cfg.maxMana            = static_cast<float>(getDoubleProperty(root.get(), "maxMana", cfg.maxMana));
            cfg.manaRegenPerSec    = static_cast<float>(getDoubleProperty(root.get(), "manaRegenPerSec", cfg.manaRegenPerSec));
            cfg.damage             = static_cast<float>(getDoubleProperty(root.get(), "damage", cfg.damage));
            cfg.attackRange        = static_cast<float>(getDoubleProperty(root.get(), "attackRange", cfg.attackRange));
            cfg.attacksPerSec      = static_cast<float>(getDoubleProperty(root.get(), "attacksPerSec", cfg.attacksPerSec));
            cfg.moveSpeed          = static_cast<float>(getDoubleProperty(root.get(), "moveSpeed", cfg.moveSpeed));
            cfg.maxAccel           = static_cast<float>(getDoubleProperty(root.get(), "maxAccel", cfg.maxAccel));
            cfg.maxTurnRate        = static_cast<float>(getDoubleProperty(root.get(), "maxTurnRate", cfg.maxTurnRate));
            cfg.radius             = static_cast<float>(getDoubleProperty(root.get(), "radius", cfg.radius));
            cfg.rewardDamageDealt  = static_cast<float>(getDoubleProperty(root.get(), "rewardDamageDealt", cfg.rewardDamageDealt));
            cfg.rewardDamageTakenMul = static_cast<float>(getDoubleProperty(root.get(), "rewardDamageTakenMul", cfg.rewardDamageTakenMul));
            cfg.rewardKill         = static_cast<float>(getDoubleProperty(root.get(), "rewardKill", cfg.rewardKill));
            cfg.rewardDeath        = static_cast<float>(getDoubleProperty(root.get(), "rewardDeath", cfg.rewardDeath));
            cfg.rewardStep         = static_cast<float>(getDoubleProperty(root.get(), "rewardStep", cfg.rewardStep));
            cfg.rewardTimeout      = static_cast<float>(getDoubleProperty(root.get(), "rewardTimeout", cfg.rewardTimeout));
        }
        if (cfg.numEnvs < 1 || cfg.numEnvs > 65536) {
            return ev::throwRangeError("createVecSimulation: numEnvs must be 1..65536");
        }
        auto cell = std::make_unique<HostVecSim>();
        cell->sim = std::make_unique<brogameagent::VecSimulation>(cfg);
        return g_vecSimClass.createInstance(std::move(cell));
    };

    game.def("createVecSimulation", 1, createVecSimFn);
    game.def("createVecSim", 1, createVecSimFn);

    // ── Constants ────────────────────────────────────────────────────────────
    game.set("OBS_TOTAL", ev::fromDouble(brogameagent::observation::TOTAL));
    game.set("MASK_TOTAL", ev::fromDouble(brogameagent::action_mask::TOTAL));
    game.set("N_ENEMY_SLOTS", ev::fromDouble(brogameagent::action_mask::N_ENEMY_SLOTS));

    {
        ObjectBuilder m;
        m.set("Single", ev::fromUtf8("single"));
        m.set("Pierce", ev::fromUtf8("pierce"));
        m.set("AoE",    ev::fromUtf8("aoe"));
        game.set("PROJECTILE_MODE", m.get());
    }
    {
        ObjectBuilder m;
        m.set("Physical", ev::fromUtf8("physical"));
        m.set("Magical",  ev::fromUtf8("magical"));
        m.set("True",     ev::fromUtf8("true"));
        game.set("DAMAGE_KIND", m.get());
    }
    {
        ObjectBuilder t;
        t.set("Hold",          ev::fromUtf8("Hold"));
        t.set("FocusLowestHp", ev::fromUtf8("FocusLowestHp"));
        t.set("Scatter",       ev::fromUtf8("Scatter"));
        t.set("Retreat",       ev::fromUtf8("Retreat"));
        game.set("TACTIC", t.get());
    }
    {
        ObjectBuilder m;
        m.set("Hold", ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::Hold)));
        m.set("N",    ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::N)));
        m.set("NE",   ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::NE)));
        m.set("E",    ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::E)));
        m.set("SE",   ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::SE)));
        m.set("S",    ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::S)));
        m.set("SW",   ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::SW)));
        m.set("W",    ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::W)));
        m.set("NW",   ev::fromDouble(static_cast<int>(brogameagent::mcts::MoveDir::NW)));
        game.set("MOVE_DIR", m.get());
    }

    // nn / learn / grid and the evaluator-prior-rollout factories are
    // installed by installAINeural, installAILearn, installAIGrid and
    // installAIPrimitives — this file used to publish `{available:false}`
    // stubs and factories returning `{}` in their place.
}

} // namespace brogameagent::api
