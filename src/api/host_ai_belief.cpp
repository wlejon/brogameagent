// Belief / observability / Information-Set MCTS.
//
// createTeamBelief, observe, mergeObservations, createInfoSetMcts and
// createInfoSetTeamMcts were all absent after the port, and with them every
// TeamBelief method (clear, registerEnemy, propagate, update, sample, mean,
// enemies) and the IS-MCTS setBelief / setEvaluator / setPrior wiring.

#include "host_ai_mcts_shared.h"

#include <brogameagent/belief.h>
#include <brogameagent/info_set_mcts.h>
#include <brogameagent/observability.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>

namespace brogameagent::api {

namespace belief = brogameagent::belief;
namespace obs = brogameagent::obs;

HostClass g_teamBeliefClass;
HostClass g_infoSetMctsClass;
HostClass g_infoSetTeamMctsClass;

namespace {

constexpr uint32_t kHostTeamBeliefTag      = 0x54424C46u;  // 'TBLF'
constexpr uint32_t kHostInfoSetMctsTag     = 0x49534D43u;  // 'ISMC'
constexpr uint32_t kHostInfoSetTeamMctsTag = 0x4953544Du;  // 'ISTM'

struct HostTeamBelief {
    uint32_t tag = kHostTeamBeliefTag;
    std::shared_ptr<belief::TeamBelief> b;
};

struct HostInfoSetMcts {
    uint32_t tag = kHostInfoSetMctsTag;
    std::unique_ptr<bgm::InfoSetMcts> m;
    std::shared_ptr<belief::TeamBelief> beliefRef;
    std::shared_ptr<bgm::IEvaluator> evalRef;
    std::shared_ptr<bgm::IPrior> priorRef;
};

struct HostInfoSetTeamMcts {
    uint32_t tag = kHostInfoSetTeamMctsTag;
    std::unique_ptr<bgm::InfoSetTeamMcts> m;
    std::shared_ptr<belief::TeamBelief> beliefRef;
};

HostTeamBelief* unwrapTeamBelief(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamBelief*>(ev::handleData(v));
    return (h && h->tag == kHostTeamBeliefTag) ? h : nullptr;
}

HostInfoSetMcts* unwrapInfoSetMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostInfoSetMcts*>(ev::handleData(v));
    return (h && h->tag == kHostInfoSetMctsTag) ? h : nullptr;
}

HostInfoSetTeamMcts* unwrapInfoSetTeamMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostInfoSetTeamMcts*>(ev::handleData(v));
    return (h && h->tag == kHostInfoSetTeamMctsTag) ? h : nullptr;
}

obs::VisibilityConfig parseVisibilityConfig(Value o) {
    obs::VisibilityConfig c{};
    if (!ev::isObject(o)) return c;
    ev::Persistent root(o);
    c.fov_radians = static_cast<float>(getDoubleProperty(root.get(), "fovRadians", c.fov_radians));
    c.max_range = static_cast<float>(getDoubleProperty(root.get(), "maxRange", c.max_range));
    c.check_los = getBoolProperty(root.get(), "checkLos", c.check_los);
    return c;
}

Value makeAgentObservation(const obs::AgentObservation& a) {
    ObjectBuilder o;
    o.set("id", ev::fromDouble(a.id));
    o.set("teamId", ev::fromDouble(a.team_id));
    o.set("x", ev::fromDouble(a.pos.x));
    o.set("z", ev::fromDouble(a.pos.y));
    o.set("vx", ev::fromDouble(a.vel.x));
    o.set("vz", ev::fromDouble(a.vel.y));
    o.set("hp", ev::fromDouble(a.hp));
    o.set("maxHp", ev::fromDouble(a.max_hp));
    o.set("heading", ev::fromDouble(a.heading));
    o.set("alive", ev::fromBool(a.alive));
    o.set("visible", ev::fromBool(a.visible));
    o.set("lastSeenElapsed", ev::fromDouble(a.last_seen_elapsed));
    return o.get();
}

obs::AgentObservation parseAgentObservation(Value v) {
    obs::AgentObservation a{};
    if (!ev::isObject(v)) return a;
    ev::Persistent root(v);
    a.id = static_cast<int>(getDoubleProperty(root.get(), "id", 0));
    a.team_id = static_cast<int>(getDoubleProperty(root.get(), "teamId", 0));
    a.pos.x = static_cast<float>(getDoubleProperty(root.get(), "x", 0));
    a.pos.y = static_cast<float>(getDoubleProperty(root.get(), "z", 0));
    a.vel.x = static_cast<float>(getDoubleProperty(root.get(), "vx", 0));
    a.vel.y = static_cast<float>(getDoubleProperty(root.get(), "vz", 0));
    a.hp = static_cast<float>(getDoubleProperty(root.get(), "hp", 0));
    a.max_hp = static_cast<float>(getDoubleProperty(root.get(), "maxHp", 0));
    a.heading = static_cast<float>(getDoubleProperty(root.get(), "heading", 0));
    a.alive = getBoolProperty(root.get(), "alive", false);
    a.visible = getBoolProperty(root.get(), "visible", false);
    a.last_seen_elapsed = static_cast<float>(getDoubleProperty(root.get(), "lastSeenElapsed", 0));
    return a;
}

Value makeTeamObservation(const obs::TeamObservation& t) {
    ObjectBuilder o;
    o.set("teamId", ev::fromDouble(t.team_id));
    o.set("timestamp", ev::fromDouble(t.timestamp));
    o.set("allies", hostArrayOf(t.allies.size(), [&](size_t i) {
        return makeAgentObservation(t.allies[i]);
    }));
    o.set("enemies", hostArrayOf(t.enemies.size(), [&](size_t i) {
        return makeAgentObservation(t.enemies[i]);
    }));
    return o.get();
}

obs::TeamObservation parseTeamObservation(Value v) {
    obs::TeamObservation t{};
    if (!ev::isObject(v)) return t;
    ev::Persistent root(v);
    t.team_id = static_cast<int>(getDoubleProperty(root.get(), "teamId", 0));
    t.timestamp = static_cast<float>(getDoubleProperty(root.get(), "timestamp", 0));
    auto readArr = [&](const char* key, std::vector<obs::AgentObservation>& dst) {
        ev::Persistent arr(ev::getProperty(root.get(), key));
        if (!ev::isObject(arr.get())) return;
        Value lenV = ev::getProperty(arr.get(), "length");
        uint32_t n = ev::isNumber(lenV) ? static_cast<uint32_t>(ev::toDouble(lenV)) : 0u;
        for (uint32_t i = 0; i < n; ++i) {
            dst.push_back(parseAgentObservation(ev::getElement(arr.get(), i)));
        }
    };
    readArr("allies", t.allies);
    readArr("enemies", t.enemies);
    return t;
}

Value makeEnemyParticle(const belief::EnemyParticle& p) {
    ObjectBuilder o;
    o.set("x", ev::fromDouble(p.pos.x));
    o.set("z", ev::fromDouble(p.pos.y));
    o.set("vx", ev::fromDouble(p.vel.x));
    o.set("vz", ev::fromDouble(p.vel.y));
    o.set("hp", ev::fromDouble(p.hp));
    o.set("heading", ev::fromDouble(p.heading));
    o.set("weight", ev::fromDouble(p.weight));
    return o.get();
}

Value makeParticleMap(const std::unordered_map<int, belief::EnemyParticle>& m) {
    ObjectBuilder o;
    char key[32];
    for (const auto& kv : m) {
        std::snprintf(key, sizeof(key), "%d", kv.first);
        o.set(key, makeEnemyParticle(kv.second));
    }
    return o.get();
}

} // namespace

// The particle map is keyed by stringified enemy id, so reading one back needs
// own-key enumeration; bronze has no ownKeys entry point, so this goes through
// Object.keys the way the rest of the host does.
static std::unordered_map<int, belief::EnemyParticle> parseParticleMap(Value o) {
    std::unordered_map<int, belief::EnemyParticle> m;
    if (!ev::isObject(o)) return m;
    ev::Persistent root(o);
    ev::Persistent objCtor(ev::globalValue("Object").value);
    if (!ev::isObject(objCtor.get())) return m;
    Value keysFn = ev::getProperty(objCtor.get(), "keys");
    if (!ev::isFunction(keysFn)) return m;
    Value arg = root.get();
    auto res = ev::call(keysFn, objCtor.get(), std::span<const Value>(&arg, 1));
    if (res.thrown || !ev::isObject(res.value)) return m;
    ev::Persistent keys(res.value);
    Value lenV = ev::getProperty(keys.get(), "length");
    uint32_t n = ev::isNumber(lenV) ? static_cast<uint32_t>(ev::toDouble(lenV)) : 0u;
    for (uint32_t i = 0; i < n; ++i) {
        std::string name = ev::toUtf8(ev::getElement(keys.get(), i));
        ev::Persistent val(ev::getProperty(root.get(), name));
        belief::EnemyParticle p{};
        p.pos.x = static_cast<float>(getDoubleProperty(val.get(), "x", 0));
        p.pos.y = static_cast<float>(getDoubleProperty(val.get(), "z", 0));
        p.vel.x = static_cast<float>(getDoubleProperty(val.get(), "vx", 0));
        p.vel.y = static_cast<float>(getDoubleProperty(val.get(), "vz", 0));
        p.hp = static_cast<float>(getDoubleProperty(val.get(), "hp", 0));
        p.heading = static_cast<float>(getDoubleProperty(val.get(), "heading", 0));
        p.weight = static_cast<float>(getDoubleProperty(val.get(), "weight", 1.0));
        m[std::atoi(name.c_str())] = p;
    }
    return m;
}

void ensureAIBeliefClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    // ── TeamBelief ─────────────────────────────────────────────────────────
    g_teamBeliefClass.init("AITeamBelief", [](ObjectBuilder& b) {
        b.accessor("teamId", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            return ev::fromDouble((h && h->b) ? h->b->team_id() : 0);
        }, nullptr);
        b.accessor("numParticles", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            return ev::fromDouble((h && h->b) ? h->b->num_particles() : 0);
        }, nullptr);
        b.accessor("ess", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            return ev::fromDouble((h && h->b) ? h->b->effective_sample_size() : 0.0);
        }, nullptr);

        b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (h && h->b) h->b->clear();
            return ev::undefined();
        });

        b.def("registerEnemy", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return ev::throwTypeError("TeamBelief.prototype.registerEnemy: invalid receiver");
            if (a.size() < 2) return ev::throwTypeError("TeamBelief.prototype.registerEnemy: expected (id, radius, pos?)");
            bromath::Vec2 pos{};
            const bromath::Vec2* posPtr = nullptr;
            if (a.size() >= 3 && ev::isObject(a[2])) {
                pos.x = static_cast<float>(getDoubleProperty(a[2], "x", 0));
                pos.y = static_cast<float>(getDoubleProperty(a[2], "z", 0));
                posPtr = &pos;
            }
            h->b->register_enemy(i32At(a, 0), static_cast<float>(numAt(a, 1)), posPtr);
            return ev::undefined();
        });

        b.def("propagate", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return ev::throwTypeError("TeamBelief.prototype.propagate: invalid receiver");
            if (a.size() < 3) return ev::throwTypeError("TeamBelief.prototype.propagate: expected (world, config, dt)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("propagate: expected a World");
            h->b->propagate(w->world, parseVisibilityConfig(a[1]),
                            static_cast<float>(numAt(a, 2)));
            return ev::undefined();
        });

        b.def("update", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return ev::throwTypeError("TeamBelief.prototype.update: invalid receiver");
            if (a.empty()) return ev::throwTypeError("TeamBelief.prototype.update: expected observation argument");
            h->b->update(parseTeamObservation(a[0]));
            return ev::undefined();
        });

        b.def("sample", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return ObjectBuilder{}.get();
            return makeParticleMap(h->b->sample(h->b->rng()));
        });

        b.def("mean", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return ObjectBuilder{}.get();
            return makeParticleMap(h->b->mean());
        });

        b.def("enemies", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamBelief(self);
            if (!h || !h->b) return hostArrayOf(0, [](size_t) { return ev::null(); });
            const auto& v = h->b->enemies();
            return hostArrayOf(v.size(), [&](size_t i) {
                ObjectBuilder o;
                o.set("enemyId", ev::fromDouble(v[i].enemy_id));
                o.set("maxHp", ev::fromDouble(v[i].max_hp));
                o.set("everSeen", ev::fromBool(v[i].ever_seen));
                o.set("visible", ev::fromBool(v[i].visible));
                o.set("lastSeenElapsed", ev::fromDouble(v[i].last_seen_elapsed));
                o.set("particleCount", ev::fromDouble(static_cast<double>(v[i].particles.size())));
                return o.get();
            });
        });
    });

    // ── InfoSetMcts ────────────────────────────────────────────────────────
    g_infoSetMctsClass.init("AIInfoSetMcts", [](ObjectBuilder& b) {
        b.def("setBelief", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m || a.empty()) return ev::undefined();
            auto* bd = unwrapTeamBelief(a[0]);
            if (!bd || !bd->b) return ev::throwTypeError("setBelief: expected a TeamBelief");
            h->beliefRef = bd->b;
            h->m->set_belief(bd->b);
            return ev::undefined();
        });

        b.def("setEvaluator", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m || a.empty()) return ev::undefined();
            std::shared_ptr<bgm::IEvaluator> e = extractHeroEvaluatorShared(a[0]);
            if (!e) {
                if (auto* cell = unwrapEvaluatorCell(a[0])) e = cell->p;
            }
            if (!e && ev::isString(a[0]) && ev::toUtf8(a[0]) == "hpDelta") {
                e = std::make_shared<bgm::HpDeltaEvaluator>();
            }
            if (e) {
                h->evalRef = e;
                h->m->set_evaluator(std::move(e));
            }
            return ev::undefined();
        });

        b.def("setPrior", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m || a.empty()) return ev::undefined();
            std::shared_ptr<bgm::IPrior> p = extractPriorShared(a[0]);
            if (!p) {
                if (auto* cell = unwrapPriorCell(a[0])) p = cell->p;
            }
            if (!p && ev::isString(a[0])) {
                std::string kind = ev::toUtf8(a[0]);
                if (kind == "uniform") p = std::make_shared<bgm::UniformPrior>();
                else if (kind == "attackBias") p = std::make_shared<bgm::AttackBiasPrior>();
            }
            if (p) {
                h->priorRef = p;
                h->m->set_prior(std::move(p));
            }
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m || a.empty() || !ev::isObject(a[0])) return ev::undefined();
            ev::Persistent cfg(a[0]);
            bgm::MctsConfig c = h->m->config();
            c.iterations = static_cast<int>(getDoubleProperty(cfg.get(), "iterations", c.iterations));
            c.budget_ms = static_cast<int>(getDoubleProperty(cfg.get(), "budgetMs", c.budget_ms));
            c.rollout_horizon = static_cast<int>(
                getDoubleProperty(cfg.get(), "rolloutHorizon", c.rollout_horizon));
            c.sim_dt = static_cast<float>(getDoubleProperty(cfg.get(), "simDt", c.sim_dt));
            c.action_repeat = static_cast<int>(
                getDoubleProperty(cfg.get(), "actionRepeat", c.action_repeat));
            c.uct_c = static_cast<float>(getDoubleProperty(cfg.get(), "uctC", c.uct_c));
            c.seed = getU64Property(cfg.get(), "seed", c.seed);
            c.pw_alpha = static_cast<float>(getDoubleProperty(cfg.get(), "pwAlpha", c.pw_alpha));
            c.prior_c = static_cast<float>(getDoubleProperty(cfg.get(), "priorC", c.prior_c));
            c.use_leaf_value = getBoolProperty(cfg.get(), "useLeafValue", c.use_leaf_value);
            h->m->set_config(c);
            return ev::undefined();
        });

        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m || a.size() < 2) return ev::throwTypeError("search(world, hero)");
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            if (!w || !hero) return ev::throwTypeError("search: expected world, hero");
            return makeCombatAction(h->m->search(w->world, hero->agent));
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (h && h->m && !a.empty() && ev::isObject(a[0])) {
                h->m->advance_root(parseCombatAction(a[0]));
            }
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (h && h->m) h->m->reset_tree();
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapInfoSetMcts(self);
            if (!h || !h->m) return ObjectBuilder{}.get();
            const auto& s = h->m->last_stats();
            ObjectBuilder o;
            o.set("iterations", ev::fromDouble(s.iterations));
            o.set("rootChildren", ev::fromDouble(s.root_children));
            o.set("treeSize", ev::fromDouble(s.tree_size));
            o.set("bestMean", ev::fromDouble(s.best_mean));
            o.set("bestVisits", ev::fromDouble(s.best_visits));
            o.set("elapsedMs", ev::fromDouble(s.elapsed_ms));
            o.set("reusedRoot", ev::fromBool(s.reused_root));
            o.set("meanEss", ev::fromDouble(s.mean_ess));
            return o.get();
        }, nullptr);
    });

    // ── InfoSetTeamMcts ────────────────────────────────────────────────────
    g_infoSetTeamMctsClass.init("AIInfoSetTeamMcts", [](ObjectBuilder& b) {
        b.def("setBelief", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetTeamMcts(self);
            if (!h || !h->m || a.empty()) return ev::undefined();
            auto* bd = unwrapTeamBelief(a[0]);
            if (!bd || !bd->b) return ev::throwTypeError("setBelief: expected a TeamBelief");
            h->beliefRef = bd->b;
            h->m->set_belief(bd->b);
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetTeamMcts(self);
            if (!h || !h->m || a.empty() || !ev::isObject(a[0])) return ev::undefined();
            ev::Persistent cfg(a[0]);
            bgm::MctsConfig c = h->m->config();
            c.iterations = static_cast<int>(getDoubleProperty(cfg.get(), "iterations", c.iterations));
            c.budget_ms = static_cast<int>(getDoubleProperty(cfg.get(), "budgetMs", c.budget_ms));
            c.rollout_horizon = static_cast<int>(
                getDoubleProperty(cfg.get(), "rolloutHorizon", c.rollout_horizon));
            c.sim_dt = static_cast<float>(getDoubleProperty(cfg.get(), "simDt", c.sim_dt));
            c.action_repeat = static_cast<int>(
                getDoubleProperty(cfg.get(), "actionRepeat", c.action_repeat));
            c.uct_c = static_cast<float>(getDoubleProperty(cfg.get(), "uctC", c.uct_c));
            c.seed = getU64Property(cfg.get(), "seed", c.seed);
            h->m->set_config(c);
            return ev::undefined();
        });

        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapInfoSetTeamMcts(self);
            if (!h || !h->m || a.size() < 2) return ev::throwTypeError("search(world, heroes)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("search: expected a World");
            auto out = h->m->search(w->world, parseHeroes(a[1]));
            return makeCombatActionArray(out.per_hero);
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapInfoSetTeamMcts(self);
            if (h && h->m) h->m->reset_tree();
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapInfoSetTeamMcts(self);
            if (!h || !h->m) return ObjectBuilder{}.get();
            const auto& s = h->m->last_stats();
            ObjectBuilder o;
            o.set("iterations", ev::fromDouble(s.iterations));
            o.set("treeSize", ev::fromDouble(s.tree_size));
            o.set("bestMean", ev::fromDouble(s.best_mean));
            o.set("bestVisits", ev::fromDouble(s.best_visits));
            o.set("elapsedMs", ev::fromDouble(s.elapsed_ms));
            o.set("meanEss", ev::fromDouble(s.mean_ess));
            return o.get();
        }, nullptr);
    });
}

void installAIBelief(ObjectBuilder& game) {
    ensureAIBeliefClassesInstalled();

    game.def("createTeamBelief", 1, [](Value, std::span<const Value> a) -> Value {
        int teamId = 0;
        int numParticles = 32;
        const brogameagent::NavGrid* nav = nullptr;
        belief::MotionParams mp{};
        uint64_t seed = 0xBE11EFCAFEULL;

        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            teamId = static_cast<int>(getDoubleProperty(opts.get(), "teamId", teamId));
            numParticles = static_cast<int>(
                getDoubleProperty(opts.get(), "numParticles", numParticles));
            if (auto* ng = unwrapNavGrid(ev::getProperty(opts.get(), "navGrid"))) {
                nav = ng->grid.get();
            }
            ev::Persistent mpV(ev::getProperty(opts.get(), "motion"));
            if (ev::isObject(mpV.get())) {
                mp.max_speed = static_cast<float>(getDoubleProperty(mpV.get(), "maxSpeed", mp.max_speed));
                mp.accel_std = static_cast<float>(getDoubleProperty(mpV.get(), "accelStd", mp.accel_std));
                mp.spread_on_loss = static_cast<float>(
                    getDoubleProperty(mpV.get(), "spreadOnLoss", mp.spread_on_loss));
            }
            seed = getU64Property(opts.get(), "seed", seed);
        }

        auto cell = std::make_unique<HostTeamBelief>();
        cell->b = std::make_shared<belief::TeamBelief>(teamId, numParticles, nav, mp, seed);
        return g_teamBeliefClass.createInstance(std::move(cell));
    });

    game.def("observe", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 4) return ev::throwTypeError("observe(world, teamId, visCfg, now)");
        auto* w = unwrapWorld(a[0]);
        if (!w) return ev::throwTypeError("observe: expected a World");
        auto t = obs::observe(w->world, i32At(a, 1), parseVisibilityConfig(a[2]),
                              static_cast<float>(numAt(a, 3)));
        return makeTeamObservation(t);
    });

    game.def("mergeObservations", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::throwTypeError("mergeObservations(prior, fresh, now)");
        auto prior = parseTeamObservation(a[0]);
        auto fresh = parseTeamObservation(a[1]);
        return makeTeamObservation(obs::merge(prior, fresh, static_cast<float>(numAt(a, 2))));
    });

    game.def("createInfoSetMcts", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostInfoSetMcts>();
        cell->m = std::make_unique<bgm::InfoSetMcts>();
        return g_infoSetMctsClass.createInstance(std::move(cell));
    });

    game.def("createInfoSetTeamMcts", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostInfoSetTeamMcts>();
        cell->m = std::make_unique<bgm::InfoSetTeamMcts>();
        return g_infoSetTeamMctsClass.createInstance(std::move(cell));
    });

    // Overwrite a snapshot's hidden enemies with one determinization's
    // particles — the IS-MCTS helper the port left as a no-op.
    game.def("patchSnapshotWithParticles", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) {
            return ev::throwTypeError("patchSnapshotWithParticles(snapshot, particleMap)");
        }
        auto* snap = unwrapWorldSnapshot(a[0]);
        if (!snap) return ev::throwTypeError("patchSnapshotWithParticles: expected a WorldSnapshot");
        bgm::patch_snapshot_with_particles(snap->s, parseParticleMap(a[1]));
        return ev::undefined();
    });
}

} // namespace brogameagent::api
