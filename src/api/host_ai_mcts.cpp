// GenericMcts + the classic Mcts / DecoupledMcts / TeamMcts / Option /
// OptionMcts family, and the root-parallel search entry points.
//
// The marshalling helpers this file used to keep private now live in
// host_ai_mcts_support.cpp, because the planner, belief and simulation
// bindings need the same CombatAction/Tactic/config shapes.

#include "host_ai_mcts_shared.h"

#include <cmath>
#include <memory>
#include <vector>

namespace brogameagent::api {

namespace {

// ─── GenericMcts Host Wrapper ──────────────────────────────────────────────

struct HostGenericMcts {
    uint32_t tag = kHostGenericMctsTag;
    std::unique_ptr<bgm::GenericMcts> mcts;

    ev::Persistent envObj;
    ev::Persistent snapshotFn;
    ev::Persistent restoreFn;
    ev::Persistent stepFn;
    ev::Persistent legalFn;
    ev::Persistent observeFn;
    ev::Persistent priorFn;
    ev::Persistent valueFn;
    ev::Persistent backendRef;

    int numActions = 0;
};

HostGenericMcts* unwrapGenericMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostGenericMcts*>(ev::handleData(v));
    return (h && h->tag == kHostGenericMctsTag) ? h : nullptr;
}

struct HostClassicMcts {
    uint32_t tag = kHostClassicMctsTag;
    std::unique_ptr<bgm::Mcts> mcts;
};

HostClassicMcts* unwrapClassicMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostClassicMcts*>(ev::handleData(v));
    return (h && h->tag == kHostClassicMctsTag) ? h : nullptr;
}

struct HostDecoupledMcts {
    uint32_t tag = kHostDecoupledMctsTag;
    std::unique_ptr<bgm::DecoupledMcts> mcts;
};

HostDecoupledMcts* unwrapDecoupledMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostDecoupledMcts*>(ev::handleData(v));
    return (h && h->tag == kHostDecoupledMctsTag) ? h : nullptr;
}

struct HostTeamMcts {
    uint32_t tag = kHostTeamMctsTag;
    std::unique_ptr<bgm::TeamMcts> mcts;
};

HostTeamMcts* unwrapTeamMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamMcts*>(ev::handleData(v));
    return (h && h->tag == kHostTeamMctsTag) ? h : nullptr;
}

struct HostOptionMcts {
    uint32_t tag = kHostOptionMctsTag;
    std::unique_ptr<bgm::OptionMcts> mcts;
    std::vector<std::shared_ptr<bgm::Option>> options;
};

HostOptionMcts* unwrapOptionMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostOptionMcts*>(ev::handleData(v));
    return (h && h->tag == kHostOptionMctsTag) ? h : nullptr;
}

std::vector<float> readFloatsFromValue(Value v) {
    std::vector<float> out;
    readFloatVector(v, out);
    return out;
}

Value makeInt32ArrayFromInts(const std::vector<int>& v) {
    std::vector<int32_t> tmp(v.begin(), v.end());
    return makeInt32Array(tmp.data(), tmp.size());
}

void rewireGenericPrior(HostGenericMcts* h) {
    if (!h || !h->mcts) return;
    if (!ev::isFunction(h->priorFn.get())) {
        h->mcts->set_prior_fn(nullptr);
        return;
    }
    ev::Persistent fn = h->priorFn;
    h->mcts->set_prior_fn([fn](const std::vector<float>& obs,
                               const std::vector<int>& legal) -> std::vector<float> {
        ev::Persistent obsV(makeFloat32Array(obs.data(), obs.size()));
        ev::Persistent legV(makeInt32ArrayFromInts(legal));
        Value args[2] = { obsV.get(), legV.get() };
        auto r = ev::call(fn.get(), ev::undefined(), args);
        if (r.thrown) return {};
        return readFloatsFromValue(r.value);
    });
}

void rewireGenericValue(HostGenericMcts* h) {
    if (!h || !h->mcts) return;
    if (!ev::isFunction(h->valueFn.get())) {
        h->mcts->set_value_fn(nullptr);
        return;
    }
    ev::Persistent fn = h->valueFn;
    h->mcts->set_value_fn([fn](const std::vector<float>& obs) -> float {
        ev::Persistent obsV(makeFloat32Array(obs.data(), obs.size()));
        Value arg = obsV.get();
        auto r = ev::call(fn.get(), ev::undefined(), std::span<const Value>(&arg, 1));
        if (r.thrown) return 0.0f;
        double d = ev::toDouble(r.value);
        return std::isnan(d) ? 0.0f : static_cast<float>(d);
    });
}

bgm::GenericMctsConfig parseGenericConfig(Value opts, bgm::GenericMctsConfig c) {
    if (!ev::isObject(opts)) return c;
    ev::Persistent root(opts);
    c.iterations = static_cast<int>(getDoubleProperty(root.get(), "iterations", c.iterations));
    c.c_puct = static_cast<float>(getDoubleProperty(root.get(), "cPuct", c.c_puct));
    c.gamma = static_cast<float>(getDoubleProperty(root.get(), "gamma", c.gamma));
    c.rollout_depth = static_cast<int>(getDoubleProperty(root.get(), "rolloutDepth", c.rollout_depth));
    c.dirichlet_alpha = static_cast<float>(
        getDoubleProperty(root.get(), "dirichletAlpha", c.dirichlet_alpha));
    c.dirichlet_epsilon = static_cast<float>(
        getDoubleProperty(root.get(), "dirichletEpsilon", c.dirichlet_epsilon));
    c.seed = getU64Property(root.get(), "seed", c.seed);
    return c;
}

std::shared_ptr<bgm::IRolloutPolicy> rolloutFromValueOrDefault(Value v) {
    if (auto* cell = unwrapRolloutCell(v)) return cell->p;
    std::string kind = ev::isString(v) ? ev::toUtf8(v) : "aggressive";
    if (kind == "scripted") return std::make_shared<bgm::ScriptedRollout>();
    if (kind == "random") return std::make_shared<bgm::RandomRollout>();
    return std::make_shared<bgm::AggressiveRollout>();
}

Value makeParallelStats(const bgm::ParallelSearchStats& stats) {
    ObjectBuilder s;
    s.set("numThreads", ev::fromDouble(stats.num_threads));
    s.set("totalIterations", ev::fromDouble(stats.total_iterations));
    s.set("elapsedMs", ev::fromDouble(stats.elapsed_ms));
    s.set("mergedBestVisits", ev::fromDouble(stats.merged_best_visits));
    return s.get();
}

} // namespace

bgm::Mcts* classicMctsFromValue(Value v) {
    auto* h = unwrapClassicMcts(v);
    return h ? h->mcts.get() : nullptr;
}

// ---------------------------------------------------------------------------
// Class Registration & Decorations
// ---------------------------------------------------------------------------

void ensureAIMctsClassesInstalled() {
    // Per thread, like ensureAIClassesInstalled.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    ensureAIPrimitiveClassesInstalled();
    ensureAIPlannerClassesInstalled();
    ensureAISimClassesInstalled();
    ensureAIBeliefClassesInstalled();

    // GenericMcts
    g_genericMctsClass.install("AIGenericMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.accessor("numActions", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            return ev::fromDouble(h ? h->numActions : 0);
        }, nullptr);

        b.def("search", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ev::fromDouble(-1);
            if (ev::isFunction(h->legalFn.get())) {
                ev::CallResult lres = ev::call(h->legalFn.get(), h->envObj.get(), {});
                if (!lres.thrown && ev::isObject(lres.value)) {
                    Value lenV = ev::getProperty(lres.value, "length");
                    if (ev::isNumber(lenV) && ev::toDouble(lenV) == 0) return ev::fromDouble(-1);
                }
            }
            return ev::fromDouble(h->mcts->search());
        });

        b.def("rootVisits", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ev::null();
            auto visits = h->mcts->root_visits();
            return makeFloat32Array(visits.data(), visits.size());
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (h && h->mcts) h->mcts->advance_root(i32At(a, 0));
            return ev::undefined();
        });

        b.def("lastStats", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ObjectBuilder{}.get();
            auto s = h->mcts->last_stats();
            ObjectBuilder o;
            o.set("iterations", ev::fromDouble(s.iterations));
            o.set("treeSize", ev::fromDouble(s.tree_size));
            o.set("bestVisits", ev::fromDouble(s.best_visits));
            o.set("bestAction", ev::fromDouble(s.best_action));
            return o.get();
        });

        b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (h && h->mcts) h->mcts->reset();
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                h->mcts->set_config(parseGenericConfig(a[0], h->mcts->config()));
            }
            return ev::undefined();
        });

        b.def("setPriorFn", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h) return ev::undefined();
            h->priorFn.set((!a.empty() && ev::isFunction(a[0])) ? a[0] : ev::undefined());
            rewireGenericPrior(h);
            return ev::undefined();
        });

        b.def("setValueFn", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h) return ev::undefined();
            h->valueFn.set((!a.empty() && ev::isFunction(a[0])) ? a[0] : ev::undefined());
            rewireGenericValue(h);
            return ev::undefined();
        });
    });

    // Classic Mcts
    g_mctsClass.install("AIMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapClassicMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::null();
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            if (!w || !hero) return ev::null();
            return makeCombatAction(h->mcts->search(w->world, hero->agent));
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapClassicMcts(self);
            if (h && h->mcts && !a.empty()) h->mcts->advance_root(parseCombatAction(a[0]));
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapClassicMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapClassicMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                h->mcts->set_config(parseMctsConfig(a[0]));
            }
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapClassicMcts(self);
            return h && h->mcts ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // DecoupledMcts
    g_decoupledMctsClass.install("AIDecoupledMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.def("search", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapDecoupledMcts(self);
            if (!h || !h->mcts || a.size() < 3) return ev::throwTypeError("search(world, hero, opp)");
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            auto* opp = unwrapAgent(a[2]);
            if (!w || !hero || !opp) return ev::throwTypeError("search: invalid world, hero or opp");
            auto joint = h->mcts->search(w->world, hero->agent, opp->agent);
            ObjectBuilder o;
            o.set("hero", makeCombatAction(joint.hero));
            o.set("opp", makeCombatAction(joint.opp));
            return o.get();
        });

        b.def("advanceRoot", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapDecoupledMcts(self);
            if (h && h->mcts && a.size() >= 2) {
                h->mcts->advance_root(parseCombatAction(a[0]), parseCombatAction(a[1]));
            }
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapDecoupledMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapDecoupledMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                h->mcts->set_config(parseMctsConfig(a[0]));
            }
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapDecoupledMcts(self);
            return h && h->mcts ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // TeamMcts
    g_teamMctsClass.install("AITeamMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::throwTypeError("search(world, heroes)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("search: invalid world");
            auto joint = h->mcts->search(w->world, parseHeroes(a[1]));
            return makeCombatActionArray(joint.per_hero);
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                bgm::TeamMcts::JointAction j;
                j.per_hero = parseCombatActionArray(a[0]);
                h->mcts->advance_root(j);
            }
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                h->mcts->set_config(parseMctsConfig(a[0]));
            }
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamMcts(self);
            return h && h->mcts ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // Option & OptionMcts
    g_optionClass.install("AIOption", 0, nullptr, [](ObjectBuilder& b) {
        b.accessor("name", [](Value self, std::span<const Value>) -> Value {
            auto* cell = unwrapOptionCell(self);
            return (cell && cell->opt) ? ev::fromUtf8(cell->opt->name()) : ev::null();
        }, nullptr);
    });

    g_optionMctsClass.install("AIOptionMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::null();
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            if (!w || !hero) return ev::null();
            const bgm::Option* opt = h->mcts->search(w->world, hero->agent);
            return opt ? ev::fromUtf8(opt->name()) : ev::null();
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (!h || !h->mcts) return ev::undefined();
            std::string name;
            if (!a.empty()) {
                if (auto* cell = unwrapOptionCell(a[0])) {
                    if (cell->opt) name = cell->opt->name();
                } else if (ev::isString(a[0])) {
                    name = ev::toUtf8(a[0]);
                } else if (ev::isObject(a[0])) {
                    name = readStringProp(a[0], "name");
                }
            }
            if (name.empty()) {
                h->mcts->reset_tree();
                return ev::undefined();
            }
            const bgm::Option* match = nullptr;
            for (const auto& opt : h->mcts->options()) {
                if (opt && opt->name() == name) { match = opt.get(); break; }
            }
            h->mcts->advance_root(match);
            return ev::undefined();
        });

        b.def("executeOption", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (!h || !h->mcts || a.size() < 3) {
                return ev::throwTypeError("executeOption(world, hero, name)");
            }
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            if (!w || !hero) return ev::throwTypeError("executeOption: invalid world/hero");
            std::string target;
            if (ev::isString(a[2])) target = ev::toUtf8(a[2]);
            else if (auto* cell = unwrapOptionCell(a[2])) target = cell->opt ? cell->opt->name() : "";
            for (const auto& sp : h->options) {
                if (sp && sp->name() == target) {
                    return ev::fromDouble(h->mcts->execute_option(w->world, hero->agent, *sp));
                }
            }
            return ev::fromDouble(0);
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });

        b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                h->mcts->set_config(parseMctsConfig(a[0]));
            }
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapOptionMcts(self);
            return h && h->mcts ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });
}

// ---------------------------------------------------------------------------
// Install MCTS Factories & Free Functions
// ---------------------------------------------------------------------------

void installAIMcts(ObjectBuilder& game) {
    ensureAIMctsClassesInstalled();

    installAIPrimitives(game);
    installAIPlanner(game);
    installAISim(game);
    installAIBelief(game);
    installAINeural(game);
    installAILearn(game);
    installAIGrid(game);

    game.def("createGenericMcts", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) {
            return ev::throwTypeError("createGenericMcts requires an options object");
        }
        ev::Persistent opts(a[0]);

        // The env is opts.env, or opts itself when the env is passed inline.
        Value env = ev::getProperty(opts.get(), "env");
        if (!ev::isObject(env)) env = opts.get();

        auto h = std::make_unique<HostGenericMcts>();
        h->envObj = ev::Persistent(env);

        auto method = [&](const char* name) -> Value {
            Value fn = ev::getProperty(env, name);
            return ev::isFunction(fn) ? fn : ev::undefined();
        };
        h->snapshotFn = ev::Persistent(method("snapshot"));
        h->restoreFn = ev::Persistent(method("restore"));
        h->stepFn = ev::Persistent(method("step"));
        h->legalFn = ev::Persistent(method("legalActions"));
        if (!ev::isFunction(h->legalFn.get())) h->legalFn = ev::Persistent(method("legal"));
        h->observeFn = ev::Persistent(method("observe"));

        Value numActV = ev::getProperty(env, "numActions");
        if (!ev::isNumber(numActV)) numActV = ev::getProperty(opts.get(), "numActions");
        h->numActions = ev::isNumber(numActV) ? static_cast<int>(ev::toDouble(numActV)) : 0;

        if (!ev::isFunction(h->snapshotFn.get()) || !ev::isFunction(h->restoreFn.get()) ||
            !ev::isFunction(h->stepFn.get()) || !ev::isFunction(h->legalFn.get()) ||
            !ev::isFunction(h->observeFn.get())) {
            return ev::throwTypeError(
                "createGenericMcts: env must define snapshot/restore/step/legalActions/observe");
        }
        if (h->numActions <= 0) {
            return ev::throwTypeError("createGenericMcts: numActions must be a positive integer");
        }

        bgm::GenericEnv envBridge;
        envBridge.num_actions = h->numActions;
        envBridge.snapshot_fn = [ptr = h.get()]() -> std::any {
            auto res = ev::call(ptr->snapshotFn.get(), ptr->envObj.get(), {});
            if (res.thrown) return {};
            // The snapshot is a JS value, so it has to be held as a root while
            // the search keeps it — a bare Value would go stale at the next
            // collection.
            return std::any(std::make_shared<ev::Persistent>(res.value));
        };
        envBridge.restore_fn = [ptr = h.get()](const std::any& s) {
            if (!s.has_value()) return;
            const auto* held = std::any_cast<std::shared_ptr<ev::Persistent>>(&s);
            if (!held || !*held) return;
            Value sv = (*held)->get();
            ev::call(ptr->restoreFn.get(), ptr->envObj.get(), std::span<const Value>(&sv, 1));
        };
        envBridge.step_fn = [ptr = h.get()](int action) -> bgm::GenericStepResult {
            Value av = ev::fromDouble(action);
            auto res = ev::call(ptr->stepFn.get(), ptr->envObj.get(),
                                std::span<const Value>(&av, 1));
            if (res.thrown || !ev::isObject(res.value)) return {};
            ev::Persistent r(res.value);
            bgm::GenericStepResult sr;
            sr.reward = static_cast<float>(getDoubleProperty(r.get(), "reward", 0.0));
            sr.done = getBoolProperty(r.get(), "done", false);
            return sr;
        };
        envBridge.legal_actions_fn = [ptr = h.get()]() -> std::vector<int> {
            auto res = ev::call(ptr->legalFn.get(), ptr->envObj.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            ev::Persistent arr(res.value);
            std::vector<int> acts;
            Value lenV = ev::getProperty(arr.get(), "length");
            if (ev::isNumber(lenV)) {
                uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
                acts.reserve(n);
                for (uint32_t i = 0; i < n; ++i) {
                    Value el = ev::getElement(arr.get(), i);
                    if (!ev::isUndefined(el) && !ev::isObject(el)) {
                        acts.push_back(static_cast<int>(ev::toDouble(el)));
                    }
                }
            }
            return acts;
        };
        envBridge.observe_fn = [ptr = h.get()]() -> std::vector<float> {
            auto res = ev::call(ptr->observeFn.get(), ptr->envObj.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            return readFloatsFromValue(res.value);
        };

        h->mcts = std::make_unique<bgm::GenericMcts>(std::move(envBridge));
        h->mcts->set_config(parseGenericConfig(opts.get(), h->mcts->config()));

        Value pv = ev::getProperty(opts.get(), "priorFn");
        if (ev::isFunction(pv)) h->priorFn = ev::Persistent(pv);
        rewireGenericPrior(h.get());

        Value vv = ev::getProperty(opts.get(), "valueFn");
        if (ev::isFunction(vv)) h->valueFn = ev::Persistent(vv);
        rewireGenericValue(h.get());

        // A DirectBackend / ServerBackend fills in whichever of prior/value
        // was not given explicitly — an explicit priorFn/valueFn always wins.
        Value bv = ev::getProperty(opts.get(), "backend");
        if (ev::isObject(bv)) {
            auto* backend = inferenceBackendFromJS(bv);
            if (!backend) {
                return ev::throwTypeError(
                    "createGenericMcts: opts.backend must be a DirectBackend/ServerBackend "
                    "(bro.ai.game.learn.createDirectBackend/createServerBackend)");
            }
            h->backendRef = ev::Persistent(bv);
            if (!ev::isFunction(h->priorFn.get())) h->mcts->set_prior_fn(makeNativePriorFn(backend));
            if (!ev::isFunction(h->valueFn.get())) h->mcts->set_value_fn(makeNativeValueFn(backend));
        }

        auto* raw = h.release();
        return g_genericMctsClass.make(raw, [](void* p) { delete static_cast<HostGenericMcts*>(p); });
    });

    game.def("createMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostClassicMcts>();
        cell->mcts = std::make_unique<bgm::Mcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get())) cell->mcts->set_rollout_policy(std::move(p));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto pr = parsePrior(opts.get())) cell->mcts->set_prior(std::move(pr));
            if (auto e = parseHeroEvaluator(opts.get())) cell->mcts->set_evaluator(std::move(e));
        }
        return g_mctsClass.createInstance(std::move(cell));
    });

    game.def("createDecoupledMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostDecoupledMcts>();
        cell->mcts = std::make_unique<bgm::DecoupledMcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get())) cell->mcts->set_rollout_policy(std::move(p));
            if (auto pr = parsePrior(opts.get())) cell->mcts->set_prior(std::move(pr));
            if (auto e = parseHeroEvaluator(opts.get())) cell->mcts->set_evaluator(std::move(e));
        }
        return g_decoupledMctsClass.createInstance(std::move(cell));
    });

    game.def("createTeamMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostTeamMcts>();
        cell->mcts = std::make_unique<bgm::TeamMcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get())) cell->mcts->set_rollout_policy(std::move(p));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto pr = parsePrior(opts.get())) cell->mcts->set_prior(std::move(pr));
            if (auto tev = parseTeamEvaluator(opts.get())) cell->mcts->set_evaluator(std::move(tev));
        }
        return g_teamMctsClass.createInstance(std::move(cell));
    });

    game.def("createOption", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createOption(spec)");
        ev::Persistent opts(a[0]);
        std::string name = readStringProp(opts.get(), "name");
        if (name.empty()) return ev::throwTypeError("createOption: name required");

        Value canInit = ev::getProperty(opts.get(), "canInitiate");
        Value step = ev::getProperty(opts.get(), "step");
        Value term = ev::getProperty(opts.get(), "shouldTerminate");
        if (!ev::isFunction(canInit)) return ev::throwTypeError("createOption: canInitiate must be a function");
        if (!ev::isFunction(step)) return ev::throwTypeError("createOption: step must be a function");
        if (!ev::isFunction(term)) return ev::throwTypeError("createOption: shouldTerminate must be a function");

        auto cell = std::make_unique<HostOptionCell>();
        cell->opt = makeJsOption(std::move(name), canInit, step, term);
        return g_optionClass.createInstance(std::move(cell));
    });

    game.def("createOptionMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostOptionMcts>();
        cell->mcts = std::make_unique<bgm::OptionMcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto e = parseHeroEvaluator(opts.get())) cell->mcts->set_evaluator(std::move(e));
            cell->options = parseOptionArray(opts.get());
            if (!cell->options.empty()) {
                auto copy = cell->options;
                cell->mcts->set_options(std::move(copy));
            }
        }
        return g_optionMctsClass.createInstance(std::move(cell));
    });

    game.def("legalActions", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto* ag = unwrapAgent(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!ag || !w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        return makeCombatActionArray(bgm::legal_actions(ag->agent, w->world));
    });

    game.def("legalTactics", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto heroes = parseHeroes(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto ts = bgm::legal_tactics(heroes, w->world);
        return hostArrayOf(ts.size(), [&](size_t i) { return makeTactic(ts[i]); });
    });

    game.def("tacticToAction", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::null();
        auto tactic = parseTactic(a[0]);
        auto* ag = unwrapAgent(a[1]);
        auto* w = unwrapWorld(a[2]);
        if (!ag || !w) return ev::null();
        return makeCombatAction(bgm::tactic_to_action(tactic, ag->agent, w->world));
    });

    game.def("applyCombatAction", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::undefined();
        auto* ag = unwrapAgent(a[0]);
        auto act = parseCombatAction(a[1]);
        auto* w = unwrapWorld(a[2]);
        float dt = a.size() >= 4 ? static_cast<float>(numAt(a, 3)) : 0.016f;
        if (ag && w) bgm::apply(ag->agent, w->world, act, dt);
        return ev::undefined();
    });

    game.def("rootParallelSearch", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("rootParallelSearch(opts): opts required");
        ev::Persistent opts(a[0]);
        Value worldsArr = ev::getProperty(opts.get(), "worlds");
        if (!ev::isObject(worldsArr)) return ev::throwTypeError("rootParallelSearch: opts.worlds required");
        Value lenV = ev::getProperty(worldsArr, "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) <= 0) return ev::throwTypeError("opts.worlds must be non-empty");

        // A JS callback would be run from the worker threads root-parallel
        // search spawns, and bronze's runtime is per-thread — so a function
        // here is still refused, exactly as before.
        Value evalV = ev::getProperty(opts.get(), "evaluator");
        if (ev::isFunction(evalV)) {
            return ev::throwTypeError("rootParallelSearch: opts.evaluator cannot be a JS function");
        }
        Value rollV = ev::getProperty(opts.get(), "rolloutPolicy");
        if (ev::isFunction(rollV)) {
            return ev::throwTypeError("rootParallelSearch: opts.rolloutPolicy cannot be a JS function");
        }

        int nWorlds = static_cast<int>(ev::toDouble(lenV));
        std::vector<brogameagent::World*> worlds;
        worlds.reserve(nWorlds);
        for (int i = 0; i < nWorlds; i++) {
            auto* w = unwrapWorld(ev::getElement(worldsArr, static_cast<uint32_t>(i)));
            if (!w) return ev::throwTypeError("invalid World in worlds array");
            worlds.push_back(&w->world);
        }

        int heroId = static_cast<int>(getDoubleProperty(opts.get(), "heroId", -1));
        if (heroId < 0) return ev::throwTypeError("opts.heroId required");

        auto cfg = parseMctsConfig(opts.get());
        std::shared_ptr<bgm::IEvaluator> evaluator;
        if (auto* cell = unwrapEvaluatorCell(evalV)) evaluator = cell->p;
        if (!evaluator) evaluator = extractHeroEvaluatorShared(evalV);
        if (!evaluator) evaluator = std::make_shared<bgm::HpDeltaEvaluator>();

        auto rollout = rolloutFromValueOrDefault(rollV);

        bgm::OpponentPolicy oppPolicy = bgm::policy_aggressive;
        if (auto op = parseOpponentPolicy(opts.get())) oppPolicy = std::move(op);

        bgm::ParallelSearchStats stats{};
        auto action = bgm::root_parallel_search(
            worlds, heroId, cfg, evaluator, rollout, oppPolicy, &stats);

        ObjectBuilder res;
        res.set("action", makeCombatAction(action));
        res.set("stats", makeParallelStats(stats));
        return res.get();
    });

    game.def("rootParallelSearchDecoupled", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("rootParallelSearchDecoupled(opts): opts required");
        ev::Persistent opts(a[0]);
        Value worldsArr = ev::getProperty(opts.get(), "worlds");
        if (!ev::isObject(worldsArr)) return ev::throwTypeError("opts.worlds required");
        Value lenV = ev::getProperty(worldsArr, "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) <= 0) return ev::throwTypeError("opts.worlds must be non-empty");

        Value evalV = ev::getProperty(opts.get(), "evaluator");
        if (ev::isFunction(evalV)) return ev::throwTypeError("rootParallelSearchDecoupled: opts.evaluator cannot be a JS function");
        Value rollV = ev::getProperty(opts.get(), "rolloutPolicy");
        if (ev::isFunction(rollV)) return ev::throwTypeError("rootParallelSearchDecoupled: opts.rolloutPolicy cannot be a JS function");

        int nWorlds = static_cast<int>(ev::toDouble(lenV));
        std::vector<brogameagent::World*> worlds;
        worlds.reserve(nWorlds);
        for (int i = 0; i < nWorlds; i++) {
            auto* w = unwrapWorld(ev::getElement(worldsArr, static_cast<uint32_t>(i)));
            if (!w) return ev::throwTypeError("invalid World in worlds array");
            worlds.push_back(&w->world);
        }

        int heroId = static_cast<int>(getDoubleProperty(opts.get(), "heroId", -1));
        int oppId = static_cast<int>(getDoubleProperty(opts.get(), "oppId", -1));
        if (heroId < 0 || oppId < 0) return ev::throwTypeError("opts.heroId and opts.oppId required");

        auto cfg = parseMctsConfig(opts.get());
        std::shared_ptr<bgm::IEvaluator> evaluator;
        if (auto* cell = unwrapEvaluatorCell(evalV)) evaluator = cell->p;
        if (!evaluator) evaluator = extractHeroEvaluatorShared(evalV);
        if (!evaluator) evaluator = std::make_shared<bgm::HpDeltaEvaluator>();

        auto rollout = rolloutFromValueOrDefault(rollV);

        bgm::ParallelSearchStats stats{};
        auto joint = bgm::root_parallel_search_decoupled(
            worlds, heroId, oppId, cfg, evaluator, rollout, &stats);

        ObjectBuilder res;
        res.set("hero", makeCombatAction(joint.hero));
        res.set("opp", makeCombatAction(joint.opp));
        res.set("stats", makeParallelStats(stats));
        return res.get();
    });
}

} // namespace brogameagent::api
