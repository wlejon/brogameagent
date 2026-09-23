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

// The env and the prior/value callbacks live on the GenericMcts's own JS
// object (`_env`, `_priorFn`, `_valueFn`), not in the native half: a host
// root is invisible to the collector as an edge, so the everyday shape —
// a game object that is its own env and keeps `this.mcts` — would pin both
// forever. A LiveScope roots them only while a method that can run the env
// is on the stack; the search closures read them from `live`.
struct GenericLive {
    ev::Persistent self;  // the search object stays alive while it runs
    ev::Persistent env, snapshot, restore, step, legal, observe, prior, value;
    ev::Persistent snapshots;  // self._snapshots
    bool backendOk = false;    // self._backend is still the backend the closures use
};

// The env snapshots the search tree keeps are JS values, so they too live on
// the search's JS object — `_snapshots[slot]` — and a tree node holds only a
// SnapshotRef naming the slot. A snapshot that refers back to its env (a
// clone of the game, `{ game: this, ... }`) is then one more traced edge
// rather than a root pinning the env, and the search, for as long as the
// tree exists. A released slot is cleared on the JS side at the next
// LiveScope (a node can be dropped outside one: reset(), or the finalizer).
struct SnapshotTable {
    std::vector<uint32_t> released;  // dropped by the tree, JS element not yet cleared
    std::vector<uint32_t> free;      // cleared, ready for reuse
    uint32_t next = 0;
};

struct SnapshotRef {
    std::shared_ptr<SnapshotTable> table;
    uint32_t slot = 0;
    SnapshotRef(std::shared_ptr<SnapshotTable> t, uint32_t s) : table(std::move(t)), slot(s) {}
    ~SnapshotRef() { table->released.push_back(slot); }
    SnapshotRef(const SnapshotRef&) = delete;
    SnapshotRef& operator=(const SnapshotRef&) = delete;
};

struct HostGenericMcts {
    uint32_t tag = kHostGenericMctsTag;
    std::shared_ptr<SnapshotTable> snapshots = std::make_shared<SnapshotTable>();
    std::unique_ptr<bgm::GenericMcts> mcts;
    GenericLive* live = nullptr;  // non-null only inside a LiveScope

    // opts.backend rides on the handle as `_backend`, like the env; the
    // native prior/value closures use this pointer only while a LiveScope
    // has checked `_backend` still names it.
    learn::IInferenceBackend* backend = nullptr;

    int numActions = 0;
};

Value envMethod(Value env, const char* name) {
    ev::Persistent envP(env);
    Value fn = ev::getProperty(envP.get(), name);
    return ev::isFunction(fn) ? fn : ev::undefined();
}

class LiveScope {
public:
    LiveScope(HostGenericMcts* h, Value self) : h_(h), prev_(h->live) {
        live_.self.set(self);
        const ev::Persistent& selfP = live_.self;
        live_.env.set(ev::getProperty(selfP.get(), "_env"));
        live_.snapshot.set(envMethod(live_.env.get(), "snapshot"));
        live_.restore.set(envMethod(live_.env.get(), "restore"));
        live_.step.set(envMethod(live_.env.get(), "step"));
        live_.legal.set(envMethod(live_.env.get(), "legalActions"));
        if (!ev::isFunction(live_.legal.get())) live_.legal.set(envMethod(live_.env.get(), "legal"));
        live_.observe.set(envMethod(live_.env.get(), "observe"));
        Value p = ev::getProperty(selfP.get(), "_priorFn");
        live_.prior.set(ev::isFunction(p) ? p : ev::undefined());
        Value v = ev::getProperty(selfP.get(), "_valueFn");
        live_.value.set(ev::isFunction(v) ? v : ev::undefined());
        if (h_->backend) {
            Value b = ev::getProperty(selfP.get(), "_backend");
            live_.backendOk = inferenceBackendFromJS(b) == h_->backend;
        }

        live_.snapshots.set(ev::getProperty(selfP.get(), "_snapshots"));
        if (!ev::isObject(live_.snapshots.get())) {
            // Replaced from JS: every slot the tree holds is gone with it.
            h_->mcts->reset();
            SnapshotTable& t = *h_->snapshots;
            t.released.clear();
            t.free.clear();
            t.next = 0;
            live_.snapshots.set(ev::makeArray(0));
            live_.self.set(ev::setProperty(live_.self.get(), "_snapshots", live_.snapshots.get()));
        }
        clearReleased();
        h_->live = &live_;
    }
    ~LiveScope() { h_->live = prev_; }
    LiveScope(const LiveScope&) = delete;
    LiveScope& operator=(const LiveScope&) = delete;

private:
    void clearReleased() {
        SnapshotTable& t = *h_->snapshots;
        while (!t.released.empty()) {
            const uint32_t slot = t.released.back();
            t.released.pop_back();
            live_.snapshots.set(ev::setElement(live_.snapshots.get(), slot, ev::undefined()));
            t.free.push_back(slot);
        }
    }

    HostGenericMcts* h_;
    GenericLive* prev_;
    GenericLive live_;
};

HostGenericMcts* unwrapGenericMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostGenericMcts*>(ev::handleData(v));
    return (h && h->tag == kHostGenericMctsTag) ? h : nullptr;
}

// The classic family's JS rollout policy / prior / evaluator / options live
// on the handle as `_callbacks`; `slots` are the adapters' ends of them, bound
// by a SearchScope in each method that runs the search.
struct HostClassicMcts {
    uint32_t tag = kHostClassicMctsTag;
    std::unique_ptr<bgm::Mcts> mcts;
    std::vector<JsSlotPtr> slots;
};

HostClassicMcts* unwrapClassicMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostClassicMcts*>(ev::handleData(v));
    return (h && h->tag == kHostClassicMctsTag) ? h : nullptr;
}

struct HostDecoupledMcts {
    uint32_t tag = kHostDecoupledMctsTag;
    std::unique_ptr<bgm::DecoupledMcts> mcts;
    std::vector<JsSlotPtr> slots;
};

HostDecoupledMcts* unwrapDecoupledMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostDecoupledMcts*>(ev::handleData(v));
    return (h && h->tag == kHostDecoupledMctsTag) ? h : nullptr;
}

struct HostTeamMcts {
    uint32_t tag = kHostTeamMctsTag;
    std::unique_ptr<bgm::TeamMcts> mcts;
    std::vector<JsSlotPtr> slots;
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
    std::vector<JsSlotPtr> slots;
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

// The backend's native prior/value, used only while a LiveScope has checked
// that `_backend` still names the backend `h->backend` points at.
void installBackendFns(HostGenericMcts* h, bool prior, bool value) {
    if (!h->backend) return;
    if (prior) {
        h->mcts->set_prior_fn([h, fn = makeNativePriorFn(h->backend)](
                                  const std::vector<float>& obs,
                                  const std::vector<int>& legal) -> std::vector<float> {
            if (!h->live || !h->live->backendOk) return {};
            return fn(obs, legal);
        });
    }
    if (value) {
        h->mcts->set_value_fn([h, fn = makeNativeValueFn(h->backend)](
                                  const std::vector<float>& obs) -> float {
            if (!h->live || !h->live->backendOk) return 0.0f;
            return fn(obs);
        });
    }
}

// Point the search's prior at the JS `_priorFn` (read through `live` at
// call time), or at nothing. The closure holds no JS value.
void rewireGenericPrior(HostGenericMcts* h, bool hasFn) {
    if (!h || !h->mcts) return;
    if (!hasFn) {
        h->mcts->set_prior_fn(nullptr);
        return;
    }
    h->mcts->set_prior_fn([h](const std::vector<float>& obs,
                              const std::vector<int>& legal) -> std::vector<float> {
        if (!h->live || !ev::isFunction(h->live->prior.get())) return {};
        ev::Persistent obsV(makeFloat32Array(obs.data(), obs.size()));
        ev::Persistent legV(makeInt32ArrayFromInts(legal));
        Value args[2] = { obsV.get(), legV.get() };
        auto r = ev::call(h->live->prior.get(), ev::undefined(), args);
        if (r.thrown) return {};
        return readFloatsFromValue(r.value);
    });
}

void rewireGenericValue(HostGenericMcts* h, bool hasFn) {
    if (!h || !h->mcts) return;
    if (!hasFn) {
        h->mcts->set_value_fn(nullptr);
        return;
    }
    h->mcts->set_value_fn([h](const std::vector<float>& obs) -> float {
        if (!h->live || !ev::isFunction(h->live->value.get())) return 0.0f;
        ev::Persistent obsV(makeFloat32Array(obs.data(), obs.size()));
        Value arg = obsV.get();
        auto r = ev::call(h->live->value.get(), ev::undefined(), std::span<const Value>(&arg, 1));
        if (r.thrown || !ev::isNumber(r.value)) return 0.0f;
        double d = ev::toDouble(r.value);
        return (!std::isfinite(d)) ? 0.0f : static_cast<float>(d);
    });
}

bgm::GenericMctsConfig parseGenericConfig(Value opts, bgm::GenericMctsConfig c) {
    if (!ev::isObject(opts)) return c;
    ev::Persistent root(opts);
    c.iterations = getI32Property(root.get(), "iterations", c.iterations, 0);
    c.c_puct = static_cast<float>(getDoubleProperty(root.get(), "cPuct", c.c_puct));
    c.gamma = static_cast<float>(getDoubleProperty(root.get(), "gamma", c.gamma));
    c.rollout_depth = getI32Property(root.get(), "rolloutDepth", c.rollout_depth, 0);
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
            LiveScope scope(h, self);
            if (ev::isFunction(h->live->legal.get())) {
                ev::CallResult lres = ev::call(h->live->legal.get(), h->live->env.get(), {});
                if (!lres.thrown && ev::isObject(lres.value)) {
                    Value lenV = ev::getProperty(lres.value, "length");
                    if (ev::isNumber(lenV) && ev::toDouble(lenV) == 0) return ev::fromDouble(-1);
                }
            }
            // A native backend's evaluate() throws on an observation of the
            // wrong width (env.observe() is user code); that must surface as
            // a JS Error, not unwind through compiled frames.
            try {
                return ev::fromDouble(h->mcts->search());
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        });

        b.def("rootVisits", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ev::null();
            auto visits = h->mcts->root_visits();
            return makeFloat32Array(visits.data(), visits.size());
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (h && h->mcts) {
                const int action = i32At(a, 0, "advanceRoot: action");
                LiveScope scope(h, self);
                h->mcts->advance_root(action);
            }
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
            const bool hasFn = !a.empty() && ev::isFunction(a[0]);
            ev::setProperty(self, "_priorFn", hasFn ? a[0] : ev::undefined());
            rewireGenericPrior(h, hasFn);
            return ev::undefined();
        });

        b.def("setValueFn", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h) return ev::undefined();
            const bool hasFn = !a.empty() && ev::isFunction(a[0]);
            ev::setProperty(self, "_valueFn", hasFn ? a[0] : ev::undefined());
            rewireGenericValue(h, hasFn);
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
            bgm::CombatAction act;
            {
                SearchScope scope(self, h->slots, a[0]);
                act = h->mcts->search(w->world, hero->agent);
            }
            return makeCombatAction(act);
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
            bgm::DecoupledMcts::Joint joint;
            {
                SearchScope scope(self, h->slots, a[0]);
                joint = h->mcts->search(w->world, hero->agent, opp->agent);
            }
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
            // parseHeroes allocates, so self and the world are rooted first.
            ev::Persistent selfP(self), worldP(a[0]);
            auto heroes = parseHeroes(a[1]);
            bgm::TeamMcts::JointAction joint;
            {
                SearchScope scope(selfP.get(), h->slots, worldP.get());
                joint = h->mcts->search(w->world, heroes);
            }
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
            const bgm::Option* opt = nullptr;
            {
                SearchScope scope(self, h->slots, a[0]);
                opt = h->mcts->search(w->world, hero->agent);
            }
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
            if (auto* cell = unwrapOptionCell(a[2])) target = cell->opt ? cell->opt->name() : "";
            else if (ev::isString(a[2])) target = ev::toUtf8(a[2]);
            for (const auto& sp : h->options) {
                if (sp && sp->name() == target) {
                    SearchScope scope(self, h->slots, a[0]);
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
        auto h = std::make_unique<HostGenericMcts>();
        ev::Persistent envObj;
        {
            Value env = ev::getProperty(opts.get(), "env");
            envObj.set(ev::isObject(env) ? env : opts.get());
        }

        // Checked here; read again from the env at every search (LiveScope).
        const bool envOk = ev::isFunction(envMethod(envObj.get(), "snapshot")) &&
                           ev::isFunction(envMethod(envObj.get(), "restore")) &&
                           ev::isFunction(envMethod(envObj.get(), "step")) &&
                           (ev::isFunction(envMethod(envObj.get(), "legalActions")) ||
                            ev::isFunction(envMethod(envObj.get(), "legal"))) &&
                           ev::isFunction(envMethod(envObj.get(), "observe"));

        Value numActV = ev::getProperty(envObj.get(), "numActions");
        if (!ev::isNumber(numActV)) numActV = ev::getProperty(opts.get(), "numActions");
        // Sizes the search's per-action arrays.
        h->numActions = ev::isNumber(numActV)
            ? static_cast<int>(checkedInt(ev::toDouble(numActV), 0, 1 << 24, "numActions"))
            : 0;

        if (!envOk) {
            return ev::throwTypeError(
                "createGenericMcts: env must define snapshot/restore/step/legalActions/observe");
        }
        if (h->numActions <= 0) {
            return ev::throwTypeError("createGenericMcts: numActions must be a positive integer");
        }

        bgm::GenericEnv envBridge;
        envBridge.num_actions = h->numActions;
        // Each bridge runs only inside a LiveScope (search / advanceRoot);
        // outside one there is no env to call and it answers a default.
        envBridge.snapshot_fn = [ptr = h.get()]() -> std::any {
            if (!ptr->live) return {};
            auto res = ev::call(ptr->live->snapshot.get(), ptr->live->env.get(), {});
            if (res.thrown) return {};
            // Stored in self._snapshots (SnapshotTable), not in a root.
            ev::Persistent snap(res.value);
            SnapshotTable& t = *ptr->snapshots;
            uint32_t slot;
            if (!t.free.empty()) {
                slot = t.free.back();
                t.free.pop_back();
            } else {
                slot = t.next++;
            }
            ptr->live->snapshots.set(
                ev::setElement(ptr->live->snapshots.get(), slot, snap.get()));
            return std::any(std::make_shared<SnapshotRef>(ptr->snapshots, slot));
        };
        envBridge.restore_fn = [ptr = h.get()](const std::any& s) {
            if (!ptr->live || !s.has_value()) return;
            const auto* held = std::any_cast<std::shared_ptr<SnapshotRef>>(&s);
            if (!held || !*held) return;
            Value sv = ev::getElement(ptr->live->snapshots.get(), (*held)->slot);
            ev::call(ptr->live->restore.get(), ptr->live->env.get(), std::span<const Value>(&sv, 1));
        };
        envBridge.step_fn = [ptr = h.get()](int action) -> bgm::GenericStepResult {
            if (!ptr->live) return {};
            Value av = ev::fromDouble(action);
            auto res = ev::call(ptr->live->step.get(), ptr->live->env.get(),
                                std::span<const Value>(&av, 1));
            if (res.thrown || !ev::isObject(res.value)) return {};
            ev::Persistent r(res.value);
            bgm::GenericStepResult sr;
            sr.reward = static_cast<float>(getDoubleProperty(r.get(), "reward", 0.0));
            sr.done = getBoolProperty(r.get(), "done", false);
            return sr;
        };
        envBridge.legal_actions_fn = [ptr = h.get()]() -> std::vector<int> {
            if (!ptr->live) return {};
            auto res = ev::call(ptr->live->legal.get(), ptr->live->env.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            ev::Persistent arr(res.value);
            std::vector<int> acts;
            const uint32_t n = toLength(ev::getProperty(arr.get(), "length"));
            acts.reserve(reserveHint(n));
            for (uint32_t i = 0; i < n; ++i) {
                Value el = ev::getElement(arr.get(), i);
                // The search indexes per-action arrays with these, so an
                // action outside [0, numActions) is dropped here.
                const double d = ev::isNumber(el) ? ev::toDouble(el) : -1.0;
                if (d >= 0.0 && d < static_cast<double>(ptr->numActions)) {
                    acts.push_back(static_cast<int>(d));
                }
            }
            return acts;
        };
        envBridge.observe_fn = [ptr = h.get()]() -> std::vector<float> {
            if (!ptr->live) return {};
            auto res = ev::call(ptr->live->observe.get(), ptr->live->env.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            return readFloatsFromValue(res.value);
        };

        h->mcts = std::make_unique<bgm::GenericMcts>(std::move(envBridge));
        h->mcts->set_config(parseGenericConfig(opts.get(), h->mcts->config()));

        // A DirectBackend / ServerBackend fills in whichever of prior/value
        // was not given explicitly — an explicit priorFn/valueFn always wins.
        ev::Persistent backendV(ev::getProperty(opts.get(), "backend"));
        if (ev::isObject(backendV.get())) {
            h->backend = inferenceBackendFromJS(backendV.get());
            if (!h->backend) {
                return ev::throwTypeError(
                    "createGenericMcts: opts.backend must be a DirectBackend/ServerBackend "
                    "(bro.ai.game.learn.createDirectBackend/createServerBackend)");
            }
        }

        ev::Persistent priorFn(ev::getProperty(opts.get(), "priorFn"));
        const bool hasPrior = ev::isFunction(priorFn.get());
        rewireGenericPrior(h.get(), hasPrior);

        ev::Persistent valueFn(ev::getProperty(opts.get(), "valueFn"));
        const bool hasValue = ev::isFunction(valueFn.get());
        rewireGenericValue(h.get(), hasValue);
        installBackendFns(h.get(), !hasPrior, !hasValue);

        auto* raw = h.release();
        ev::Persistent self(g_genericMctsClass.make(
            raw, [](void* p) { delete static_cast<HostGenericMcts*>(p); }));
        ev::Persistent snapshots(ev::makeArray(0));
        self.set(ev::setProperty(self.get(), "_env", envObj.get()));
        self.set(ev::setProperty(self.get(), "_priorFn", hasPrior ? priorFn.get() : ev::undefined()));
        self.set(ev::setProperty(self.get(), "_valueFn", hasValue ? valueFn.get() : ev::undefined()));
        self.set(ev::setProperty(self.get(), "_backend", raw->backend ? backendV.get() : ev::undefined()));
        self.set(ev::setProperty(self.get(), "_snapshots", snapshots.get()));
        return self.get();
    });

    game.def("createMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostClassicMcts>();
        cell->mcts = std::make_unique<bgm::Mcts>();
        JsCallbackSet cbs;
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get(), cbs)) cell->mcts->set_rollout_policy(std::move(p));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto pr = parsePrior(opts.get(), cbs)) cell->mcts->set_prior(std::move(pr));
            if (auto e = parseHeroEvaluator(opts.get(), cbs)) cell->mcts->set_evaluator(std::move(e));
        }
        cell->slots = cbs.takeSlots();
        return cbs.attach(g_mctsClass.createInstance(std::move(cell)));
    });

    game.def("createDecoupledMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostDecoupledMcts>();
        cell->mcts = std::make_unique<bgm::DecoupledMcts>();
        JsCallbackSet cbs;
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get(), cbs)) cell->mcts->set_rollout_policy(std::move(p));
            if (auto pr = parsePrior(opts.get(), cbs)) cell->mcts->set_prior(std::move(pr));
            if (auto e = parseHeroEvaluator(opts.get(), cbs)) cell->mcts->set_evaluator(std::move(e));
        }
        cell->slots = cbs.takeSlots();
        return cbs.attach(g_decoupledMctsClass.createInstance(std::move(cell)));
    });

    game.def("createTeamMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostTeamMcts>();
        cell->mcts = std::make_unique<bgm::TeamMcts>();
        JsCallbackSet cbs;
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto p = parseRolloutPolicy(opts.get(), cbs)) cell->mcts->set_rollout_policy(std::move(p));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto pr = parsePrior(opts.get(), cbs)) cell->mcts->set_prior(std::move(pr));
            if (auto tev = parseTeamEvaluator(opts.get(), cbs)) cell->mcts->set_evaluator(std::move(tev));
        }
        cell->slots = cbs.takeSlots();
        return cbs.attach(g_teamMctsClass.createInstance(std::move(cell)));
    });

    game.def("createOption", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createOption(spec)");
        ev::Persistent opts(a[0]);
        std::string name = readStringProp(opts.get(), "name");
        if (name.empty()) return ev::throwTypeError("createOption: name required");

        ev::Persistent canInit(ev::getProperty(opts.get(), "canInitiate"));
        ev::Persistent step(ev::getProperty(opts.get(), "step"));
        ev::Persistent term(ev::getProperty(opts.get(), "shouldTerminate"));
        if (!ev::isFunction(canInit.get())) return ev::throwTypeError("createOption: canInitiate must be a function");
        if (!ev::isFunction(step.get())) return ev::throwTypeError("createOption: step must be a function");
        if (!ev::isFunction(term.get())) return ev::throwTypeError("createOption: shouldTerminate must be a function");

        auto cell = std::make_unique<HostOptionCell>();
        JsCallbackSet cbs;
        cell->opt = makeJsOption(std::move(name), canInit.get(), step.get(), term.get(), cbs);
        cell->slots = cbs.takeSlots();
        return cbs.attach(g_optionClass.createInstance(std::move(cell)));
    });

    game.def("createOptionMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostOptionMcts>();
        cell->mcts = std::make_unique<bgm::OptionMcts>();
        JsCallbackSet cbs;
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto e = parseHeroEvaluator(opts.get(), cbs)) cell->mcts->set_evaluator(std::move(e));
            cell->options = parseOptionArray(opts.get(), cbs);
            if (!cell->options.empty()) {
                auto copy = cell->options;
                cell->mcts->set_options(std::move(copy));
            }
        }
        cell->slots = cbs.takeSlots();
        return cbs.attach(g_optionMctsClass.createInstance(std::move(cell)));
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
        if (ag && w) {
            WorldArgScope scope(a[2]);  // an ability it casts reaches its JS fn
            bgm::apply(ag->agent, w->world, act, dt);
        }
        return ev::undefined();
    });

    game.def("rootParallelSearch", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("rootParallelSearch(opts): opts required");
        ev::Persistent opts(a[0]);
        ev::Persistent worldsArr(ev::getProperty(opts.get(), "worlds"));
        if (!ev::isObject(worldsArr.get())) return ev::throwTypeError("rootParallelSearch: opts.worlds required");
        Value lenV = ev::getProperty(worldsArr.get(), "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) <= 0) return ev::throwTypeError("opts.worlds must be non-empty");
        const int nWorlds = static_cast<int>(std::min<uint32_t>(toLength(lenV), 1u << 16));

        // A JS callback would be run from the worker threads root-parallel
        // search spawns, and bronze's runtime is per-thread — so a function
        // here is still refused, exactly as before.
        ev::Persistent evalV(ev::getProperty(opts.get(), "evaluator"));
        if (ev::isFunction(evalV.get())) {
            return ev::throwTypeError("rootParallelSearch: opts.evaluator cannot be a JS function");
        }
        ev::Persistent rollV(ev::getProperty(opts.get(), "rolloutPolicy"));
        if (ev::isFunction(rollV.get())) {
            return ev::throwTypeError("rootParallelSearch: opts.rolloutPolicy cannot be a JS function");
        }

        std::vector<brogameagent::World*> worlds;
        worlds.reserve(nWorlds);
        for (int i = 0; i < nWorlds; i++) {
            auto* w = unwrapWorld(ev::getElement(worldsArr.get(), static_cast<uint32_t>(i)));
            if (!w) return ev::throwTypeError("invalid World in worlds array");
            worlds.push_back(&w->world);
        }

        int heroId = getI32Property(opts.get(), "heroId", -1);
        if (heroId < 0) return ev::throwTypeError("opts.heroId required");

        auto cfg = parseMctsConfig(opts.get());
        std::shared_ptr<bgm::IEvaluator> evaluator;
        if (auto* cell = unwrapEvaluatorCell(evalV.get())) evaluator = cell->p;
        if (!evaluator) evaluator = extractHeroEvaluatorShared(evalV.get());
        if (!evaluator) evaluator = std::make_shared<bgm::HpDeltaEvaluator>();

        auto rollout = rolloutFromValueOrDefault(rollV.get());

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
        ev::Persistent worldsArr(ev::getProperty(opts.get(), "worlds"));
        if (!ev::isObject(worldsArr.get())) return ev::throwTypeError("opts.worlds required");
        Value lenV = ev::getProperty(worldsArr.get(), "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) <= 0) return ev::throwTypeError("opts.worlds must be non-empty");
        const int nWorlds = static_cast<int>(std::min<uint32_t>(toLength(lenV), 1u << 16));

        ev::Persistent evalV(ev::getProperty(opts.get(), "evaluator"));
        if (ev::isFunction(evalV.get())) return ev::throwTypeError("rootParallelSearchDecoupled: opts.evaluator cannot be a JS function");
        ev::Persistent rollV(ev::getProperty(opts.get(), "rolloutPolicy"));
        if (ev::isFunction(rollV.get())) return ev::throwTypeError("rootParallelSearchDecoupled: opts.rolloutPolicy cannot be a JS function");

        std::vector<brogameagent::World*> worlds;
        worlds.reserve(nWorlds);
        for (int i = 0; i < nWorlds; i++) {
            auto* w = unwrapWorld(ev::getElement(worldsArr.get(), static_cast<uint32_t>(i)));
            if (!w) return ev::throwTypeError("invalid World in worlds array");
            worlds.push_back(&w->world);
        }

        int heroId = getI32Property(opts.get(), "heroId", -1);
        int oppId = getI32Property(opts.get(), "oppId", -1);
        if (heroId < 0 || oppId < 0) return ev::throwTypeError("opts.heroId and opts.oppId required");

        auto cfg = parseMctsConfig(opts.get());
        std::shared_ptr<bgm::IEvaluator> evaluator;
        if (auto* cell = unwrapEvaluatorCell(evalV.get())) evaluator = cell->p;
        if (!evaluator) evaluator = extractHeroEvaluatorShared(evalV.get());
        if (!evaluator) evaluator = std::make_shared<bgm::HpDeltaEvaluator>();

        auto rollout = rolloutFromValueOrDefault(rollV.get());

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
