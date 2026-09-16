#include "host_ai_internal.h"
#include <cmath>
#include <memory>
#include <vector>

namespace brogameagent::api {

namespace {

Value makeCombatAction(const brogameagent::mcts::CombatAction& a) {
    ObjectBuilder o;
    o.set("moveDir", ev::fromDouble(static_cast<int>(a.move_dir)));
    o.set("attackSlot", ev::fromDouble(a.attack_slot));
    o.set("abilitySlot", ev::fromDouble(a.ability_slot));
    return o.get();
}

brogameagent::mcts::CombatAction parseCombatAction(Value v) {
    brogameagent::mcts::CombatAction a{};
    if (ev::isObject(v)) {
        a.move_dir = static_cast<brogameagent::mcts::MoveDir>(static_cast<int>(getDoubleProperty(v, "moveDir", 0)));
        a.attack_slot = static_cast<int8_t>(getDoubleProperty(v, "attackSlot", -1));
        a.ability_slot = static_cast<int8_t>(getDoubleProperty(v, "abilitySlot", -1));
    }
    return a;
}

Value makeTactic(const brogameagent::mcts::Tactic& t) {
    ObjectBuilder o;
    const char* kindStr = "Hold";
    switch (t.kind) {
        case brogameagent::mcts::TacticKind::FocusLowestHp: kindStr = "FocusLowestHp"; break;
        case brogameagent::mcts::TacticKind::Scatter:       kindStr = "Scatter"; break;
        case brogameagent::mcts::TacticKind::Retreat:       kindStr = "Retreat"; break;
        default:                                             kindStr = "Hold"; break;
    }
    o.set("kind", ev::fromUtf8(kindStr));
    return o.get();
}

brogameagent::mcts::Tactic parseTactic(Value v) {
    brogameagent::mcts::Tactic t{};
    if (ev::isObject(v)) {
        std::string kindStr = "Hold";
        Value kv = ev::getProperty(v, "kind");
        if (ev::isString(kv)) kindStr = ev::toUtf8(kv);
        if (kindStr == "FocusLowestHp") t.kind = brogameagent::mcts::TacticKind::FocusLowestHp;
        else if (kindStr == "Scatter")  t.kind = brogameagent::mcts::TacticKind::Scatter;
        else if (kindStr == "Retreat")  t.kind = brogameagent::mcts::TacticKind::Retreat;
        else                            t.kind = brogameagent::mcts::TacticKind::Hold;
    }
    return t;
}

brogameagent::mcts::MctsConfig parseMctsConfig(Value opts) {
    brogameagent::mcts::MctsConfig cfg{};
    if (ev::isObject(opts)) {
        cfg.iterations = static_cast<int>(getDoubleProperty(opts, "iterations", cfg.iterations));
        cfg.rollout_horizon = static_cast<int>(getDoubleProperty(opts, "rolloutHorizon", cfg.rollout_horizon));
        cfg.sim_dt = static_cast<float>(getDoubleProperty(opts, "simDt", cfg.sim_dt));
        cfg.seed = getU64Property(opts, "seed", cfg.seed);
        cfg.option_max_windows = static_cast<int>(getDoubleProperty(opts, "optionMaxWindows", cfg.option_max_windows));
        cfg.use_leaf_value = getBoolProperty(opts, "useLeafValue", cfg.use_leaf_value);
    }
    return cfg;
}

Value makeSearchStats(const brogameagent::mcts::SearchStats& s) {
    ObjectBuilder o;
    o.set("iterations", ev::fromDouble(s.iterations));
    o.set("rootChildren", ev::fromDouble(s.root_children));
    o.set("treeSize", ev::fromDouble(s.tree_size));
    o.set("bestMean", ev::fromDouble(s.best_mean));
    o.set("bestVisits", ev::fromDouble(s.best_visits));
    o.set("elapsedMs", ev::fromDouble(s.elapsed_ms));
    return o.get();
}

std::vector<brogameagent::Agent*> parseHeroes(Value arr) {
    std::vector<brogameagent::Agent*> heroes;
    if (ev::isObject(arr)) {
        Value lenV = ev::getProperty(arr, "length");
        if (ev::isNumber(lenV)) {
            int n = static_cast<int>(ev::toDouble(lenV));
            for (int i = 0; i < n; i++) {
                Value v = ev::getElement(arr, static_cast<uint32_t>(i));
                if (auto* ag = unwrapAgent(v)) heroes.push_back(&ag->agent);
            }
        }
    }
    return heroes;
}

inline brogameagent::mcts::OpponentPolicy parseOpponentPolicy(const std::string& name) {
    if (name == "aggressive") return brogameagent::mcts::policy_aggressive;
    if (name == "scripted") return brogameagent::mcts::policy_scripted;
    return brogameagent::mcts::policy_idle;
}

// ─── GenericMcts Host Wrapper ──────────────────────────────────────────────

struct HostGenericMcts {
    uint32_t tag = kHostGenericMctsTag;
    std::unique_ptr<brogameagent::mcts::GenericMcts> mcts;

    ev::Persistent envObj;
    ev::Persistent snapshotFn;
    ev::Persistent restoreFn;
    ev::Persistent stepFn;
    ev::Persistent legalFn;
    ev::Persistent observeFn;

    int numActions = 0;
};

HostGenericMcts* unwrapGenericMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostGenericMcts*>(ev::handleData(v));
    return (h && h->tag == kHostGenericMctsTag) ? h : nullptr;
}

struct HostClassicMcts {
    uint32_t tag = kHostClassicMctsTag;
    std::unique_ptr<brogameagent::mcts::Mcts> mcts;
};

HostClassicMcts* unwrapClassicMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostClassicMcts*>(ev::handleData(v));
    return (h && h->tag == kHostClassicMctsTag) ? h : nullptr;
}

struct HostDecoupledMcts {
    uint32_t tag = kHostDecoupledMctsTag;
    std::unique_ptr<brogameagent::mcts::DecoupledMcts> mcts;
};

HostDecoupledMcts* unwrapDecoupledMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostDecoupledMcts*>(ev::handleData(v));
    return (h && h->tag == kHostDecoupledMctsTag) ? h : nullptr;
}

struct HostTeamMcts {
    uint32_t tag = kHostTeamMctsTag;
    std::unique_ptr<brogameagent::mcts::TeamMcts> mcts;
};

HostTeamMcts* unwrapTeamMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamMcts*>(ev::handleData(v));
    return (h && h->tag == kHostTeamMctsTag) ? h : nullptr;
}

class BronzeOption : public brogameagent::mcts::Option {
public:
    BronzeOption(std::string name, Value canInit, Value step, Value term)
        : name_(std::move(name)), canInit_(canInit), step_(step), term_(term) {}

    const std::string& name() const override { return name_; }

    bool can_initiate(const brogameagent::Agent&, const brogameagent::World&) const override {
        if (ev::isUndefined(canInit_.get()) || ev::isNull(canInit_.get())) return true;
        auto r = ev::call(canInit_.get(), ev::undefined(), {});
        return !r.thrown && ev::toBool(r.value);
    }

    brogameagent::mcts::CombatAction step(brogameagent::Agent&, brogameagent::World&, int ticks) const override {
        if (ev::isUndefined(step_.get()) || ev::isNull(step_.get())) return {};
        Value tv = ev::fromDouble(ticks);
        auto r = ev::call(step_.get(), ev::undefined(), std::span<const Value>(&tv, 1));
        if (r.thrown || !ev::isObject(r.value)) return {};
        return parseCombatAction(r.value);
    }

    bool should_terminate(const brogameagent::Agent&, const brogameagent::World&, int ticks) const override {
        if (ev::isUndefined(term_.get()) || ev::isNull(term_.get())) return true;
        Value args[3] = { ev::undefined(), ev::undefined(), ev::fromDouble(ticks) };
        auto r = ev::call(term_.get(), ev::undefined(), args);
        return !r.thrown && ev::toBool(r.value);
    }

private:
    std::string name_;
    ev::Persistent canInit_;
    ev::Persistent step_;
    ev::Persistent term_;
};

struct HostOptionCell {
    uint32_t tag = kHostOptionTag;
    std::shared_ptr<BronzeOption> opt;
};

HostOptionCell* unwrapOption(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostOptionCell*>(ev::handleData(v));
    return (h && h->tag == kHostOptionTag) ? h : nullptr;
}

struct HostOptionMcts {
    uint32_t tag = kHostOptionMctsTag;
    std::unique_ptr<brogameagent::mcts::OptionMcts> mcts;
    std::vector<std::shared_ptr<brogameagent::mcts::Option>> options;
};

HostOptionMcts* unwrapOptionMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostOptionMcts*>(ev::handleData(v));
    return (h && h->tag == kHostOptionMctsTag) ? h : nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// Class Registration & Decorations
// ---------------------------------------------------------------------------

void ensureAIMctsClassesInstalled() {
    static bool installed = false;
    if (installed) return;
    installed = true;

    // GenericMcts
    g_genericMctsClass.install("AIGenericMcts", 0, nullptr, [](ObjectBuilder& b) {
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
            int act = h->mcts->search();
            return ev::fromDouble(act);
        });

        b.def("rootVisits", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ev::null();
            auto visits = h->mcts->root_visits();
            return makeFloat32Array(visits.data(), visits.size());
        });

        b.def("lastStats", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapGenericMcts(self);
            if (!h || !h->mcts) return ev::null();
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
                ev::Persistent root(a[0]);
                auto cfg = h->mcts->config();
                cfg.iterations = static_cast<int>(getDoubleProperty(root.get(), "iterations", cfg.iterations));
                cfg.c_puct = static_cast<float>(getDoubleProperty(root.get(), "cPuct", cfg.c_puct));
                cfg.gamma = static_cast<float>(getDoubleProperty(root.get(), "gamma", cfg.gamma));
                cfg.rollout_depth = static_cast<int>(getDoubleProperty(root.get(), "rolloutDepth", cfg.rollout_depth));
                h->mcts->set_config(cfg);
            }
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
            auto act = h->mcts->search(w->world, hero->agent);
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
            auto heroes = parseHeroes(a[1]);
            auto joint = h->mcts->search(w->world, heroes);
            return hostArrayOf(joint.per_hero.size(), [&](size_t i) {
                return makeCombatAction(joint.per_hero[i]);
            });
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (h && h->mcts && !a.empty() && ev::isObject(a[0])) {
                brogameagent::mcts::TeamMcts::JointAction j;
                Value lenV = ev::getProperty(a[0], "length");
                if (ev::isNumber(lenV)) {
                    int n = static_cast<int>(ev::toDouble(lenV));
                    for (int i = 0; i < n; i++) {
                        j.per_hero.push_back(parseCombatAction(ev::getElement(a[0], static_cast<uint32_t>(i))));
                    }
                }
                h->mcts->advance_root(j);
            }
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });

        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamMcts(self);
            return h && h->mcts ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // Option & OptionMcts
    g_optionClass.install("AIOption", 0, nullptr, [](ObjectBuilder&) {});
    g_optionMctsClass.install("AIOptionMcts", 0, nullptr, [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::null();
            auto* w = unwrapWorld(a[0]);
            auto* hero = unwrapAgent(a[1]);
            if (!w || !hero) return ev::null();
            const brogameagent::mcts::Option* opt = h->mcts->search(w->world, hero->agent);
            if (!opt) return ev::null();
            return ev::fromUtf8(opt->name());
        });

        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (!h || !h->mcts || a.empty()) return ev::undefined();
            std::string name;
            if (auto* opt = unwrapOption(a[0])) {
                if (opt->opt) name = opt->opt->name();
            } else if (ev::isString(a[0])) {
                name = ev::toUtf8(a[0]);
            } else if (ev::isObject(a[0])) {
                Value nameV = ev::getProperty(a[0], "name");
                if (ev::isString(nameV)) name = ev::toUtf8(nameV);
            }
            const brogameagent::mcts::Option* match = nullptr;
            for (const auto& opt : h->mcts->options()) {
                if (opt && opt->name() == name) {
                    match = opt.get();
                    break;
                }
            }
            h->mcts->advance_root(match);
            return ev::undefined();
        });

        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapOptionMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
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

    game.def("createGenericMcts", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createGenericMcts requires an options object");
        ev::Persistent opts(a[0]);

        auto h = std::make_unique<HostGenericMcts>();
        Value env = ev::getProperty(opts.get(), "env");
        if (!ev::isObject(env)) return ev::throwTypeError("createGenericMcts: opts.env required");
        h->envObj = ev::Persistent(env);

        Value numActV = ev::getProperty(opts.get(), "numActions");
        if (!ev::isNumber(numActV)) numActV = ev::getProperty(env, "numActions");
        h->numActions = ev::isNumber(numActV) ? static_cast<int>(ev::toDouble(numActV)) : 4;

        h->snapshotFn = ev::Persistent(ev::getProperty(env, "snapshot"));
        h->restoreFn  = ev::Persistent(ev::getProperty(env, "restore"));
        h->stepFn     = ev::Persistent(ev::getProperty(env, "step"));
        h->legalFn    = ev::Persistent(ev::getProperty(env, "legalActions"));
        if (!ev::isFunction(h->legalFn.get())) h->legalFn = ev::Persistent(ev::getProperty(env, "legal"));
        h->observeFn  = ev::Persistent(ev::getProperty(env, "observe"));

        brogameagent::mcts::GenericEnv envBridge;
        envBridge.num_actions = h->numActions;
        envBridge.snapshot_fn = [ptr = h.get()]() -> std::any {
            if (!ev::isFunction(ptr->snapshotFn.get())) return {};
            auto res = ev::call(ptr->snapshotFn.get(), ptr->envObj.get(), {});
            return res.thrown ? std::any{} : std::any(res.value);
        };
        envBridge.restore_fn = [ptr = h.get()](const std::any& s) {
            if (!ev::isFunction(ptr->restoreFn.get()) || !s.has_value()) return;
            Value sv = std::any_cast<Value>(s);
            ev::call(ptr->restoreFn.get(), ptr->envObj.get(), std::span<const Value>(&sv, 1));
        };
        envBridge.step_fn = [ptr = h.get()](int action) -> brogameagent::mcts::GenericStepResult {
            if (!ev::isFunction(ptr->stepFn.get())) return {};
            Value av = ev::fromDouble(action);
            auto res = ev::call(ptr->stepFn.get(), ptr->envObj.get(), std::span<const Value>(&av, 1));
            if (res.thrown || !ev::isObject(res.value)) return {};
            brogameagent::mcts::GenericStepResult sr;
            sr.reward = static_cast<float>(getDoubleProperty(res.value, "reward", 0.0));
            sr.done = getBoolProperty(res.value, "done", false);
            return sr;
        };
        envBridge.legal_actions_fn = [ptr = h.get()]() -> std::vector<int> {
            if (!ev::isFunction(ptr->legalFn.get())) {
                std::vector<int> all(ptr->numActions);
                for (int i = 0; i < ptr->numActions; i++) all[i] = i;
                return all;
            }
            auto res = ev::call(ptr->legalFn.get(), ptr->envObj.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            std::vector<int> acts;
            Value lenV = ev::getProperty(res.value, "length");
            if (ev::isNumber(lenV)) {
                int n = static_cast<int>(ev::toDouble(lenV));
                for (int i = 0; i < n; i++) {
                    Value el = ev::getElement(res.value, static_cast<uint32_t>(i));
                    if (ev::isNumber(el)) acts.push_back(static_cast<int>(ev::toDouble(el)));
                }
            }
            return acts;
        };
        envBridge.observe_fn = [ptr = h.get()]() -> std::vector<float> {
            if (!ev::isFunction(ptr->observeFn.get())) return {};
            auto res = ev::call(ptr->observeFn.get(), ptr->envObj.get(), {});
            if (res.thrown || !ev::isObject(res.value)) return {};
            std::vector<float> obs;
            readFloatVector(res.value, obs);
            return obs;
        };

        brogameagent::mcts::GenericMctsConfig cfg;
        cfg.iterations = static_cast<int>(getDoubleProperty(opts.get(), "iterations", cfg.iterations));
        cfg.c_puct = static_cast<float>(getDoubleProperty(opts.get(), "cPuct", cfg.c_puct));
        cfg.gamma = static_cast<float>(getDoubleProperty(opts.get(), "gamma", cfg.gamma));
        cfg.rollout_depth = static_cast<int>(getDoubleProperty(opts.get(), "rolloutDepth", cfg.rollout_depth));

        h->mcts = std::make_unique<brogameagent::mcts::GenericMcts>(std::move(envBridge));
        h->mcts->set_config(cfg);
        auto* raw = h.release();
        return g_genericMctsClass.make(raw, [](void* p) { delete static_cast<HostGenericMcts*>(p); });
    });

    game.def("createMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cfg = a.empty() ? brogameagent::mcts::MctsConfig{} : parseMctsConfig(a[0]);
        auto* h = new HostClassicMcts();
        h->mcts = std::make_unique<brogameagent::mcts::Mcts>(cfg);
        return g_mctsClass.make(h, [](void* p) { delete static_cast<HostClassicMcts*>(p); });
    });

    game.def("createDecoupledMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cfg = a.empty() ? brogameagent::mcts::MctsConfig{} : parseMctsConfig(a[0]);
        auto* h = new HostDecoupledMcts();
        h->mcts = std::make_unique<brogameagent::mcts::DecoupledMcts>(cfg);
        return g_decoupledMctsClass.make(h, [](void* p) { delete static_cast<HostDecoupledMcts*>(p); });
    });

    game.def("createTeamMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cfg = a.empty() ? brogameagent::mcts::MctsConfig{} : parseMctsConfig(a[0]);
        auto* h = new HostTeamMcts();
        h->mcts = std::make_unique<brogameagent::mcts::TeamMcts>(cfg);
        return g_teamMctsClass.make(h, [](void* p) { delete static_cast<HostTeamMcts*>(p); });
    });

    game.def("createOption", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createOption requires options object");
        ev::Persistent opts(a[0]);
        Value nameV = ev::getProperty(opts.get(), "name");
        std::string name = ev::isString(nameV) ? ev::toUtf8(nameV) : "option";
        Value canInit = ev::getProperty(opts.get(), "canInitiate");
        Value step = ev::getProperty(opts.get(), "step");
        Value term = ev::getProperty(opts.get(), "shouldTerminate");

        auto opt = std::make_shared<BronzeOption>(std::move(name), canInit, step, term);
        auto* cell = new HostOptionCell();
        cell->opt = std::move(opt);
        return g_optionClass.make(cell, [](void* p) { delete static_cast<HostOptionCell*>(p); });
    });

    game.def("createOptionMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cfg = a.empty() ? brogameagent::mcts::MctsConfig{} : parseMctsConfig(a[0]);
        auto* cell = new HostOptionMcts();
        cell->mcts = std::make_unique<brogameagent::mcts::OptionMcts>(cfg);

        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            Value optsArr = ev::getProperty(opts.get(), "options");
            if (ev::isObject(optsArr)) {
                Value lenV = ev::getProperty(optsArr, "length");
                if (ev::isNumber(lenV)) {
                    int n = static_cast<int>(ev::toDouble(lenV));
                    for (int i = 0; i < n; ++i) {
                        Value el = ev::getElement(optsArr, static_cast<uint32_t>(i));
                        if (auto* ho = unwrapOption(el)) {
                            cell->options.push_back(ho->opt);
                        }
                    }
                }
            }
            cell->mcts->set_options(cell->options);

            Value oppPol = ev::getProperty(opts.get(), "opponentPolicy");
            if (ev::isString(oppPol)) {
                cell->mcts->set_opponent_policy(parseOpponentPolicy(ev::toUtf8(oppPol)));
            }

            Value evalV = ev::getProperty(opts.get(), "evaluator");
            if (ev::isString(evalV)) {
                std::string evStr = ev::toUtf8(evalV);
                if (evStr == "hpDelta") {
                    cell->mcts->set_evaluator(std::make_shared<brogameagent::mcts::HpDeltaEvaluator>());
                }
            }
        }

        return g_optionMctsClass.make(cell, [](void* p) { delete static_cast<HostOptionMcts*>(p); });
    });

    game.def("legalActions", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto* ag = unwrapAgent(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!ag || !w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto acts = brogameagent::mcts::legal_actions(ag->agent, w->world);
        return hostArrayOf(acts.size(), [&](size_t i) { return makeCombatAction(acts[i]); });
    });

    game.def("legalTactics", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto heroes = parseHeroes(a[0]);
        auto* w = unwrapWorld(a[1]);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        auto ts = brogameagent::mcts::legal_tactics(heroes, w->world);
        return hostArrayOf(ts.size(), [&](size_t i) { return makeTactic(ts[i]); });
    });

    game.def("tacticToAction", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::null();
        auto tactic = parseTactic(a[0]);
        auto* ag = unwrapAgent(a[1]);
        auto* w = unwrapWorld(a[2]);
        if (!ag || !w) return ev::null();
        auto act = brogameagent::mcts::tactic_to_action(tactic, ag->agent, w->world);
        return makeCombatAction(act);
    });

    game.def("applyCombatAction", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::undefined();
        auto* ag = unwrapAgent(a[0]);
        auto act = parseCombatAction(a[1]);
        auto* w = unwrapWorld(a[2]);
        float dt = a.size() >= 4 ? static_cast<float>(numAt(a, 3)) : 0.016f;
        if (ag && w) brogameagent::mcts::apply(ag->agent, w->world, act, dt);
        return ev::undefined();
    });

    game.def("rootParallelSearch", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("rootParallelSearch(opts): opts required");
        ev::Persistent opts(a[0]);
        Value worldsArr = ev::getProperty(opts.get(), "worlds");
        if (!ev::isObject(worldsArr)) return ev::throwTypeError("rootParallelSearch: opts.worlds required");
        Value lenV = ev::getProperty(worldsArr, "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) <= 0) return ev::throwTypeError("opts.worlds must be non-empty");

        int nWorlds = static_cast<int>(ev::toDouble(lenV));
        Value evalV = ev::getProperty(opts.get(), "evaluator");
        if (ev::isFunction(evalV)) {
            return ev::throwTypeError("rootParallelSearch: opts.evaluator cannot be a JS function");
        }
        Value rollV = ev::getProperty(opts.get(), "rolloutPolicy");
        if (ev::isFunction(rollV)) {
            return ev::throwTypeError("rootParallelSearch: opts.rolloutPolicy cannot be a JS function");
        }

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
        auto evaluator = std::make_shared<brogameagent::mcts::HpDeltaEvaluator>();

        std::string rollStr = "aggressive";
        if (ev::isString(rollV)) rollStr = ev::toUtf8(rollV);
        std::shared_ptr<brogameagent::mcts::IRolloutPolicy> rollout;
        if (rollStr == "scripted") rollout = std::make_shared<brogameagent::mcts::ScriptedRollout>();
        else if (rollStr == "random") rollout = std::make_shared<brogameagent::mcts::RandomRollout>();
        else rollout = std::make_shared<brogameagent::mcts::AggressiveRollout>();

        std::string oppStr = "aggressive";
        Value oppV = ev::getProperty(opts.get(), "opponentPolicy");
        if (ev::isString(oppV)) oppStr = ev::toUtf8(oppV);
        brogameagent::mcts::OpponentPolicy oppPolicy = brogameagent::mcts::policy_aggressive;
        if (oppStr == "idle") oppPolicy = brogameagent::mcts::policy_idle;
        else if (oppStr == "scripted") oppPolicy = brogameagent::mcts::policy_scripted;

        brogameagent::mcts::ParallelSearchStats stats{};
        auto action = brogameagent::mcts::root_parallel_search(
            worlds, heroId, cfg, evaluator, rollout, oppPolicy, &stats);

        ObjectBuilder res;
        res.set("action", makeCombatAction(action));
        ObjectBuilder s;
        s.set("numThreads", ev::fromDouble(stats.num_threads));
        s.set("totalIterations", ev::fromDouble(stats.total_iterations));
        s.set("elapsedMs", ev::fromDouble(stats.elapsed_ms));
        s.set("mergedBestVisits", ev::fromDouble(stats.merged_best_visits));
        res.set("stats", s.get());
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
        int oppId  = static_cast<int>(getDoubleProperty(opts.get(), "oppId", -1));
        if (heroId < 0 || oppId < 0) return ev::throwTypeError("opts.heroId and opts.oppId required");

        auto cfg = parseMctsConfig(opts.get());
        auto evaluator = std::make_shared<brogameagent::mcts::HpDeltaEvaluator>();

        std::string rollStr = "aggressive";
        if (ev::isString(rollV)) rollStr = ev::toUtf8(rollV);
        std::shared_ptr<brogameagent::mcts::IRolloutPolicy> rollout;
        if (rollStr == "scripted") rollout = std::make_shared<brogameagent::mcts::ScriptedRollout>();
        else if (rollStr == "random") rollout = std::make_shared<brogameagent::mcts::RandomRollout>();
        else rollout = std::make_shared<brogameagent::mcts::AggressiveRollout>();

        brogameagent::mcts::ParallelSearchStats stats{};
        auto joint = brogameagent::mcts::root_parallel_search_decoupled(
            worlds, heroId, oppId, cfg, evaluator, rollout, &stats);

        ObjectBuilder res;
        res.set("hero", makeCombatAction(joint.hero));
        res.set("opp",  makeCombatAction(joint.opp));
        ObjectBuilder s;
        s.set("numThreads", ev::fromDouble(stats.num_threads));
        s.set("totalIterations", ev::fromDouble(stats.total_iterations));
        s.set("elapsedMs", ev::fromDouble(stats.elapsed_ms));
        s.set("mergedBestVisits", ev::fromDouble(stats.merged_best_visits));
        res.set("stats", s.get());
        return res.get();
    });
}

} // namespace brogameagent::api
