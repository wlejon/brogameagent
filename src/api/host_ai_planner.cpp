// Hierarchical planners: TacticMcts, LayeredPlanner, TeamOption /
// TeamOptionMcts and Commander.
//
// None of these survived the port; every app that reached for
// createLayeredPlanner / createCommander / createTacticMcts found the name
// missing. They are restored here with the option/evaluator/prior wiring the
// old bodies did, so a planner configured from JS actually uses what it was
// handed.

#include "host_ai_mcts_shared.h"

#include <memory>
#include <vector>

namespace brogameagent::api {

HostClass g_tacticMctsClass;
HostClass g_layeredPlannerClass;
HostClass g_teamOptionClass;
HostClass g_teamOptionMctsClass;
HostClass g_commanderClass;

namespace {

struct HostTacticMcts {
    uint32_t tag = kHostTacticMctsTag;
    std::unique_ptr<bgm::TacticMcts> mcts;
};

struct HostLayeredPlanner {
    uint32_t tag = kHostLayeredPlannerTag;
    std::unique_ptr<bgm::LayeredPlanner> planner;
};

struct HostTeamOptionMcts {
    uint32_t tag = kHostTeamOptionMctsTag;
    std::unique_ptr<bgm::TeamOptionMcts> mcts;
    std::vector<std::shared_ptr<bgm::TeamOption>> options;
};

struct HostCommander {
    uint32_t tag = kHostCommanderTag;
    std::unique_ptr<bgm::Commander> commander;
    // Keeps every role's options alive for as long as the Commander is.
    std::vector<std::shared_ptr<bgm::Option>> optionRefs;
};

HostTacticMcts* unwrapTacticMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTacticMcts*>(ev::handleData(v));
    return (h && h->tag == kHostTacticMctsTag) ? h : nullptr;
}

HostLayeredPlanner* unwrapLayeredPlanner(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostLayeredPlanner*>(ev::handleData(v));
    return (h && h->tag == kHostLayeredPlannerTag) ? h : nullptr;
}

HostTeamOptionMcts* unwrapTeamOptionMcts(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamOptionMcts*>(ev::handleData(v));
    return (h && h->tag == kHostTeamOptionMctsTag) ? h : nullptr;
}

HostCommander* unwrapCommander(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostCommander*>(ev::handleData(v));
    return (h && h->tag == kHostCommanderTag) ? h : nullptr;
}

std::string optionNameArg(std::span<const Value> a, size_t i) {
    if (i >= a.size()) return {};
    if (ev::isString(a[i])) return ev::toUtf8(a[i]);
    if (auto* cell = unwrapOptionCell(a[i])) return cell->opt ? cell->opt->name() : std::string{};
    if (auto* cell = unwrapTeamOptionCell(a[i])) return cell->opt ? cell->opt->name() : std::string{};
    if (ev::isObject(a[i])) return readStringProp(a[i], "name");
    return {};
}

} // namespace

void ensureAIPlannerClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    // ── TacticMcts ─────────────────────────────────────────────────────────
    g_tacticMctsClass.init("AITacticMcts", [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTacticMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::throwTypeError("search(world, heroes)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("search: invalid world");
            return makeTactic(h->mcts->search(w->world, parseHeroes(a[1])));
        });
        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTacticMcts(self);
            if (h && h->mcts && !a.empty()) h->mcts->advance_root(parseTactic(a[0]));
            return ev::undefined();
        });
        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTacticMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });
        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTacticMcts(self);
            return (h && h->mcts) ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // ── LayeredPlanner ─────────────────────────────────────────────────────
    g_layeredPlannerClass.init("AILayeredPlanner", [](ObjectBuilder& b) {
        b.def("decide", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapLayeredPlanner(self);
            if (!h || !h->planner || a.size() < 2) return ev::throwTypeError("decide(world, heroes)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("decide: invalid world");
            auto joint = h->planner->decide(w->world, parseHeroes(a[1]));
            return makeCombatActionArray(joint.per_hero);
        });
        b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapLayeredPlanner(self);
            if (h && h->planner) h->planner->reset();
            return ev::undefined();
        });
        b.accessor("committedTactic", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapLayeredPlanner(self);
            return (h && h->planner) ? makeTactic(h->planner->committed_tactic()) : ev::null();
        }, nullptr);
        b.accessor("windowsUntilReplan", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapLayeredPlanner(self);
            return ev::fromDouble((h && h->planner) ? h->planner->windows_until_replan() : 0);
        }, nullptr);
        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapLayeredPlanner(self);
            if (!h || !h->planner) return ev::null();
            const auto& s = h->planner->last_stats();
            ObjectBuilder o;
            o.set("committedTactic", makeTactic(s.committed_tactic));
            o.set("windowsUntilReplan", ev::fromDouble(s.windows_until_replan));
            o.set("replannedThisCall", ev::fromBool(s.replanned_this_call));
            o.set("tacticStats", makeSearchStats(s.tactic_stats));
            o.set("fineStats", makeSearchStats(s.fine_stats));
            return o.get();
        }, nullptr);
    });

    // ── TeamOption (handle for JS-authored team options) ───────────────────
    g_teamOptionClass.init("AITeamOption", [](ObjectBuilder& b) {
        b.accessor("name", [](Value self, std::span<const Value>) -> Value {
            auto* cell = unwrapTeamOptionCell(self);
            return (cell && cell->opt) ? ev::fromUtf8(cell->opt->name()) : ev::null();
        }, nullptr);
    });

    // ── TeamOptionMcts ─────────────────────────────────────────────────────
    g_teamOptionMctsClass.init("AITeamOptionMcts", [](ObjectBuilder& b) {
        b.def("search", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamOptionMcts(self);
            if (!h || !h->mcts || a.size() < 2) return ev::throwTypeError("search(world, heroes)");
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("search: invalid world");
            const auto* opt = h->mcts->search(w->world, parseHeroes(a[1]));
            return opt ? ev::fromUtf8(opt->name()) : ev::null();
        });
        b.def("advanceRoot", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamOptionMcts(self);
            if (!h || !h->mcts) return ev::undefined();
            std::string target = optionNameArg(a, 0);
            if (target.empty()) {
                h->mcts->reset_tree();
                return ev::undefined();
            }
            const bgm::TeamOption* match = nullptr;
            for (const auto& sp : h->options) {
                if (sp && sp->name() == target) { match = sp.get(); break; }
            }
            h->mcts->advance_root(match);
            return ev::undefined();
        });
        b.def("executeOption", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapTeamOptionMcts(self);
            if (!h || !h->mcts || a.size() < 3) {
                return ev::throwTypeError("executeOption(world, heroes, name)");
            }
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("executeOption: invalid world");
            auto heroes = parseHeroes(a[1]);
            std::string target = optionNameArg(a, 2);
            for (const auto& sp : h->options) {
                if (sp && sp->name() == target) {
                    return ev::fromDouble(h->mcts->execute_option(w->world, heroes, *sp));
                }
            }
            return ev::fromDouble(0);
        });
        b.def("resetTree", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamOptionMcts(self);
            if (h && h->mcts) h->mcts->reset_tree();
            return ev::undefined();
        });
        b.accessor("lastStats", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapTeamOptionMcts(self);
            return (h && h->mcts) ? makeSearchStats(h->mcts->last_stats()) : ev::null();
        }, nullptr);
    });

    // ── Commander ──────────────────────────────────────────────────────────
    g_commanderClass.init("AICommander", [](ObjectBuilder& b) {
        b.def("decide", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapCommander(self);
            if (!h || !h->commander || a.size() < 2) {
                return ev::throwTypeError("decide(world, heroes)");
            }
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("decide: invalid world");
            return makeCombatActionArray(h->commander->decide(w->world, parseHeroes(a[1])));
        });
        b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapCommander(self);
            if (h && h->commander) h->commander->reset();
            return ev::undefined();
        });
        b.def("committedOption", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapCommander(self);
            if (!h || !h->commander || a.empty()) return ev::null();
            std::string n = h->commander->committed_option_for_hero(
                static_cast<size_t>(std::max(0, i32At(a, 0))));
            return n.empty() ? ev::null() : ev::fromUtf8(n);
        });
        b.accessor("currentAssignments", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapCommander(self);
            if (!h || !h->commander) return hostArrayOf(0, [](size_t) { return ev::null(); });
            const auto& as = h->commander->current_assignments();
            return hostArrayOf(as.size(), [&](size_t i) { return ev::fromDouble(as[i]); });
        }, nullptr);
        b.accessor("windowsUntilReplan", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapCommander(self);
            return ev::fromDouble((h && h->commander) ? h->commander->windows_until_replan() : 0);
        }, nullptr);
        b.accessor("roles", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapCommander(self);
            if (!h || !h->commander) return hostArrayOf(0, [](size_t) { return ev::null(); });
            const auto& roles = h->commander->roles();
            return hostArrayOf(roles.size(), [&](size_t i) {
                ObjectBuilder o;
                o.set("name", ev::fromUtf8(roles[i].name));
                o.set("optionCount", ev::fromDouble(static_cast<double>(roles[i].options.size())));
                return o.get();
            });
        }, nullptr);
    });
}

void installAIPlanner(ObjectBuilder& game) {
    ensureAIPlannerClassesInstalled();

    game.def("createTacticMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostTacticMcts>();
        cell->mcts = std::make_unique<bgm::TacticMcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto op = parseOpponentPolicy(opts.get())) {
                cell->mcts->set_opponent_policy(std::move(op));
            }
            if (auto tev = parseTeamEvaluator(opts.get())) {
                cell->mcts->set_evaluator(std::move(tev));
            }
        }
        return g_tacticMctsClass.createInstance(std::move(cell));
    });

    game.def("createLayeredPlanner", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostLayeredPlanner>();
        cell->planner = std::make_unique<bgm::LayeredPlanner>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            bgm::LayeredPlanner::Config cfg{};
            Value tc = ev::getProperty(opts.get(), "tactic");
            if (ev::isObject(tc)) cfg.tactic_cfg = parseMctsConfig(tc);
            Value fc = ev::getProperty(opts.get(), "fine");
            if (ev::isObject(fc)) cfg.fine_cfg = parseMctsConfig(fc);
            Value tmw = ev::getProperty(opts.get(), "tacticMatchWeight");
            if (ev::isNumber(tmw)) cfg.tactic_match_weight = static_cast<float>(ev::toDouble(tmw));
            Value tow = ev::getProperty(opts.get(), "tacticOtherWeight");
            if (ev::isNumber(tow)) cfg.tactic_other_weight = static_cast<float>(ev::toDouble(tow));
            cell->planner->set_config(cfg);

            if (auto p = parseRolloutPolicy(opts.get())) cell->planner->set_rollout_policy(std::move(p));
            if (auto op = parseOpponentPolicy(opts.get())) cell->planner->set_opponent_policy(std::move(op));
            if (auto tev = parseTeamEvaluator(opts.get())) cell->planner->set_team_evaluator(std::move(tev));
        }
        return g_layeredPlannerClass.createInstance(std::move(cell));
    });

    game.def("createTeamOption", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createTeamOption(spec)");
        ev::Persistent spec(a[0]);
        std::string name = readStringProp(spec.get(), "name");
        if (name.empty()) return ev::throwTypeError("createTeamOption: name required");

        ev::Persistent ci(ev::getProperty(spec.get(), "canInitiate"));
        ev::Persistent st(ev::getProperty(spec.get(), "step"));
        ev::Persistent te(ev::getProperty(spec.get(), "shouldTerminate"));
        if (!ev::isFunction(ci.get())) return ev::throwTypeError("createTeamOption: canInitiate must be a function");
        if (!ev::isFunction(st.get())) return ev::throwTypeError("createTeamOption: step must be a function");
        if (!ev::isFunction(te.get())) return ev::throwTypeError("createTeamOption: shouldTerminate must be a function");

        auto cell = std::make_unique<HostTeamOptionCell>();
        cell->opt = makeJsTeamOption(std::move(name), ci.get(), st.get(), te.get());
        return g_teamOptionClass.createInstance(std::move(cell));
    });

    game.def("createTeamOptionMcts", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostTeamOptionMcts>();
        cell->mcts = std::make_unique<bgm::TeamOptionMcts>();
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent opts(a[0]);
            cell->mcts->set_config(parseMctsConfig(opts.get()));
            if (auto op = parseOpponentPolicy(opts.get())) cell->mcts->set_opponent_policy(std::move(op));
            if (auto tev = parseTeamEvaluator(opts.get())) cell->mcts->set_evaluator(std::move(tev));
            cell->options = parseTeamOptionArray(opts.get());
            if (!cell->options.empty()) {
                auto copy = cell->options;
                cell->mcts->set_options(std::move(copy));
            }
        }
        return g_teamOptionMctsClass.createInstance(std::move(cell));
    });

    game.def("createCommander", 1, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostCommander>();
        cell->commander = std::make_unique<bgm::Commander>();
        if (a.empty() || !ev::isObject(a[0])) {
            return g_commanderClass.createInstance(std::move(cell));
        }
        ev::Persistent opts(a[0]);

        bgm::Commander::Config cfg{};
        Value rc = ev::getProperty(opts.get(), "roleCfg");
        if (ev::isObject(rc)) cfg.role_cfg = parseMctsConfig(rc);
        cfg.replan_every_windows = static_cast<int>(
            getDoubleProperty(opts.get(), "replanEveryWindows", cfg.replan_every_windows));
        cell->commander->set_config(cfg);

        if (auto op = parseOpponentPolicy(opts.get())) cell->commander->set_opponent_policy(std::move(op));
        if (auto hev = parseHeroEvaluator(opts.get())) cell->commander->set_default_evaluator(std::move(hev));

        ev::Persistent rolesArr(ev::getProperty(opts.get(), "roles"));
        if (ev::isObject(rolesArr.get())) {
            Value lenV = ev::getProperty(rolesArr.get(), "length");
            uint32_t n = ev::isNumber(lenV) ? static_cast<uint32_t>(ev::toDouble(lenV)) : 0u;
            for (uint32_t i = 0; i < n; ++i) {
                ev::Persistent role(ev::getElement(rolesArr.get(), i));
                if (!ev::isObject(role.get())) continue;
                std::string name = readStringProp(role.get(), "name");
                auto roleOptions = parseOptionArray(role.get());
                auto roleEval = parseHeroEvaluator(role.get());
                for (const auto& sp : roleOptions) cell->optionRefs.push_back(sp);
                cell->commander->add_role(std::move(name), std::move(roleOptions),
                                          std::move(roleEval));
            }
        }

        Value assignFn = ev::getProperty(opts.get(), "assign");
        if (ev::isFunction(assignFn)) {
            cell->commander->set_assigner(makeJsAssigner(assignFn));
        }

        return g_commanderClass.createInstance(std::move(cell));
    });
}

} // namespace brogameagent::api
