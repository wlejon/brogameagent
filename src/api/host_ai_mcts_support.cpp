// CombatAction / Tactic / MctsConfig marshalling, the agent & world views JS
// callbacks receive, and the JS-callback adapters for rollout policies,
// priors, evaluators, options and the Commander's role assigner.

#include "host_ai_mcts_shared.h"

#include <algorithm>
#include <cmath>

namespace brogameagent::api {

namespace {

const char* tacticKindStr(bgm::TacticKind k) {
    switch (k) {
        case bgm::TacticKind::FocusLowestHp: return "FocusLowestHp";
        case bgm::TacticKind::Scatter:       return "Scatter";
        case bgm::TacticKind::Retreat:       return "Retreat";
        default:                             return "Hold";
    }
}

bgm::TacticKind parseTacticKindStr(const std::string& s) {
    if (s == "FocusLowestHp") return bgm::TacticKind::FocusLowestHp;
    if (s == "Scatter")       return bgm::TacticKind::Scatter;
    if (s == "Retreat")       return bgm::TacticKind::Retreat;
    return bgm::TacticKind::Hold;
}

uint32_t arrayLength(Value v) {
    if (!ev::isObject(v)) return 0;
    return toLength(ev::getProperty(v, "length"));
}

// ─── JS-callback adapters ──────────────────────────────────────────────────
//
// Each holds JsCallbackSlots, not the functions: the functions live on the
// owning search's JS object and are bound only while one of its methods runs
// (host_js_callbacks.h). An unbound slot answers the neutral default — a
// default action, uniform weights, a 0 value, an option that never starts.

class JsRolloutPolicy : public bgm::IRolloutPolicy {
public:
    explicit JsRolloutPolicy(JsSlotPtr fn) : fn_(std::move(fn)) {}

    bgm::CombatAction choose(brogameagent::Agent& self,
                             brogameagent::World& world) const override {
        if (!fn_->bound()) return {};
        ev::Persistent selfV(buildAgentFields(self));
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { selfV.get(), worldV.get() };
        auto r = fn_->call(args);
        if (r.thrown || !ev::isObject(r.value)) return {};
        return parseCombatAction(r.value);
    }

private:
    JsSlotPtr fn_;
};

class JsPrior : public bgm::IPrior {
public:
    explicit JsPrior(JsSlotPtr fn) : fn_(std::move(fn)) {}

    std::vector<float> score(const brogameagent::Agent& self,
                             const brogameagent::World& world,
                             const std::vector<bgm::CombatAction>& actions) const override {
        if (!fn_->bound()) return std::vector<float>(actions.size(), 1.0f);
        ev::Persistent selfV(buildAgentFields(self));
        ev::Persistent worldV(buildWorldView(world));
        ev::Persistent actsV(makeCombatActionArray(actions));
        Value args[3] = { selfV.get(), worldV.get(), actsV.get() };
        auto r = fn_->call(args);

        std::vector<float> weights(actions.size(), 1.0f);
        if (!r.thrown && ev::isObject(r.value)) {
            ev::Persistent res(r.value);
            uint32_t len = arrayLength(res.get());
            uint32_t n = std::min<uint32_t>(len, static_cast<uint32_t>(actions.size()));
            for (uint32_t i = 0; i < n; ++i) {
                Value el = ev::getElement(res.get(), i);
                double d = ev::isNumber(el) ? ev::toDouble(el) : 0.0;
                if (!std::isfinite(d)) d = 0.0;
                weights[i] = static_cast<float>(std::max(0.0, d));
            }
        }
        return weights;
    }

private:
    JsSlotPtr fn_;
};

// Evaluators answer a clamped [-1, 1] score; 0 when unbound or non-numeric.
float clampedScore(const ev::CallResult& r) {
    if (r.thrown || !ev::isNumber(r.value)) return 0.0f;
    double d = ev::toDouble(r.value);
    if (!std::isfinite(d)) return 0.0f;
    return static_cast<float>(std::clamp(d, -1.0, 1.0));
}

class JsEvaluator : public bgm::IEvaluator {
public:
    explicit JsEvaluator(JsSlotPtr fn) : fn_(std::move(fn)) {}

    float evaluate(const brogameagent::World& world, int heroId) const override {
        if (!fn_->bound()) return 0.0f;
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { worldV.get(), ev::fromDouble(heroId) };
        return clampedScore(fn_->call(args));
    }

private:
    JsSlotPtr fn_;
};

class JsTeamEvaluator : public bgm::ITeamEvaluator {
public:
    explicit JsTeamEvaluator(JsSlotPtr fn) : fn_(std::move(fn)) {}

    float evaluate(const brogameagent::World& world, int teamId) const override {
        if (!fn_->bound()) return 0.0f;
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { worldV.get(), ev::fromDouble(teamId) };
        return clampedScore(fn_->call(args));
    }

private:
    JsSlotPtr fn_;
};

// An unbound option never starts and, if running, ends at once.
class JsOption : public bgm::Option {
public:
    JsOption(std::string name, JsSlotPtr canInit, JsSlotPtr step, JsSlotPtr shouldTerm)
        : name_(std::move(name)), canInit_(std::move(canInit)), step_(std::move(step)),
          shouldTerm_(std::move(shouldTerm)) {}

    const std::string& name() const override { return name_; }

    bool can_initiate(const brogameagent::Agent& self,
                      const brogameagent::World& world) const override {
        if (!canInit_->bound()) return false;
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { sv.get(), wv.get() };
        auto r = canInit_->call(args);
        return !r.thrown && ev::toBool(r.value);
    }

    bgm::CombatAction step(brogameagent::Agent& self, brogameagent::World& world,
                           int ticksInOption) const override {
        if (!step_->bound()) return {};
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { sv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = step_->call(args);
        if (r.thrown || !ev::isObject(r.value)) return {};
        return parseCombatAction(r.value);
    }

    bool should_terminate(const brogameagent::Agent& self,
                          const brogameagent::World& world,
                          int ticksInOption) const override {
        if (!shouldTerm_->bound()) return true;
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { sv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = shouldTerm_->call(args);
        return !r.thrown && ev::toBool(r.value);
    }

private:
    std::string name_;
    JsSlotPtr canInit_, step_, shouldTerm_;
};

class JsTeamOption : public bgm::TeamOption {
public:
    JsTeamOption(std::string name, JsSlotPtr canInit, JsSlotPtr step, JsSlotPtr shouldTerm)
        : name_(std::move(name)), canInit_(std::move(canInit)), step_(std::move(step)),
          shouldTerm_(std::move(shouldTerm)) {}

    const std::string& name() const override { return name_; }

    bool can_initiate(const std::vector<brogameagent::Agent*>& heroes,
                      const brogameagent::World& world) const override {
        if (!canInit_->bound()) return false;
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { hv.get(), wv.get() };
        auto r = canInit_->call(args);
        return !r.thrown && ev::toBool(r.value);
    }

    std::vector<bgm::CombatAction> step(const std::vector<brogameagent::Agent*>& heroes,
                                        brogameagent::World& world,
                                        int ticksInOption) const override {
        std::vector<bgm::CombatAction> out(heroes.size());
        if (!step_->bound()) return out;
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { hv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = step_->call(args);
        if (!r.thrown && ev::isObject(r.value)) {
            out = parseCombatActionArray(r.value);
            out.resize(heroes.size());
        }
        return out;
    }

    bool should_terminate(const std::vector<brogameagent::Agent*>& heroes,
                          const brogameagent::World& world,
                          int ticksInOption) const override {
        if (!shouldTerm_->bound()) return true;
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { hv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = shouldTerm_->call(args);
        return !r.thrown && ev::toBool(r.value);
    }

private:
    std::string name_;
    JsSlotPtr canInit_, step_, shouldTerm_;
};

} // namespace

// ---------------------------------------------------------------------------
// Marshalling
// ---------------------------------------------------------------------------

Value makeCombatAction(const bgm::CombatAction& a) {
    ObjectBuilder o;
    o.set("moveDir", ev::fromDouble(static_cast<int>(a.move_dir)));
    o.set("attackSlot", ev::fromDouble(a.attack_slot));
    o.set("abilitySlot", ev::fromDouble(a.ability_slot));
    return o.get();
}

/// Also parses rollout / opponent policy results mid-search, so it never
/// throws: a moveDir outside Hold..NW is Hold, a slot outside int8 is -1
/// (none) — an out-of-range enum or int8 cast would index the search's
/// tables with garbage.
bgm::CombatAction parseCombatAction(Value v) {
    bgm::CombatAction a{};
    if (ev::isObject(v)) {
        ev::Persistent root(v);
        a.move_dir = static_cast<bgm::MoveDir>(intOr(
            getDoubleProperty(root.get(), "moveDir", 0), 0, 0,
            static_cast<int32_t>(bgm::MoveDir::NW)));
        a.attack_slot = static_cast<int8_t>(
            intOr(getDoubleProperty(root.get(), "attackSlot", -1), -1, -128, 127));
        a.ability_slot = static_cast<int8_t>(
            intOr(getDoubleProperty(root.get(), "abilitySlot", -1), -1, -128, 127));
    }
    return a;
}

std::vector<bgm::CombatAction> parseCombatActionArray(Value arr) {
    std::vector<bgm::CombatAction> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent root(arr);
    uint32_t len = arrayLength(root.get());
    out.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        out.push_back(parseCombatAction(ev::getElement(root.get(), i)));
    }
    return out;
}

Value makeCombatActionArray(const std::vector<bgm::CombatAction>& acts) {
    return hostArrayOf(acts.size(), [&](size_t i) { return makeCombatAction(acts[i]); });
}

Value makeTactic(const bgm::Tactic& t) {
    ObjectBuilder o;
    o.set("kind", ev::fromUtf8(tacticKindStr(t.kind)));
    return o.get();
}

bgm::Tactic parseTactic(Value v) {
    bgm::Tactic t{};
    if (ev::isString(v)) {
        t.kind = parseTacticKindStr(ev::toUtf8(v));
    } else if (ev::isObject(v)) {
        t.kind = parseTacticKindStr(readStringProp(v, "kind"));
    }
    return t;
}

bgm::MctsConfig parseMctsConfig(Value opts, const bgm::MctsConfig& base) {
    bgm::MctsConfig c = base;
    if (!ev::isObject(opts)) return c;
    ev::Persistent root(opts);
    c.iterations = getI32Property(root.get(), "iterations", c.iterations, 0);
    c.budget_ms = getI32Property(root.get(), "budgetMs", c.budget_ms, 0);
    c.rollout_horizon = getI32Property(root.get(), "rolloutHorizon", c.rollout_horizon, 0);
    c.sim_dt = static_cast<float>(getDoubleProperty(root.get(), "simDt", c.sim_dt));
    c.action_repeat = getI32Property(root.get(), "actionRepeat", c.action_repeat, 1);
    c.uct_c = static_cast<float>(getDoubleProperty(root.get(), "uctC", c.uct_c));
    c.seed = getU64Property(root.get(), "seed", c.seed);
    c.tactic_window_decisions = getI32Property(root.get(), "tacticWindowDecisions",
                                               c.tactic_window_decisions, 0);
    c.pw_alpha = static_cast<float>(getDoubleProperty(root.get(), "pwAlpha", c.pw_alpha));
    c.prior_c = static_cast<float>(getDoubleProperty(root.get(), "priorC", c.prior_c));
    c.option_max_windows = getI32Property(root.get(), "optionMaxWindows",
                                          c.option_max_windows, 0);
    c.use_leaf_value = getBoolProperty(root.get(), "useLeafValue", c.use_leaf_value);
    return c;
}

Value makeSearchStats(const bgm::SearchStats& s) {
    ObjectBuilder o;
    o.set("iterations", ev::fromDouble(s.iterations));
    o.set("rootChildren", ev::fromDouble(s.root_children));
    o.set("treeSize", ev::fromDouble(s.tree_size));
    o.set("bestMean", ev::fromDouble(s.best_mean));
    o.set("bestVisits", ev::fromDouble(s.best_visits));
    o.set("elapsedMs", ev::fromDouble(s.elapsed_ms));
    o.set("reusedRoot", ev::fromBool(s.reused_root));
    return o.get();
}

std::vector<brogameagent::Agent*> parseHeroes(Value arr) {
    std::vector<brogameagent::Agent*> heroes;
    if (!ev::isObject(arr)) return heroes;
    ev::Persistent root(arr);
    uint32_t n = arrayLength(root.get());
    heroes.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (auto* ag = unwrapAgent(ev::getElement(root.get(), i))) heroes.push_back(&ag->agent);
    }
    return heroes;
}

std::string readStringProp(Value obj, const char* key) {
    if (!ev::isObject(obj)) return {};
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    return ev::isString(v) ? ev::toUtf8(v) : std::string{};
}

Value buildAgentFields(const brogameagent::Agent& a) {
    const auto& u = a.unit();
    ObjectBuilder o;
    o.set("id", ev::fromDouble(u.id));
    o.set("teamId", ev::fromDouble(u.teamId));
    o.set("x", ev::fromDouble(a.x()));
    o.set("z", ev::fromDouble(a.z()));
    o.set("yaw", ev::fromDouble(a.yaw()));
    o.set("hp", ev::fromDouble(u.hp));
    o.set("maxHp", ev::fromDouble(u.maxHp));
    o.set("alive", ev::fromBool(u.alive()));
    o.set("attackRange", ev::fromDouble(u.attackRange));
    o.set("attackCooldown", ev::fromDouble(u.attackCooldown));
    o.set("mana", ev::fromDouble(u.mana));
    o.set("maxMana", ev::fromDouble(u.maxMana));
    o.set("abilities", hostArrayOf(brogameagent::Unit::MAX_ABILITIES, [&](size_t i) {
        ObjectBuilder slot;
        slot.set("abilityId", ev::fromDouble(u.abilitySlot[i]));
        slot.set("cooldown", ev::fromDouble(u.abilityCooldowns[i]));
        return slot.get();
    }));
    return o.get();
}

Value buildWorldView(const brogameagent::World& world) {
    const auto& agents = world.agents();
    std::vector<const brogameagent::Agent*> live;
    live.reserve(agents.size());
    for (const brogameagent::Agent* a : agents) {
        if (a) live.push_back(a);
    }
    ObjectBuilder o;
    o.set("agents", hostArrayOf(live.size(), [&](size_t i) {
        return buildAgentFields(*live[i]);
    }));
    return o.get();
}

Value buildHeroesView(const std::vector<brogameagent::Agent*>& heroes) {
    return hostArrayOf(heroes.size(), [&](size_t i) {
        return heroes[i] ? buildAgentFields(*heroes[i]) : ev::null();
    });
}

// ---------------------------------------------------------------------------
// Option parsing
// ---------------------------------------------------------------------------

std::shared_ptr<bgm::IRolloutPolicy> parseRolloutPolicy(Value opts, JsCallbackSet& cbs) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value v = ev::getProperty(root.get(), "rolloutPolicy");
    if (auto* cell = unwrapRolloutCell(v)) return cell->p;
    if (ev::isFunction(v)) return std::make_shared<JsRolloutPolicy>(cbs.add(v));
    if (!ev::isString(v)) return nullptr;
    std::string kind = ev::toUtf8(v);
    if (kind == "aggressive") return std::make_shared<bgm::AggressiveRollout>();
    if (kind == "scripted")   return std::make_shared<bgm::ScriptedRollout>();
    if (kind == "random")     return std::make_shared<bgm::RandomRollout>();
    return nullptr;
}

bgm::OpponentPolicy parseOpponentPolicy(Value opts) {
    std::string kind = readStringProp(opts, "opponentPolicy");
    if (kind == "aggressive") return bgm::policy_aggressive;
    if (kind == "scripted")   return bgm::policy_scripted;
    if (kind == "idle")       return bgm::policy_idle;
    return {};
}

std::shared_ptr<bgm::IPrior> parsePrior(Value opts, JsCallbackSet& cbs) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    ev::Persistent pv(ev::getProperty(root.get(), "prior"));
    if (auto sp = extractPriorShared(pv.get())) return sp;
    if (auto* cell = unwrapPriorCell(pv.get())) return cell->p;
    if (ev::isFunction(pv.get())) return std::make_shared<JsPrior>(cbs.add(pv.get()));
    if (!ev::isString(pv.get())) return nullptr;
    std::string kind = ev::toUtf8(pv.get());
    if (kind == "uniform")    return std::make_shared<bgm::UniformPrior>();
    if (kind == "attackBias") return std::make_shared<bgm::AttackBiasPrior>();
    if (kind == "tacticMatch") {
        auto tp = std::make_shared<bgm::TacticPrior>();
        Value tv = ev::getProperty(root.get(), "tactic");
        if (!ev::isUndefined(tv) && !ev::isNull(tv)) tp->set_tactic(parseTactic(tv));
        tp->set_match_weight(
            static_cast<float>(getDoubleProperty(root.get(), "tacticMatchWeight", 8.0)));
        tp->set_other_weight(
            static_cast<float>(getDoubleProperty(root.get(), "tacticOtherWeight", 1.0)));
        return tp;
    }
    return nullptr;
}

std::shared_ptr<bgm::IEvaluator> parseHeroEvaluator(Value opts, JsCallbackSet& cbs) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    ev::Persistent v(ev::getProperty(root.get(), "evaluator"));
    if (auto se = extractHeroEvaluatorShared(v.get())) return se;
    if (auto* cell = unwrapEvaluatorCell(v.get())) return cell->p;
    if (ev::isFunction(v.get())) return std::make_shared<JsEvaluator>(cbs.add(v.get()));
    if (ev::isString(v.get()) && ev::toUtf8(v.get()) == "hpDelta") {
        return std::make_shared<bgm::HpDeltaEvaluator>();
    }
    return nullptr;
}

std::shared_ptr<bgm::ITeamEvaluator> parseTeamEvaluator(Value opts, JsCallbackSet& cbs) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value v = ev::getProperty(root.get(), "evaluator");
    if (auto* cell = unwrapTeamEvaluatorCell(v)) return cell->p;
    if (ev::isFunction(v)) return std::make_shared<JsTeamEvaluator>(cbs.add(v));
    if (!ev::isString(v)) return nullptr;
    std::string kind = ev::toUtf8(v);
    if (kind == "teamHpDelta")   return std::make_shared<bgm::TeamHpDeltaEvaluator>();
    if (kind == "teamAdvantage") return std::make_shared<bgm::TeamAdvantageEvaluator>();
    if (kind == "teamPosition")  return std::make_shared<bgm::TeamPositionEvaluator>();
    return nullptr;
}

std::vector<std::shared_ptr<bgm::Option>> parseOptionArray(Value opts, JsCallbackSet& cbs) {
    std::vector<std::shared_ptr<bgm::Option>> out;
    if (!ev::isObject(opts)) return out;
    ev::Persistent root(opts);
    ev::Persistent arr(ev::getProperty(root.get(), "options"));
    uint32_t n = arrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent el(ev::getElement(arr.get(), i));
        auto* cell = unwrapOptionCell(el.get());
        if (!cell || !cell->opt) continue;
        out.push_back(cell->opt);
        cbs.adopt(el.get(), cell->slots);
    }
    return out;
}

std::vector<std::shared_ptr<bgm::TeamOption>> parseTeamOptionArray(Value opts,
                                                                  JsCallbackSet& cbs) {
    std::vector<std::shared_ptr<bgm::TeamOption>> out;
    if (!ev::isObject(opts)) return out;
    ev::Persistent root(opts);
    ev::Persistent arr(ev::getProperty(root.get(), "options"));
    uint32_t n = arrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent el(ev::getElement(arr.get(), i));
        auto* cell = unwrapTeamOptionCell(el.get());
        if (!cell || !cell->opt) continue;
        out.push_back(cell->opt);
        cbs.adopt(el.get(), cell->slots);
    }
    return out;
}

bgm::Commander::AssignFn makeJsAssigner(Value fn, JsCallbackSet& cbs) {
    if (!ev::isFunction(fn)) return {};
    JsSlotPtr slot = cbs.add(fn);
    return [slot](const std::vector<brogameagent::Agent*>& heroes,
                  const brogameagent::World& world) -> std::vector<int> {
        std::vector<int> out(heroes.size(), 0);
        if (!slot->bound()) return out;
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { hv.get(), wv.get() };
        auto r = slot->call(args);
        if (!r.thrown && ev::isObject(r.value)) {
            ev::Persistent res(r.value);
            uint32_t len = arrayLength(res.get());
            uint32_t n = std::min<uint32_t>(len, static_cast<uint32_t>(heroes.size()));
            for (uint32_t i = 0; i < n; ++i) {
                Value el = ev::getElement(res.get(), i);
                const double d = ev::isNumber(el) ? ev::toDouble(el) : 0.0;
                // A NaN / out-of-int role index is 0, not an undefined cast.
                out[i] = (std::isfinite(d) && d >= 0.0 && d <= 2147483647.0)
                             ? static_cast<int>(d) : 0;
            }
        }
        return out;
    };
}

std::shared_ptr<bgm::Option> makeJsOption(std::string name, Value canInit,
                                          Value step, Value shouldTerm, JsCallbackSet& cbs) {
    if (!ev::isFunction(canInit) || !ev::isFunction(step) || !ev::isFunction(shouldTerm)) {
        return nullptr;
    }
    JsSlotPtr ci = cbs.add(canInit);
    JsSlotPtr st = cbs.add(step);
    JsSlotPtr te = cbs.add(shouldTerm);
    return std::make_shared<JsOption>(std::move(name), ci, st, te);
}

std::shared_ptr<bgm::TeamOption> makeJsTeamOption(std::string name, Value canInit,
                                                  Value step, Value shouldTerm,
                                                  JsCallbackSet& cbs) {
    if (!ev::isFunction(canInit) || !ev::isFunction(step) || !ev::isFunction(shouldTerm)) {
        return nullptr;
    }
    JsSlotPtr ci = cbs.add(canInit);
    JsSlotPtr st = cbs.add(step);
    JsSlotPtr te = cbs.add(shouldTerm);
    return std::make_shared<JsTeamOption>(std::move(name), ci, st, te);
}

WorldArgScope::WorldArgScope(Value v) : world_(v) {
    if (HostWorld* w = unwrapWorld(world_.get())) {
        scope_ = std::make_unique<ActiveWorldScope>(w, world_.get());
    }
}

} // namespace brogameagent::api
