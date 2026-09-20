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
    Value lenV = ev::getProperty(v, "length");
    if (!ev::isNumber(lenV)) return 0;
    double d = ev::toDouble(lenV);
    return (d > 0.0) ? static_cast<uint32_t>(d) : 0u;
}

// ─── JS-callback adapters ──────────────────────────────────────────────────
//
// A bronze ev::Persistent is itself a GC root, so unlike the QuickJS
// originals these need no gc_mark hook: holding the Persistent is what keeps
// the callback alive.

class JsRolloutPolicy : public bgm::IRolloutPolicy {
public:
    explicit JsRolloutPolicy(Value fn) : fn_(fn) {}

    bgm::CombatAction choose(brogameagent::Agent& self,
                             brogameagent::World& world) const override {
        ev::Persistent selfV(buildAgentFields(self));
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { selfV.get(), worldV.get() };
        auto r = ev::call(fn_.get(), ev::undefined(), args);
        if (r.thrown || !ev::isObject(r.value)) return {};
        return parseCombatAction(r.value);
    }

private:
    ev::Persistent fn_;
};

class JsPrior : public bgm::IPrior {
public:
    explicit JsPrior(Value fn) : fn_(fn) {}

    std::vector<float> score(const brogameagent::Agent& self,
                             const brogameagent::World& world,
                             const std::vector<bgm::CombatAction>& actions) const override {
        ev::Persistent selfV(buildAgentFields(self));
        ev::Persistent worldV(buildWorldView(world));
        ev::Persistent actsV(makeCombatActionArray(actions));
        Value args[3] = { selfV.get(), worldV.get(), actsV.get() };
        auto r = ev::call(fn_.get(), ev::undefined(), args);

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
    ev::Persistent fn_;
};

class JsEvaluator : public bgm::IEvaluator {
public:
    explicit JsEvaluator(Value fn) : fn_(fn) {}

    float evaluate(const brogameagent::World& world, int heroId) const override {
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { worldV.get(), ev::fromDouble(heroId) };
        auto r = ev::call(fn_.get(), ev::undefined(), args);
        if (r.thrown || !ev::isNumber(r.value)) return 0.0f;
        double d = ev::toDouble(r.value);
        if (!std::isfinite(d)) return 0.0f;
        return static_cast<float>(std::clamp(d, -1.0, 1.0));
    }

private:
    ev::Persistent fn_;
};

class JsTeamEvaluator : public bgm::ITeamEvaluator {
public:
    explicit JsTeamEvaluator(Value fn) : fn_(fn) {}

    float evaluate(const brogameagent::World& world, int teamId) const override {
        ev::Persistent worldV(buildWorldView(world));
        Value args[2] = { worldV.get(), ev::fromDouble(teamId) };
        auto r = ev::call(fn_.get(), ev::undefined(), args);
        if (r.thrown || !ev::isNumber(r.value)) return 0.0f;
        double d = ev::toDouble(r.value);
        if (!std::isfinite(d)) return 0.0f;
        return static_cast<float>(std::clamp(d, -1.0, 1.0));
    }

private:
    ev::Persistent fn_;
};

class JsOption : public bgm::Option {
public:
    JsOption(std::string name, Value canInit, Value step, Value shouldTerm)
        : name_(std::move(name)), canInit_(canInit), step_(step), shouldTerm_(shouldTerm) {}

    const std::string& name() const override { return name_; }

    bool can_initiate(const brogameagent::Agent& self,
                      const brogameagent::World& world) const override {
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { sv.get(), wv.get() };
        auto r = ev::call(canInit_.get(), ev::undefined(), args);
        return !r.thrown && ev::toBool(r.value);
    }

    bgm::CombatAction step(brogameagent::Agent& self, brogameagent::World& world,
                           int ticksInOption) const override {
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { sv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = ev::call(step_.get(), ev::undefined(), args);
        if (r.thrown || !ev::isObject(r.value)) return {};
        return parseCombatAction(r.value);
    }

    bool should_terminate(const brogameagent::Agent& self,
                          const brogameagent::World& world,
                          int ticksInOption) const override {
        ev::Persistent sv(buildAgentFields(self));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { sv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = ev::call(shouldTerm_.get(), ev::undefined(), args);
        return !r.thrown && ev::toBool(r.value);
    }

private:
    std::string name_;
    ev::Persistent canInit_;
    ev::Persistent step_;
    ev::Persistent shouldTerm_;
};

class JsTeamOption : public bgm::TeamOption {
public:
    JsTeamOption(std::string name, Value canInit, Value step, Value shouldTerm)
        : name_(std::move(name)), canInit_(canInit), step_(step), shouldTerm_(shouldTerm) {}

    const std::string& name() const override { return name_; }

    bool can_initiate(const std::vector<brogameagent::Agent*>& heroes,
                      const brogameagent::World& world) const override {
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { hv.get(), wv.get() };
        auto r = ev::call(canInit_.get(), ev::undefined(), args);
        return !r.thrown && ev::toBool(r.value);
    }

    std::vector<bgm::CombatAction> step(const std::vector<brogameagent::Agent*>& heroes,
                                        brogameagent::World& world,
                                        int ticksInOption) const override {
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { hv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = ev::call(step_.get(), ev::undefined(), args);
        std::vector<bgm::CombatAction> out(heroes.size());
        if (!r.thrown && ev::isObject(r.value)) {
            out = parseCombatActionArray(r.value);
            out.resize(heroes.size());
        }
        return out;
    }

    bool should_terminate(const std::vector<brogameagent::Agent*>& heroes,
                          const brogameagent::World& world,
                          int ticksInOption) const override {
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[3] = { hv.get(), wv.get(), ev::fromDouble(ticksInOption) };
        auto r = ev::call(shouldTerm_.get(), ev::undefined(), args);
        return !r.thrown && ev::toBool(r.value);
    }

private:
    std::string name_;
    ev::Persistent canInit_;
    ev::Persistent step_;
    ev::Persistent shouldTerm_;
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

bgm::CombatAction parseCombatAction(Value v) {
    bgm::CombatAction a{};
    if (ev::isObject(v)) {
        ev::Persistent root(v);
        a.move_dir = static_cast<bgm::MoveDir>(
            static_cast<int>(getDoubleProperty(root.get(), "moveDir", 0)));
        a.attack_slot = static_cast<int8_t>(getDoubleProperty(root.get(), "attackSlot", -1));
        a.ability_slot = static_cast<int8_t>(getDoubleProperty(root.get(), "abilitySlot", -1));
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

bgm::MctsConfig parseMctsConfig(Value opts) {
    bgm::MctsConfig c{};
    if (!ev::isObject(opts)) return c;
    ev::Persistent root(opts);
    c.iterations = static_cast<int>(getDoubleProperty(root.get(), "iterations", c.iterations));
    c.budget_ms = static_cast<int>(getDoubleProperty(root.get(), "budgetMs", c.budget_ms));
    c.rollout_horizon = static_cast<int>(getDoubleProperty(root.get(), "rolloutHorizon", c.rollout_horizon));
    c.sim_dt = static_cast<float>(getDoubleProperty(root.get(), "simDt", c.sim_dt));
    c.action_repeat = static_cast<int>(getDoubleProperty(root.get(), "actionRepeat", c.action_repeat));
    c.uct_c = static_cast<float>(getDoubleProperty(root.get(), "uctC", c.uct_c));
    c.seed = getU64Property(root.get(), "seed", c.seed);
    c.tactic_window_decisions = static_cast<int>(
        getDoubleProperty(root.get(), "tacticWindowDecisions", c.tactic_window_decisions));
    c.pw_alpha = static_cast<float>(getDoubleProperty(root.get(), "pwAlpha", c.pw_alpha));
    c.prior_c = static_cast<float>(getDoubleProperty(root.get(), "priorC", c.prior_c));
    c.option_max_windows = static_cast<int>(
        getDoubleProperty(root.get(), "optionMaxWindows", c.option_max_windows));
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

std::shared_ptr<bgm::IRolloutPolicy> parseRolloutPolicy(Value opts) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value v = ev::getProperty(root.get(), "rolloutPolicy");
    if (auto* cell = unwrapRolloutCell(v)) return cell->p;
    if (ev::isFunction(v)) return std::make_shared<JsRolloutPolicy>(v);
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

std::shared_ptr<bgm::IPrior> parsePrior(Value opts) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value pv = ev::getProperty(root.get(), "prior");
    if (auto sp = extractPriorShared(pv)) return sp;
    if (auto* cell = unwrapPriorCell(pv)) return cell->p;
    if (ev::isFunction(pv)) return std::make_shared<JsPrior>(pv);
    if (!ev::isString(pv)) return nullptr;
    std::string kind = ev::toUtf8(pv);
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

std::shared_ptr<bgm::IEvaluator> parseHeroEvaluator(Value opts) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value v = ev::getProperty(root.get(), "evaluator");
    if (auto se = extractHeroEvaluatorShared(v)) return se;
    if (auto* cell = unwrapEvaluatorCell(v)) return cell->p;
    if (ev::isFunction(v)) return std::make_shared<JsEvaluator>(v);
    if (ev::isString(v) && ev::toUtf8(v) == "hpDelta") {
        return std::make_shared<bgm::HpDeltaEvaluator>();
    }
    return nullptr;
}

std::shared_ptr<bgm::ITeamEvaluator> parseTeamEvaluator(Value opts) {
    if (!ev::isObject(opts)) return nullptr;
    ev::Persistent root(opts);
    Value v = ev::getProperty(root.get(), "evaluator");
    if (auto* cell = unwrapTeamEvaluatorCell(v)) return cell->p;
    if (ev::isFunction(v)) return std::make_shared<JsTeamEvaluator>(v);
    if (!ev::isString(v)) return nullptr;
    std::string kind = ev::toUtf8(v);
    if (kind == "teamHpDelta")   return std::make_shared<bgm::TeamHpDeltaEvaluator>();
    if (kind == "teamAdvantage") return std::make_shared<bgm::TeamAdvantageEvaluator>();
    if (kind == "teamPosition")  return std::make_shared<bgm::TeamPositionEvaluator>();
    return nullptr;
}

std::vector<std::shared_ptr<bgm::Option>> parseOptionArray(Value opts) {
    std::vector<std::shared_ptr<bgm::Option>> out;
    if (!ev::isObject(opts)) return out;
    ev::Persistent root(opts);
    ev::Persistent arr(ev::getProperty(root.get(), "options"));
    uint32_t n = arrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (auto* cell = unwrapOptionCell(ev::getElement(arr.get(), i))) {
            if (cell->opt) out.push_back(cell->opt);
        }
    }
    return out;
}

std::vector<std::shared_ptr<bgm::TeamOption>> parseTeamOptionArray(Value opts) {
    std::vector<std::shared_ptr<bgm::TeamOption>> out;
    if (!ev::isObject(opts)) return out;
    ev::Persistent root(opts);
    ev::Persistent arr(ev::getProperty(root.get(), "options"));
    uint32_t n = arrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (auto* cell = unwrapTeamOptionCell(ev::getElement(arr.get(), i))) {
            if (cell->opt) out.push_back(cell->opt);
        }
    }
    return out;
}

bgm::Commander::AssignFn makeJsAssigner(Value fn) {
    if (!ev::isFunction(fn)) return {};
    auto held = std::make_shared<ev::Persistent>(fn);
    return [held](const std::vector<brogameagent::Agent*>& heroes,
                  const brogameagent::World& world) -> std::vector<int> {
        ev::Persistent hv(buildHeroesView(heroes));
        ev::Persistent wv(buildWorldView(world));
        Value args[2] = { hv.get(), wv.get() };
        auto r = ev::call(held->get(), ev::undefined(), args);
        std::vector<int> out(heroes.size(), 0);
        if (!r.thrown && ev::isObject(r.value)) {
            ev::Persistent res(r.value);
            uint32_t len = arrayLength(res.get());
            uint32_t n = std::min<uint32_t>(len, static_cast<uint32_t>(heroes.size()));
            for (uint32_t i = 0; i < n; ++i) {
                Value el = ev::getElement(res.get(), i);
                out[i] = ev::isNumber(el) ? static_cast<int>(ev::toDouble(el)) : 0;
            }
        }
        return out;
    };
}

std::shared_ptr<bgm::Option> makeJsOption(std::string name, Value canInit,
                                          Value step, Value shouldTerm) {
    if (!ev::isFunction(canInit) || !ev::isFunction(step) || !ev::isFunction(shouldTerm)) {
        return nullptr;
    }
    return std::make_shared<JsOption>(std::move(name), canInit, step, shouldTerm);
}

std::shared_ptr<bgm::TeamOption> makeJsTeamOption(std::string name, Value canInit,
                                                  Value step, Value shouldTerm) {
    if (!ev::isFunction(canInit) || !ev::isFunction(step) || !ev::isFunction(shouldTerm)) {
        return nullptr;
    }
    return std::make_shared<JsTeamOption>(std::move(name), canInit, step, shouldTerm);
}

} // namespace brogameagent::api
