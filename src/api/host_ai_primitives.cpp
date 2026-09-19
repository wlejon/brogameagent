// MCTS primitives as first-class JS objects: the concrete evaluators, priors
// and rollout policies that createMcts/createTeamMcts/... accept in place of
// a preset name.
//
// Before the transition these were ten separate qjsbind classes holding a
// shared_ptr each; the port dropped them for factories returning `{}`, which
// silently disabled every search configured with one. They are restored here
// as ten host classes over four payload cells (one per C++ interface), so
// `AITacticPrior` still carries setMatchWeight/setOtherWeight and the
// extractors can recover the shared_ptr by tag.

#include "host_ai_mcts_shared.h"

namespace brogameagent::api {

HostClass g_hpDeltaEvaluatorClass;
HostClass g_teamHpDeltaEvaluatorClass;
HostClass g_teamAdvantageEvaluatorClass;
HostClass g_teamPositionEvaluatorClass;
HostClass g_randomRolloutClass;
HostClass g_aggressiveRolloutClass;
HostClass g_scriptedRolloutClass;
HostClass g_uniformPriorClass;
HostClass g_attackBiasPriorClass;
HostClass g_tacticPriorClass;

namespace {

template <typename Cell>
Value makeCell(const HostClass& cls, std::shared_ptr<typename decltype(Cell::p)::element_type> p) {
    auto cell = std::make_unique<Cell>();
    cell->p = std::move(p);
    return cls.createInstance(std::move(cell));
}

Value makeTacticPriorValue() {
    auto tp = std::make_shared<bgm::TacticPrior>();
    auto cell = std::make_unique<HostPriorCell>();
    cell->p = tp;
    cell->tacticPrior = tp;
    return g_tacticPriorClass.createInstance(std::move(cell));
}

} // namespace

void ensureAIPrimitiveClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_hpDeltaEvaluatorClass.init("AIHpDeltaEvaluator", [](ObjectBuilder&) {});
    g_teamHpDeltaEvaluatorClass.init("AITeamHpDeltaEvaluator", [](ObjectBuilder&) {});
    g_teamAdvantageEvaluatorClass.init("AITeamAdvantageEvaluator", [](ObjectBuilder&) {});
    g_teamPositionEvaluatorClass.init("AITeamPositionEvaluator", [](ObjectBuilder&) {});
    g_randomRolloutClass.init("AIRandomRollout", [](ObjectBuilder&) {});
    g_aggressiveRolloutClass.init("AIAggressiveRollout", [](ObjectBuilder&) {});
    g_scriptedRolloutClass.init("AIScriptedRollout", [](ObjectBuilder&) {});
    g_uniformPriorClass.init("AIUniformPrior", [](ObjectBuilder&) {});
    g_attackBiasPriorClass.init("AIAttackBiasPrior", [](ObjectBuilder&) {});

    g_tacticPriorClass.init("AITacticPrior", [](ObjectBuilder& b) {
        b.def("setMatchWeight", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* cell = unwrapPriorCell(self);
            if (cell && cell->tacticPrior) {
                cell->tacticPrior->set_match_weight(static_cast<float>(numAt(a, 0)));
            }
            return ev::undefined();
        });
        b.def("setOtherWeight", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* cell = unwrapPriorCell(self);
            if (cell && cell->tacticPrior) {
                cell->tacticPrior->set_other_weight(static_cast<float>(numAt(a, 0)));
            }
            return ev::undefined();
        });
        b.def("setTactic", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* cell = unwrapPriorCell(self);
            if (cell && cell->tacticPrior && !a.empty()) {
                cell->tacticPrior->set_tactic(parseTactic(a[0]));
            }
            return ev::undefined();
        });
    });
}

void installAIPrimitives(ObjectBuilder& game) {
    ensureAIPrimitiveClassesInstalled();

    game.def("createHpDeltaEvaluator", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostEvaluatorCell>(g_hpDeltaEvaluatorClass,
                                           std::make_shared<bgm::HpDeltaEvaluator>());
    });
    game.def("createTeamHpDeltaEvaluator", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostTeamEvaluatorCell>(g_teamHpDeltaEvaluatorClass,
                                               std::make_shared<bgm::TeamHpDeltaEvaluator>());
    });
    game.def("createTeamAdvantageEvaluator", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostTeamEvaluatorCell>(g_teamAdvantageEvaluatorClass,
                                               std::make_shared<bgm::TeamAdvantageEvaluator>());
    });
    game.def("createTeamPositionEvaluator", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostTeamEvaluatorCell>(g_teamPositionEvaluatorClass,
                                               std::make_shared<bgm::TeamPositionEvaluator>());
    });
    game.def("createRandomRollout", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostRolloutCell>(g_randomRolloutClass,
                                         std::make_shared<bgm::RandomRollout>());
    });
    game.def("createAggressiveRollout", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostRolloutCell>(g_aggressiveRolloutClass,
                                         std::make_shared<bgm::AggressiveRollout>());
    });
    game.def("createScriptedRollout", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostRolloutCell>(g_scriptedRolloutClass,
                                         std::make_shared<bgm::ScriptedRollout>());
    });
    game.def("createUniformPrior", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostPriorCell>(g_uniformPriorClass,
                                       std::make_shared<bgm::UniformPrior>());
    });
    game.def("createAttackBiasPrior", 0, [](Value, std::span<const Value>) -> Value {
        return makeCell<HostPriorCell>(g_attackBiasPriorClass,
                                       std::make_shared<bgm::AttackBiasPrior>());
    });
    game.def("createTacticPrior", 0, [](Value, std::span<const Value>) -> Value {
        return makeTacticPriorValue();
    });
}

} // namespace brogameagent::api
