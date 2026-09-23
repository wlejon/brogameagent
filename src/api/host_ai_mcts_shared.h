#pragma once

// Shared MCTS marshalling: CombatAction / Tactic / MctsConfig / SearchStats
// conversions, the agent & world *views* JS callbacks receive, and the
// option/evaluator/prior/rollout parsing every planner factory runs on its
// options object.
//
// These live in their own translation unit because five separate binding
// files (host_ai_mcts.cpp, host_ai_planner.cpp, host_ai_primitives.cpp,
// host_ai_belief.cpp, host_ai_sim.cpp) all need them.

#include "host_ai_internal.h"
#include "host_js_callbacks.h"

#include <brogameagent/belief.h>
#include <brogameagent/mcts.h>
#include <brogameagent/snapshot.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brogameagent::learn {
class IInferenceBackend;
}

namespace brogameagent::api {

namespace bgm = brogameagent::mcts;

// ---------------------------------------------------------------------------
// Additional handle tags
// ---------------------------------------------------------------------------

inline constexpr uint32_t kHostTacticMctsTag     = 0x54434D54u;  // 'TCMT'
inline constexpr uint32_t kHostLayeredPlannerTag = 0x4C594150u;  // 'LYAP'
inline constexpr uint32_t kHostTeamOptionTag     = 0x544F5054u;  // 'TOPT'
inline constexpr uint32_t kHostTeamOptionMctsTag = 0x544F4D43u;  // 'TOMC'
inline constexpr uint32_t kHostCommanderTag      = 0x434D4E44u;  // 'CMND'
inline constexpr uint32_t kHostSimulationTag     = 0x4149534Du;  // 'AISM'
inline constexpr uint32_t kHostRecorderTag       = 0x41495243u;  // 'AIRC'
inline constexpr uint32_t kHostReplayReaderTag   = 0x41495252u;  // 'AIRR'
inline constexpr uint32_t kHostEvaluatorTag      = 0x41494556u;  // 'AIEV'
inline constexpr uint32_t kHostTeamEvaluatorTag  = 0x41495456u;  // 'AITV'
inline constexpr uint32_t kHostRolloutTag        = 0x4149524Cu;  // 'AIRL'
inline constexpr uint32_t kHostPriorTag          = 0x41495052u;  // 'AIPR'

// ---------------------------------------------------------------------------
// Primitive cells — one struct per interface, shared by every concrete class
// of that interface (the class objects differ so `AIUniformPrior` and
// `AITacticPrior` stay distinct names, as they were before the transition).
// ---------------------------------------------------------------------------

struct HostEvaluatorCell {
    uint32_t tag = kHostEvaluatorTag;
    std::shared_ptr<bgm::IEvaluator> p;
};

struct HostTeamEvaluatorCell {
    uint32_t tag = kHostTeamEvaluatorTag;
    std::shared_ptr<bgm::ITeamEvaluator> p;
};

struct HostRolloutCell {
    uint32_t tag = kHostRolloutTag;
    std::shared_ptr<bgm::IRolloutPolicy> p;
};

struct HostPriorCell {
    uint32_t tag = kHostPriorTag;
    std::shared_ptr<bgm::IPrior> p;
    // Non-null only for AITacticPrior, whose setMatchWeight/setOtherWeight
    // need the concrete type.
    std::shared_ptr<bgm::TacticPrior> tacticPrior;
};

// A JS-authored option's three callbacks sit on its handle as `_callbacks`;
// `slots` are the adapter's, which a search taking the option adopts.
struct HostOptionCell {
    uint32_t tag = kHostOptionTag;
    std::shared_ptr<bgm::Option> opt;
    std::vector<JsSlotPtr> slots;
};

struct HostTeamOptionCell {
    uint32_t tag = kHostTeamOptionTag;
    std::shared_ptr<bgm::TeamOption> opt;
    std::vector<JsSlotPtr> slots;
};

inline HostEvaluatorCell* unwrapEvaluatorCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostEvaluatorCell*>(ev::handleData(v));
    return (h && h->tag == kHostEvaluatorTag) ? h : nullptr;
}

inline HostTeamEvaluatorCell* unwrapTeamEvaluatorCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamEvaluatorCell*>(ev::handleData(v));
    return (h && h->tag == kHostTeamEvaluatorTag) ? h : nullptr;
}

inline HostRolloutCell* unwrapRolloutCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostRolloutCell*>(ev::handleData(v));
    return (h && h->tag == kHostRolloutTag) ? h : nullptr;
}

inline HostPriorCell* unwrapPriorCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostPriorCell*>(ev::handleData(v));
    return (h && h->tag == kHostPriorTag) ? h : nullptr;
}

inline HostOptionCell* unwrapOptionCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostOptionCell*>(ev::handleData(v));
    return (h && h->tag == kHostOptionTag) ? h : nullptr;
}

// Snapshot cells live here rather than inside host_ai_extras.cpp because
// patchSnapshotWithParticles (belief) has to reach a WorldSnapshot too.
struct HostAgentSnapshot {
    uint32_t tag = kHostAgentSnapshotTag;
    brogameagent::AgentSnapshot s;
};

struct HostWorldSnapshot {
    uint32_t tag = kHostWorldSnapshotTag;
    brogameagent::WorldSnapshot s;
};

inline HostAgentSnapshot* unwrapAgentSnapshot(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostAgentSnapshot*>(ev::handleData(v));
    return (h && h->tag == kHostAgentSnapshotTag) ? h : nullptr;
}

inline HostWorldSnapshot* unwrapWorldSnapshot(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostWorldSnapshot*>(ev::handleData(v));
    return (h && h->tag == kHostWorldSnapshotTag) ? h : nullptr;
}

inline HostTeamOptionCell* unwrapTeamOptionCell(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostTeamOptionCell*>(ev::handleData(v));
    return (h && h->tag == kHostTeamOptionTag) ? h : nullptr;
}

// ---------------------------------------------------------------------------
// Host classes owned by the MCTS/planner/primitive files
// ---------------------------------------------------------------------------

extern HostClass g_tacticMctsClass;
extern HostClass g_layeredPlannerClass;
extern HostClass g_teamOptionClass;
extern HostClass g_teamOptionMctsClass;
extern HostClass g_commanderClass;
extern HostClass g_simulationClass;
extern HostClass g_recorderClass;
extern HostClass g_replayReaderClass;

extern HostClass g_hpDeltaEvaluatorClass;
extern HostClass g_teamHpDeltaEvaluatorClass;
extern HostClass g_teamAdvantageEvaluatorClass;
extern HostClass g_teamPositionEvaluatorClass;
extern HostClass g_randomRolloutClass;
extern HostClass g_aggressiveRolloutClass;
extern HostClass g_scriptedRolloutClass;
extern HostClass g_uniformPriorClass;
extern HostClass g_attackBiasPriorClass;
extern HostClass g_tacticPriorClass;

// ---------------------------------------------------------------------------
// Marshalling (host_ai_mcts_support.cpp)
// ---------------------------------------------------------------------------

Value makeCombatAction(const bgm::CombatAction& a);
bgm::CombatAction parseCombatAction(Value v);
std::vector<bgm::CombatAction> parseCombatActionArray(Value arr);
Value makeCombatActionArray(const std::vector<bgm::CombatAction>& acts);

Value makeTactic(const bgm::Tactic& t);
bgm::Tactic parseTactic(Value v);

/// The keys `opts` sets, over `base` (the defaults, or a search's current
/// config for setConfig). Integer keys are range-checked (RangeError).
bgm::MctsConfig parseMctsConfig(Value opts, const bgm::MctsConfig& base = {});
Value makeSearchStats(const bgm::SearchStats& s);

std::vector<brogameagent::Agent*> parseHeroes(Value arr);

/// A plain-object view of one agent, handed to JS rollout/prior/evaluator
/// and option callbacks (they have no route back to the Agent wrapper).
Value buildAgentFields(const brogameagent::Agent& a);
Value buildWorldView(const brogameagent::World& world);
Value buildHeroesView(const std::vector<brogameagent::Agent*>& heroes);

std::string readStringProp(Value obj, const char* key);

/// The classic hero Mcts behind an AIMcts handle, or null. The host cell
/// itself is private to host_ai_mcts.cpp; the learn bindings need the search
/// tree (`last_root()`) for targetsFromMcts / makeSituation /
/// gumbelImprovedPolicy, so this is the seam.
bgm::Mcts* classicMctsFromValue(Value v);

// ---------------------------------------------------------------------------
// Option parsing
//
// A JS function among the options becomes an adapter over a slot added to
// `cbs`; the factory attaches `cbs` to the handle it makes and opens a
// CallbackScope in every method that runs the search (host_js_callbacks.h).
// ---------------------------------------------------------------------------

/// opts.rolloutPolicy: a rollout handle, a JS function, or one of
/// "random" | "aggressive" | "scripted". Null when absent/unrecognised.
std::shared_ptr<bgm::IRolloutPolicy> parseRolloutPolicy(Value opts, JsCallbackSet& cbs);

/// opts.opponentPolicy: "idle" | "aggressive" | "scripted". Empty when absent.
bgm::OpponentPolicy parseOpponentPolicy(Value opts);

/// opts.prior: a prior handle, a JS function, or "uniform" | "attackBias" |
/// "tacticMatch" (which also reads opts.tactic / tacticMatchWeight /
/// tacticOtherWeight).
std::shared_ptr<bgm::IPrior> parsePrior(Value opts, JsCallbackSet& cbs);

/// opts.evaluator: an evaluator handle, a JS function, or "hpDelta".
std::shared_ptr<bgm::IEvaluator> parseHeroEvaluator(Value opts, JsCallbackSet& cbs);

/// opts.evaluator: a team-evaluator handle, a JS function, or one of
/// "teamHpDelta" | "teamAdvantage" | "teamPosition".
std::shared_ptr<bgm::ITeamEvaluator> parseTeamEvaluator(Value opts, JsCallbackSet& cbs);

/// opts.options: array of AIOption / AITeamOption handles; their callbacks
/// are adopted into `cbs`.
std::vector<std::shared_ptr<bgm::Option>> parseOptionArray(Value opts, JsCallbackSet& cbs);
std::vector<std::shared_ptr<bgm::TeamOption>> parseTeamOptionArray(Value opts,
                                                                  JsCallbackSet& cbs);

/// A JS `assign(heroesView, worldView) -> number[]` wrapped as a Commander
/// role assigner. Returns an empty function when `fn` is not callable.
bgm::Commander::AssignFn makeJsAssigner(Value fn, JsCallbackSet& cbs);

/// JS-authored Option / TeamOption from { name, canInitiate, step,
/// shouldTerminate }. Returns null when a callback is missing.
std::shared_ptr<bgm::Option> makeJsOption(std::string name, Value canInit,
                                          Value step, Value shouldTerm, JsCallbackSet& cbs);
std::shared_ptr<bgm::TeamOption> makeJsTeamOption(std::string name, Value canInit,
                                                  Value step, Value shouldTerm,
                                                  JsCallbackSet& cbs);

/// Roots a world handle and makes it the ability-dispatch receiver for the
/// scope (ActiveWorldScope), so a search that ticks it — or a clone of it —
/// can reach its registerAbility callbacks. A non-world `v` is a no-op.
class WorldArgScope {
public:
    explicit WorldArgScope(Value v);
    WorldArgScope(const WorldArgScope&) = delete;
    WorldArgScope& operator=(const WorldArgScope&) = delete;

private:
    ev::Persistent world_;
    std::unique_ptr<ActiveWorldScope> scope_;
};

/// What a planner method that runs its search opens: the owner's callbacks
/// bound (CallbackScope) and the world argument made the ability receiver.
/// Both Values must be current; the world is rooted before anything
/// allocates.
class SearchScope {
public:
    SearchScope(Value self, const std::vector<JsSlotPtr>& slots, Value world)
        : active_(world), callbacks_(self, slots) {}

private:
    WorldArgScope active_;
    CallbackScope callbacks_;
};

// ---------------------------------------------------------------------------
// Installers
// ---------------------------------------------------------------------------

void ensureAIPrimitiveClassesInstalled();
void installAIPrimitives(ObjectBuilder& game);

void ensureAIPlannerClassesInstalled();
void installAIPlanner(ObjectBuilder& game);

void ensureAISimClassesInstalled();
void installAISim(ObjectBuilder& game);

void ensureAIBeliefClassesInstalled();
void installAIBelief(ObjectBuilder& game);

/// bro.ai.game.nn — installed only when the neural layer is compiled in.
void installAINeural(ObjectBuilder& game);
/// bro.ai.game.learn
void installAILearn(ObjectBuilder& game);
/// bro.ai.game.grid
void installAIGrid(ObjectBuilder& game);

// A DirectBackend / ServerBackend handle, or null. Defined in
// host_ai_learn.cpp; answers null in a build without the neural layer.
brogameagent::learn::IInferenceBackend* inferenceBackendFromJS(Value v);

/// Masked softmax over the backend's logits / its raw value — the native
/// prior & value fast path createGenericMcts's `backend` option installs.
bgm::GenericPriorFn makeNativePriorFn(brogameagent::learn::IInferenceBackend* backend);
bgm::GenericValueFn makeNativeValueFn(brogameagent::learn::IInferenceBackend* backend);

// Learn-side extractors, consulted by parsePrior/parseHeroEvaluator so a
// NeuralPrior / NeuralEvaluator can be passed wherever a preset name can.
// Always defined; they answer null in a build without the neural layer.
std::shared_ptr<bgm::IPrior> extractPriorShared(Value v);
std::shared_ptr<bgm::IEvaluator> extractHeroEvaluatorShared(Value v);

} // namespace brogameagent::api
