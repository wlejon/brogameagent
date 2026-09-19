// bro.ai.game.learn — the hero-shaped half: ReplayBuffer, NeuralEvaluator,
// NeuralPrior, GumbelNoisePrior, ExItTrainer, the search-trace free
// functions, and the extractors the MCTS factories consult so a neural prior
// or evaluator can be passed wherever a preset name can.
//
// Ported from ai_learn_bindings.cpp.

#include "host_ai_learn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <exception>
#include <random>

namespace brogameagent::api {

HostClass g_replayBufferClass;
HostClass g_neuralEvaluatorClass;
HostClass g_neuralPriorClass;
HostClass g_gumbelPriorClass;
HostClass g_exitTrainerClass;

namespace {

/// Copy a Float32Array property into a fixed-size destination. Returns true
/// only when the source covered the whole destination, matching the old
/// copyFloatProp.
bool copyFloatProp(Value obj, const char* key, float* dst, int count) {
    if (!ev::isObject(obj)) return false;
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    size_t have = 0;
    float* src = floatPtr(v, have);
    if (!src) return false;
    int copy = static_cast<int>(have) < count ? static_cast<int>(have) : count;
    std::memcpy(dst, src, static_cast<size_t>(copy) * sizeof(float));
    return copy == count;
}

Value makeTargetsObject(const float* move, int nMove, const float* attack, int nAttack,
                        const float* ability, int nAbility) {
    ObjectBuilder o;
    o.set("move", makeFloat32Array(move, static_cast<size_t>(nMove)));
    o.set("attack", makeFloat32Array(attack, static_cast<size_t>(nAttack)));
    o.set("ability", makeFloat32Array(ability, static_cast<size_t>(nAbility)));
    return o.get();
}

Value makeTrainStepObject(float lossValue, float lossPolicy, float lossTotal, int samples) {
    ObjectBuilder o;
    o.set("lossValue", static_cast<double>(lossValue));
    o.set("lossPolicy", static_cast<double>(lossPolicy));
    o.set("lossTotal", static_cast<double>(lossTotal));
    o.set("samples", static_cast<double>(samples));
    return o.get();
}

std::mt19937_64& sampleRng() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

// ── ReplayBuffer ──────────────────────────────────────────────────────────

void decorateReplayBuffer(ObjectBuilder& b) {
    b.accessor("size", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapReplayBuffer(self);
        return ev::fromDouble(d && d->buf ? static_cast<double>(d->buf->size()) : 0.0);
    });
    b.accessor("capacity", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapReplayBuffer(self);
        return ev::fromDouble(d && d->buf ? static_cast<double>(d->buf->capacity()) : 0.0);
    });
    b.def("push", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapReplayBuffer(self);
        if (!d || !d->buf || a.empty()) return ev::undefined();
        learn::Situation s{};
        if (!readSituationValue(a[0], s)) {
            return ev::throwTypeError("push(situation): expected object");
        }
        d->buf->push(s);
        return ev::undefined();
    });
    b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapReplayBuffer(self);
        if (d && d->buf) d->buf->clear();
        return ev::undefined();
    });
    b.def("sample", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapReplayBuffer(self);
        if (!d || !d->buf) return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        int n = a.empty() ? 0 : i32At(a, 0);
        if (n < 0) n = 0;
        auto batch = d->buf->sample(static_cast<size_t>(n), sampleRng());
        return hostArrayOf(batch.size(), [&](size_t i) { return makeSituationObject(batch[i]); });
    });
    b.def("all", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapReplayBuffer(self);
        if (!d || !d->buf) return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        const auto& v = d->buf->all();
        return hostArrayOf(v.size(), [&](size_t i) { return makeSituationObject(v[i]); });
    });
}

// ── NeuralEvaluator / NeuralPrior / GumbelNoisePrior ──────────────────────

void decorateNeuralEvaluator(ObjectBuilder& b) {
    b.def("evaluate", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapNeuralEvaluator(self);
        auto* w = unwrapWorld(argAt(a, 0));
        if (!d || !d->eval || !w) return ev::fromDouble(0.0);
        return ev::fromDouble(d->eval->evaluate(w->world, i32At(a, 1)));
    });
}

void decorateNeuralPrior(ObjectBuilder& b) {
    b.def("setTemperature", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapNeuralPrior(self);
        if (d && d->prior) d->prior->set_temperature(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });
    b.def("setUniformMix", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapNeuralPrior(self);
        if (d && d->prior) d->prior->set_uniform_mix(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });
}

void decorateGumbelPrior(ObjectBuilder& b) {
    b.def("reseed", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGumbelPrior(self);
        if (!d || !d->prior || a.empty()) return ev::undefined();
        d->prior->reseed(readSeedArg(a[0], 0));
        return ev::undefined();
    });
    b.def("setScale", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGumbelPrior(self);
        if (d && d->prior) d->prior->set_scale(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });
}

// ── ExItTrainer ───────────────────────────────────────────────────────────

void decorateExItTrainer(ObjectBuilder& b) {
    b.accessor("totalSteps", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapExItTrainer(self);
        return ev::fromDouble(d && d->trainer ? d->trainer->total_steps() : 0);
    });
    b.accessor("totalPublishes", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapExItTrainer(self);
        return ev::fromDouble(d && d->trainer ? d->trainer->total_publishes() : 0);
    });
    b.def("setNet", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapExItTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        auto net = singleHeroNetShared(a[0]);
        if (!net) return ev::throwTypeError("setNet(net): expected SingleHeroNet");
        d->netRef = net;
        d->trainer->set_net(net.get());
        return ev::undefined();
    });
    b.def("setBuffer", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapExItTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        auto* bd = unwrapReplayBuffer(a[0]);
        if (!bd || !bd->buf) return ev::throwTypeError("setBuffer(buf): expected ReplayBuffer");
        d->bufRef = bd->buf;
        d->trainer->set_buffer(bd->buf.get());
        return ev::undefined();
    });
    b.def("setWeightsHandle", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapExItTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        auto h = weightsHandleShared(a[0]);
        if (!h) return ev::throwTypeError("setWeightsHandle(handle): expected WeightsHandle");
        d->handleRef = h;
        d->trainer->set_weights_handle(h.get());
        return ev::undefined();
    });
    b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapExItTrainer(self);
        Value cfgV = argAt(a, 0);
        if (!d || !d->trainer || !ev::isObject(cfgV)) return ev::undefined();
        ev::Persistent root(cfgV);
        learn::TrainerConfig c = d->trainer->config();
        c.lr = static_cast<float>(getDoubleProperty(root.get(), "lr", c.lr));
        c.momentum = static_cast<float>(getDoubleProperty(root.get(), "momentum", c.momentum));
        c.batch = getIntProp(root.get(), "batch", c.batch);
        c.policy_weight =
            static_cast<float>(getDoubleProperty(root.get(), "policyWeight", c.policy_weight));
        c.value_weight =
            static_cast<float>(getDoubleProperty(root.get(), "valueWeight", c.value_weight));
        c.publish_every = getIntProp(root.get(), "publishEvery", c.publish_every);
        c.rng_seed = readSeedArg(ev::getProperty(root.get(), "rngSeed"), c.rng_seed);
        d->trainer->set_config(c);
        return ev::undefined();
    });
    b.def("step", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapExItTrainer(self);
        if (!d || !d->trainer) return ObjectBuilder{}.get();
        learn::TrainStep s;
        try {
            s = d->trainer->step();
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeTrainStepObject(s.loss_value, s.loss_policy, s.loss_total, s.samples);
    });
    b.def("stepN", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapExItTrainer(self);
        if (!d || !d->trainer) return ObjectBuilder{}.get();
        learn::TrainStep s;
        try {
            s = d->trainer->step_n(i32At(a, 0));
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeTrainStepObject(s.loss_value, s.loss_policy, s.loss_total, s.samples);
    });
}

} // namespace

// ---------------------------------------------------------------------------
// Situation marshalling (shared with host_ai_learn_generic.cpp)
// ---------------------------------------------------------------------------

Value makeSituationObject(const learn::Situation& s) {
    ObjectBuilder o;
    o.set("obs", makeFloat32Array(s.obs.data(), s.obs.size()));
    o.set("atkMask", makeFloat32Array(s.atk_mask.data(), s.atk_mask.size()));
    o.set("abilMask", makeFloat32Array(s.abil_mask.data(), s.abil_mask.size()));
    o.set("targetMove", makeFloat32Array(s.target_move.data(), s.target_move.size()));
    o.set("targetAttack", makeFloat32Array(s.target_attack.data(), s.target_attack.size()));
    o.set("targetAbility", makeFloat32Array(s.target_ability.data(), s.target_ability.size()));
    o.set("valueTarget", static_cast<double>(s.value_target));
    return o.get();
}

bool readSituationValue(Value v, learn::Situation& out) {
    if (!ev::isObject(v)) return false;
    ev::Persistent root(v);
    copyFloatProp(root.get(), "obs", out.obs.data(), static_cast<int>(out.obs.size()));
    copyFloatProp(root.get(), "atkMask", out.atk_mask.data(), static_cast<int>(out.atk_mask.size()));
    copyFloatProp(root.get(), "abilMask", out.abil_mask.data(),
                  static_cast<int>(out.abil_mask.size()));
    copyFloatProp(root.get(), "targetMove", out.target_move.data(),
                  static_cast<int>(out.target_move.size()));
    copyFloatProp(root.get(), "targetAttack", out.target_attack.data(),
                  static_cast<int>(out.target_attack.size()));
    copyFloatProp(root.get(), "targetAbility", out.target_ability.data(),
                  static_cast<int>(out.target_ability.size()));
    out.value_target = static_cast<float>(getDoubleProperty(root.get(), "valueTarget", 0.0));
    return true;
}

// ---------------------------------------------------------------------------
// Extractors consulted by the MCTS factories
// ---------------------------------------------------------------------------

std::shared_ptr<bgm::IPrior> extractPriorShared(Value v) {
    if (auto* np = unwrapNeuralPrior(v)) return np->prior;
    if (auto* gp = unwrapGumbelPrior(v)) return gp->prior;
    return {};
}

std::shared_ptr<bgm::IEvaluator> extractHeroEvaluatorShared(Value v) {
    if (auto* ne = unwrapNeuralEvaluator(v)) return ne->eval;
    return {};
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

void ensureAILearnCoreClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_replayBufferClass.init("AIReplayBuffer", decorateReplayBuffer);
    g_neuralEvaluatorClass.init("AINeuralEvaluator", decorateNeuralEvaluator);
    g_neuralPriorClass.init("AINeuralPrior", decorateNeuralPrior);
    g_gumbelPriorClass.init("AIGumbelNoisePrior", decorateGumbelPrior);
    g_exitTrainerClass.init("AIExItTrainer", decorateExItTrainer);
}

void installAILearnCore(ObjectBuilder& learnNs) {
    ensureAILearnCoreClassesInstalled();

    learnNs.def("createReplayBuffer", 1, [](Value, std::span<const Value> a) -> Value {
        size_t cap = 4096;
        if (!a.empty()) {
            int v = i32At(a, 0);
            if (v > 0) cap = static_cast<size_t>(v);
        }
        auto cell = std::make_unique<HostReplayBuffer>();
        cell->buf = std::make_shared<learn::ReplayBuffer>(cap);
        return g_replayBufferClass.createInstance(std::move(cell));
    });

    learnNs.def("createNeuralEvaluator", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createNeuralEvaluator(net, handle?)");
        auto net = singleHeroNetShared(a[0]);
        if (!net) return ev::throwTypeError("net must be a SingleHeroNet");
        std::shared_ptr<nn::WeightsHandle> handle;
        if (a.size() >= 2 && !ev::isUndefined(a[1]) && !ev::isNull(a[1])) {
            handle = weightsHandleShared(a[1]);
        }
        auto cell = std::make_unique<HostNeuralEvaluator>();
        cell->eval = std::make_shared<learn::NeuralEvaluator>(net, handle.get());
        cell->netRef = net;
        cell->handleRef = handle;
        return g_neuralEvaluatorClass.createInstance(std::move(cell));
    });

    learnNs.def("createNeuralPrior", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createNeuralPrior(net, handle?)");
        auto net = singleHeroNetShared(a[0]);
        if (!net) return ev::throwTypeError("net must be a SingleHeroNet");
        std::shared_ptr<nn::WeightsHandle> handle;
        if (a.size() >= 2 && !ev::isUndefined(a[1]) && !ev::isNull(a[1])) {
            handle = weightsHandleShared(a[1]);
        }
        auto cell = std::make_unique<HostNeuralPrior>();
        cell->prior = std::make_shared<learn::NeuralPrior>(net, handle.get());
        cell->netRef = net;
        cell->handleRef = handle;
        return g_neuralPriorClass.createInstance(std::move(cell));
    });

    learnNs.def("createGumbelNoisePrior", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createGumbelNoisePrior(innerPrior, scale?)");
        auto inner = extractPriorShared(a[0]);
        if (!inner) {
            // The old binding took only a NeuralPrior / GumbelNoisePrior here;
            // a plain AIUniformPrior-style cell is just as valid an inner.
            if (auto* cell = unwrapPriorCell(a[0])) inner = cell->p;
        }
        if (!inner) return ev::throwTypeError("innerPrior must be a bound prior");
        float scale = a.size() >= 2 ? static_cast<float>(numAt(a, 1)) : 1.0f;
        auto cell = std::make_unique<HostGumbelPrior>();
        cell->prior = std::make_shared<learn::GumbelNoisePrior>(inner, scale);
        cell->innerRef = inner;
        return g_gumbelPriorClass.createInstance(std::move(cell));
    });

    learnNs.def("createExItTrainer", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostExItTrainer>();
        cell->trainer = std::make_unique<learn::ExItTrainer>();
        return g_exitTrainerClass.createInstance(std::move(cell));
    });

    learnNs.def("targetsFromMcts", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("targetsFromMcts(mcts)");
        auto* m = classicMctsFromValue(a[0]);
        if (!m) return ev::throwTypeError("expected Mcts");
        const bgm::Node* root = m->last_root();
        if (!root) return ev::null();
        float tm[nn::FactoredPolicyHead::N_MOVE] = {};
        float ta[nn::FactoredPolicyHead::N_ATTACK] = {};
        float tb[nn::FactoredPolicyHead::N_ABILITY] = {};
        learn::targets_from_root(*root, tm, ta, tb);
        return makeTargetsObject(tm, nn::FactoredPolicyHead::N_MOVE,
                                 ta, nn::FactoredPolicyHead::N_ATTACK,
                                 tb, nn::FactoredPolicyHead::N_ABILITY);
    });

    learnNs.def("makeSituation", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::throwTypeError("makeSituation(mcts, hero, world)");
        auto* m = classicMctsFromValue(a[0]);
        auto* hero = unwrapAgent(a[1]);
        auto* w = unwrapWorld(a[2]);
        if (!m || !hero || !w) return ev::throwTypeError("bad args");
        const bgm::Node* root = m->last_root();
        if (!root) return ev::null();
        return makeSituationObject(learn::make_situation(w->world, hero->agent, *root));
    });

    learnNs.def("gumbelImprovedPolicy", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("gumbelImprovedPolicy(mcts)");
        auto* m = classicMctsFromValue(a[0]);
        if (!m) return ev::throwTypeError("expected Mcts");
        const bgm::Node* root = m->last_root();
        if (!root) return ev::null();
        // gumbel_improved_policy spells its bounds as literals (9/6/9); the
        // head constants are those same numbers, and this keeps them so.
        static_assert(nn::FactoredPolicyHead::N_MOVE == 9);
        static_assert(nn::FactoredPolicyHead::N_ATTACK == 6);
        static_assert(nn::FactoredPolicyHead::N_ABILITY == 9);
        float tm[nn::FactoredPolicyHead::N_MOVE] = {};
        float ta[nn::FactoredPolicyHead::N_ATTACK] = {};
        float tb[nn::FactoredPolicyHead::N_ABILITY] = {};
        learn::gumbel_improved_policy(*root, tm, ta, tb);
        return makeTargetsObject(tm, nn::FactoredPolicyHead::N_MOVE,
                                 ta, nn::FactoredPolicyHead::N_ATTACK,
                                 tb, nn::FactoredPolicyHead::N_ABILITY);
    });
}

void installAILearn(ObjectBuilder& game) {
    ObjectBuilder learnNs;
    installAILearnCore(learnNs);
    installAILearnGeneric(learnNs);
    game.set("learn", learnNs.get());
}

} // namespace brogameagent::api

#else  // !BROGAMEAGENT_HAS_NN

#include "host_ai_mcts_shared.h"

namespace brogameagent::api {

// Without the neural layer there is no neural prior or evaluator to find,
// so the MCTS factories' extractors answer empty and the namespace reports
// itself unavailable rather than silently missing.
std::shared_ptr<bgm::IPrior> extractPriorShared(Value) { return {}; }
std::shared_ptr<bgm::IEvaluator> extractHeroEvaluatorShared(Value) { return {}; }

void installAILearn(ObjectBuilder& game) {
    ObjectBuilder learnNs;
    learnNs.set("available", false);
    game.set("learn", learnNs.get());
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
