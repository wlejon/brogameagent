// bro.ai.game.learn — the observation-agnostic half: GenericReplayBuffer,
// GenericExItTrainer, the batched inference server, and the two
// IInferenceBackend wrappers that are GenericMcts's native prior/value fast
// path.
//
// Ported from ai_learn_bindings.cpp; makeNativePriorFn / makeNativeValueFn
// come from ai_generic_mcts_bindings.cpp, where the `backend` option wired
// them in.

#include "host_ai_learn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <cmath>
#include <exception>
#include <future>
#include <random>

namespace brogameagent::api {

HostClass g_genericBufferClass;
HostClass g_genericTrainerClass;
HostClass g_inferenceServerClass;
HostClass g_directBackendClass;
HostClass g_serverBackendClass;

namespace {

std::mt19937_64& genericRng() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

bool readFloatVecProp(Value obj, const char* key, std::vector<float>& out, bool required) {
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        out.clear();
        return !required;
    }
    size_t n = 0;
    float* p = floatPtr(v, n);
    if (!p) return !required;
    out.assign(p, p + n);
    return true;
}

Value makeTrainStepObject(float lossValue, float lossPolicy, float lossTotal, int samples) {
    ObjectBuilder o;
    o.set("lossValue", static_cast<double>(lossValue));
    o.set("lossPolicy", static_cast<double>(lossPolicy));
    o.set("lossTotal", static_cast<double>(lossTotal));
    o.set("samples", static_cast<double>(samples));
    return o.get();
}

Value makeEvalResultObject(const std::vector<float>& logits, float value) {
    ObjectBuilder o;
    o.set("logits", makeFloat32Array(logits.data(), logits.size()));
    o.set("value", static_cast<double>(value));
    return o.get();
}

// ── GenericReplayBuffer ───────────────────────────────────────────────────

void decorateGenericBuffer(ObjectBuilder& b) {
    b.accessor("size", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericBuffer(self);
        return ev::fromDouble(d && d->buf ? static_cast<double>(d->buf->size()) : 0.0);
    });
    b.accessor("capacity", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericBuffer(self);
        return ev::fromDouble(d && d->buf ? static_cast<double>(d->buf->capacity()) : 0.0);
    });
    b.def("push", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericBuffer(self);
        if (!d || !d->buf || a.empty()) return ev::undefined();
        learn::GenericSituation s{};
        if (!readGenericSituationValue(a[0], s)) {
            return ev::throwTypeError(
                "push(situation): expected {obs, policyTarget, valueTarget, actionMask?}");
        }
        d->buf->push(std::move(s));
        return ev::undefined();
    });
    b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericBuffer(self);
        if (d && d->buf) d->buf->clear();
        return ev::undefined();
    });
    b.def("sample", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericBuffer(self);
        if (!d || !d->buf) return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        int n = a.empty() ? 0 : i32At(a, 0);
        if (n < 0) n = 0;
        auto batch = d->buf->sample(static_cast<size_t>(n), genericRng());
        return hostArrayOf(batch.size(),
                           [&](size_t i) { return makeGenericSituationObject(batch[i]); });
    });
    b.def("all", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericBuffer(self);
        if (!d || !d->buf) return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        const auto& v = d->buf->all();
        return hostArrayOf(v.size(), [&](size_t i) { return makeGenericSituationObject(v[i]); });
    });
}

// ── GenericExItTrainer ────────────────────────────────────────────────────

void decorateGenericTrainer(ObjectBuilder& b) {
    b.accessor("totalSteps", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericTrainer(self);
        return ev::fromDouble(d && d->trainer ? d->trainer->total_steps() : 0);
    });
    b.accessor("totalPublishes", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericTrainer(self);
        return ev::fromDouble(d && d->trainer ? d->trainer->total_publishes() : 0);
    });
    b.def("setNet", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        if (auto pvn = policyValueNetShared(a[0])) {
            d->pvnRef = pvn;
            d->txRef.reset();
            d->trainer->set_net(pvn.get());
        } else if (auto tx = heroNetTxShared(a[0])) {
            d->txRef = tx;
            d->pvnRef.reset();
            d->trainer->set_net(tx.get());
        } else {
            return ev::throwTypeError("setNet(net): expected PolicyValueNet or SingleHeroNetTX");
        }
        return ev::undefined();
    });
    b.def("setBuffer", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        auto* bd = unwrapGenericBuffer(a[0]);
        if (!bd || !bd->buf) {
            return ev::throwTypeError("setBuffer(buf): expected GenericReplayBuffer");
        }
        d->bufRef = bd->buf;
        d->trainer->set_buffer(bd->buf.get());
        return ev::undefined();
    });
    b.def("setWeightsHandle", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericTrainer(self);
        if (!d || !d->trainer || a.empty()) return ev::undefined();
        auto h = weightsHandleShared(a[0]);
        if (!h) return ev::throwTypeError("setWeightsHandle(handle): expected WeightsHandle");
        d->handleRef = h;
        d->trainer->set_weights_handle(h.get());
        return ev::undefined();
    });
    b.def("setConfig", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericTrainer(self);
        Value cfgV = argAt(a, 0);
        if (!d || !d->trainer || !ev::isObject(cfgV)) return ev::undefined();
        ev::Persistent root(cfgV);
        learn::GenericTrainerConfig c = d->trainer->config();
        c.lr = static_cast<float>(getDoubleProperty(root.get(), "lr", c.lr));
        c.momentum = static_cast<float>(getDoubleProperty(root.get(), "momentum", c.momentum));
        c.batch = getCountProp(root.get(), "batch", c.batch);
        c.policy_weight =
            static_cast<float>(getDoubleProperty(root.get(), "policyWeight", c.policy_weight));
        c.value_weight =
            static_cast<float>(getDoubleProperty(root.get(), "valueWeight", c.value_weight));
        c.publish_every = getCountProp(root.get(), "publishEvery", c.publish_every);
        c.rng_seed = readSeedArg(ev::getProperty(root.get(), "rngSeed"), c.rng_seed);

        // Where compute happens. "gpu" requires net.to('gpu') first. Unlike
        // the net's own to(), the old binding did not throw here — an absent
        // GPU left the trainer on the CPU device it already had.
        Value devV = ev::getProperty(root.get(), "device");
        if (ev::isString(devV)) {
            brotensor::Device target = brotensor::Device::CPU;
            std::string error;
            if (resolveDevice(ev::toUtf8(devV), target, error)) {
                c.device = target;
            } else {
                c.device = brotensor::Device::CPU;
            }
        }
        d->trainer->set_config(c);
        return ev::undefined();
    });
    b.def("step", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGenericTrainer(self);
        if (!d || !d->trainer) return ObjectBuilder{}.get();
        learn::GenericTrainStep s;
        try {
            s = d->trainer->step();
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeTrainStepObject(s.loss_value, s.loss_policy, s.loss_total, s.samples);
    });
    b.def("stepN", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGenericTrainer(self);
        if (!d || !d->trainer) return ObjectBuilder{}.get();
        learn::GenericTrainStep s;
        // Outside the try: its RangeError must not become a plain Error.
        const int n = static_cast<int>(intAt(a, 0, 0, INT32_MAX, "stepN: n"));
        try {
            s = d->trainer->step_n(n);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeTrainStepObject(s.loss_value, s.loss_policy, s.loss_total, s.samples);
    });
}

// ── BatchedInferenceServer ────────────────────────────────────────────────

void decorateInferenceServer(ObjectBuilder& b) {
    b.accessor("batchesRun", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapInferenceServer(self);
        return ev::fromDouble(d && d->server ? d->server->batches_run() : 0);
    });
    b.def("evaluate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapInferenceServer(self);
        if (!d || !d->server || a.empty()) return ev::throwTypeError("evaluate(obsF32)");
        size_t n = 0;
        float* obsPtr = floatPtr(a[0], n);
        if (!obsPtr) return ev::throwTypeError("evaluate: obs must be a Float32Array");
        std::vector<float> obs(obsPtr, obsPtr + n);
        learn::BatchedInferenceServer::EvalResult r;
        try {
            r = d->server->evaluate(obs);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeEvalResultObject(r.logits, r.value);
    });
    b.def("evaluateBatch", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapInferenceServer(self);
        if (!d || !d->server || a.empty() || !ev::isObject(a[0])) {
            return ev::throwTypeError("evaluateBatch(obsArray)");
        }
        ev::Persistent arr(a[0]);
        Value lenV = ev::getProperty(arr.get(), "length");
        if (ev::isUndefined(lenV) || ev::isObject(lenV)) {
            return ev::throwTypeError("evaluateBatch(obsArray)");
        }
        const uint32_t len = toLength(lenV);

        // Fan out through evaluate_async so concurrent rows coalesce into one
        // batch on the server's worker thread, then wait on each future.
        std::vector<std::future<learn::BatchedInferenceServer::EvalResult>> futures;
        futures.reserve(reserveHint(len));
        for (uint32_t i = 0; i < len; ++i) {
            Value rowV = ev::getElement(arr.get(), i);
            size_t n = 0;
            float* obsPtr = floatPtr(rowV, n);
            if (!obsPtr) return ev::throwTypeError("evaluateBatch: each row must be a Float32Array");
            // evaluate_async throws on a wrong-length row or a stopping
            // server; no C++ exception may cross back into compiled JS.
            try {
                futures.push_back(d->server->evaluate_async(std::vector<float>(obsPtr, obsPtr + n)));
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        }

        std::vector<std::vector<float>> logits(futures.size());
        std::vector<float> values(futures.size(), 0.0f);
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                auto r = futures[i].get();
                logits[i] = std::move(r.logits);
                values[i] = r.value;
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        }
        return hostArrayOf(futures.size(),
                           [&](size_t i) { return makeEvalResultObject(logits[i], values[i]); });
    });
    b.def("shutdown", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapInferenceServer(self);
        if (d) d->server.reset();
        return ev::undefined();
    });
}

// ── DirectBackend / ServerBackend ─────────────────────────────────────────

template <typename Host, Host* (*Unwrap)(Value)>
void decorateBackend(ObjectBuilder& b) {
    b.accessor("numActions", [](Value self, std::span<const Value>) -> Value {
        auto* d = Unwrap(self);
        return ev::fromDouble(d && d->backend ? d->backend->num_actions() : 0);
    });
    b.accessor("inDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = Unwrap(self);
        return ev::fromDouble(d && d->backend ? d->backend->in_dim() : 0);
    });
}

} // namespace

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

std::shared_ptr<learn::BatchedNet> batchedNetShared(Value v) {
    if (auto pvn = policyValueNetShared(v)) return pvn;
    if (auto tx = heroNetTxShared(v)) return tx;
    return nullptr;
}

Value makeGenericSituationObject(const learn::GenericSituation& s) {
    ObjectBuilder o;
    o.set("obs", makeFloat32Array(s.obs.data(), s.obs.size()));
    o.set("policyTarget", makeFloat32Array(s.policy_target.data(), s.policy_target.size()));
    o.set("actionMask", makeFloat32Array(s.action_mask.data(), s.action_mask.size()));
    o.set("valueTarget", static_cast<double>(s.value_target));
    return o.get();
}

bool readGenericSituationValue(Value v, learn::GenericSituation& out) {
    if (!ev::isObject(v)) return false;
    // Each property read allocates; `v` is re-read from a root between them.
    ev::Persistent root(v);
    if (!readFloatVecProp(root.get(), "obs", out.obs, /*required=*/true)) return false;
    if (!readFloatVecProp(root.get(), "policyTarget", out.policy_target, /*required=*/true)) {
        return false;
    }
    readFloatVecProp(root.get(), "actionMask", out.action_mask, /*required=*/false);
    out.value_target = static_cast<float>(getDoubleProperty(root.get(), "valueTarget", 0.0));
    return true;
}

// ---------------------------------------------------------------------------
// The GenericMcts native prior / value fast path
// ---------------------------------------------------------------------------

learn::IInferenceBackend* inferenceBackendFromJS(Value v) {
    if (auto* d = unwrapDirectBackend(v)) return d->backend.get();
    if (auto* d = unwrapServerBackend(v)) return d->backend.get();
    return nullptr;
}

bgm::GenericPriorFn makeNativePriorFn(learn::IInferenceBackend* backend) {
    if (!backend) return nullptr;
    return [backend](const std::vector<float>& obs,
                     const std::vector<int>& legal) -> std::vector<float> {
        const auto r = backend->evaluate(obs);
        const int A = backend->num_actions();
        std::vector<float> probs(static_cast<size_t>(A), 0.0f);
        std::vector<uint8_t> mask(static_cast<size_t>(A), 0);
        for (int a : legal) {
            if (a >= 0 && a < A) mask[a] = 1;
        }
        float m = -1e30f;
        for (int a = 0; a < A; ++a) {
            if (mask[a] && r.logits[a] > m) m = r.logits[a];
        }
        float s = 0.0f;
        for (int a = 0; a < A; ++a) {
            if (!mask[a]) {
                probs[a] = 0.0f;
                continue;
            }
            probs[a] = std::exp(r.logits[a] - m);
            s += probs[a];
        }
        if (s > 0.0f) {
            for (int a = 0; a < A; ++a) probs[a] /= s;
        }
        return probs;
    };
}

bgm::GenericValueFn makeNativeValueFn(learn::IInferenceBackend* backend) {
    if (!backend) return nullptr;
    return [backend](const std::vector<float>& obs) -> float {
        return backend->evaluate(obs).value;
    };
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

void ensureAILearnGenericClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_genericBufferClass.init("AIGenericReplayBuffer", decorateGenericBuffer);
    g_genericTrainerClass.init("AIGenericExItTrainer", decorateGenericTrainer);
    g_inferenceServerClass.init("AIInferenceServer", decorateInferenceServer);
    g_directBackendClass.init("AIDirectBackend", [](ObjectBuilder& b) {
        decorateBackend<HostDirectBackend, unwrapDirectBackend>(b);
    });
    g_serverBackendClass.init("AIServerBackend", [](ObjectBuilder& b) {
        decorateBackend<HostServerBackend, unwrapServerBackend>(b);
    });
}

void installAILearnGeneric(ObjectBuilder& learnNs) {
    ensureAILearnGenericClassesInstalled();

    learnNs.def("createGenericReplayBuffer", 1, [](Value, std::span<const Value> a) -> Value {
        size_t cap = 4096;
        if (!a.empty()) {
            int v = i32At(a, 0);
            if (v > 0) cap = static_cast<size_t>(v);
        }
        auto cell = std::make_unique<HostGenericBuffer>();
        cell->buf = std::make_shared<learn::GenericReplayBuffer>(cap);
        return g_genericBufferClass.createInstance(std::move(cell));
    });

    learnNs.def("createGenericExItTrainer", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostGenericTrainer>();
        cell->trainer = std::make_unique<learn::GenericExItTrainer>();
        return g_genericTrainerClass.createInstance(std::move(cell));
    });

    learnNs.def("createInferenceServer", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createInferenceServer(net, config?)");
        auto net = batchedNetShared(a[0]);
        if (!net) {
            return ev::throwTypeError(
                "createInferenceServer: net must be a PolicyValueNet or SingleHeroNetTX "
                "(a BatchedNet) — SingleHeroNet is not batched-inference capable");
        }
        learn::BatchedInferenceServer::Config cfg{};
        if (a.size() >= 2 && ev::isObject(a[1])) {
            cfg.max_batch_size = getCountProp(a[1], "maxBatchSize", cfg.max_batch_size);
            cfg.max_wait_micros = getCountProp(a[1], "maxWaitMicros", cfg.max_wait_micros);
        }
        auto cell = std::make_unique<HostInferenceServer>();
        cell->netRef = net;
        cell->server = std::make_shared<learn::BatchedInferenceServer>(net.get(), cfg);
        return g_inferenceServerClass.createInstance(std::move(cell));
    });

    learnNs.def("createDirectBackend", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createDirectBackend(net)");
        auto net = batchedNetShared(a[0]);
        if (!net) {
            return ev::throwTypeError(
                "createDirectBackend: net must be a PolicyValueNet or SingleHeroNetTX");
        }
        auto cell = std::make_unique<HostDirectBackend>();
        cell->netRef = net;
        cell->backend = std::make_unique<learn::DirectBatchedNetBackend>(net.get());
        return g_directBackendClass.createInstance(std::move(cell));
    });

    learnNs.def("createServerBackend", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("createServerBackend(server, net)");
        auto* sd = unwrapInferenceServer(a[0]);
        if (!sd || !sd->server) {
            return ev::throwTypeError("createServerBackend: expected an InferenceServer");
        }
        auto net = batchedNetShared(a[1]);
        if (!net) {
            return ev::throwTypeError(
                "createServerBackend: net must be a PolicyValueNet or SingleHeroNetTX");
        }
        auto cell = std::make_unique<HostServerBackend>();
        cell->netRef = net;
        // Shares ownership: server.shutdown() drops only the server's own
        // reference, so a backend still in use never calls a freed server.
        cell->serverRef = sd->server;
        cell->backend = std::make_unique<learn::ServerBackend>(sd->server.get(), net.get());
        return g_serverBackendClass.createInstance(std::move(cell));
    });
}

} // namespace brogameagent::api

#else  // !BROGAMEAGENT_HAS_NN

#include "host_ai_mcts_shared.h"

namespace brogameagent::api {

brogameagent::learn::IInferenceBackend* inferenceBackendFromJS(Value) { return nullptr; }
bgm::GenericPriorFn makeNativePriorFn(brogameagent::learn::IInferenceBackend*) { return nullptr; }
bgm::GenericValueFn makeNativeValueFn(brogameagent::learn::IInferenceBackend*) { return nullptr; }

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
