// bro.ai.game.nn — the net half: SingleHeroNet, PolicyValueNet,
// SingleHeroNetTX, WeightsHandle, their factories, and the installer that
// mounts the whole `nn` namespace.
//
// Ported from ai_nn_bindings.cpp. Device migration keeps the old GPU-first
// rule: `to("gpu")` on a build without a GPU backend throws rather than
// quietly staying on the CPU.

#include "host_ai_nn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <exception>

namespace brogameagent::api {

HostClass g_singleHeroNetClass;
HostClass g_policyValueNetClass;
HostClass g_heroNetTxClass;
HostClass g_weightsHandleClass;

namespace {

constexpr uint64_t kDefaultSeed = 0xC0DE1234ULL;

Value intArrayMethod(const std::vector<int>& v) { return makeIntArrayValue(v); }

/// net.forward(x, logits) -> value
template <typename Net>
Value netForward(Net* net, std::span<const Value> a, const char* label) {
    if (!net) return ev::throwError("net not initialized");
    TensorArg x = tensorArg(argAt(a, 0));
    TensorArg l = tensorArg(argAt(a, 1));
    if (!x || !l) return ev::throwTypeError(std::string(label) + ".forward(x,logits)");
    float v = 0.0f;
    try {
        net->forward(*x.ptr, v, *l.ptr);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::fromDouble(v);
}

template <typename Net>
Value netBackward(Net* net, std::span<const Value> a, const char* label) {
    if (!net) return ev::throwError("net not initialized");
    TensorArg dl = tensorArg(argAt(a, 1));
    if (!dl) return ev::throwTypeError(std::string(label) + ".backward(dValue,dLogits)");
    try {
        net->backward(static_cast<float>(numAt(a, 0)), *dl.ptr);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

template <typename Net>
Value netForwardBatched(Net* net, std::span<const Value> a, const char* label) {
    if (!net) return ev::throwError("net not initialized");
    TensorArg x = tensorArg(argAt(a, 0));
    TensorArg l = tensorArg(argAt(a, 1));
    TensorArg v = tensorArg(argAt(a, 2));
    if (!x || !l || !v) {
        return ev::throwTypeError(std::string(label) + ".forwardBatched(x,logits,values) expects Tensors");
    }
    try {
        net->forward_batched(*x.ptr, *l.ptr, *v.ptr);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

template <typename Net>
Value netSave(Net* net) {
    if (!net) return makeUint8Array(nullptr, 0);
    std::vector<uint8_t> blob;
    try {
        blob = net->save();
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return makeUint8Array(blob.data(), blob.size());
}

/// net.load(blob). The C++ loaders throw std::runtime_error on a malformed
/// blob (magic / version / size mismatch); that has to arrive in JS as a
/// catchable TypeError rather than unwinding through compiled frames.
template <typename Net>
Value netLoad(Net* net, Value arg, const char* label) {
    if (!net) return ev::undefined();
    RawBytes raw = rawBytes(arg);
    if (!raw.data) return ev::throwTypeError("expected TypedArray");
    std::vector<uint8_t> blob(raw.data, raw.data + raw.size);
    try {
        net->load(blob);
    } catch (const std::exception& e) {
        return ev::throwTypeError(std::string(label) + ".load: " + e.what());
    }
    return ev::undefined();
}

template <typename Net>
Value netTo(Net* net, std::span<const Value> a) {
    if (!net) return ev::throwError("net not initialized");
    brotensor::Device target = brotensor::Device::CPU;
    std::string error;
    if (!resolveDevice(strAt(a, 0), target, error)) return ev::throwError(error);
    try {
        net->to(target);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

// ── SingleHeroNet ─────────────────────────────────────────────────────────

void decorateSingleHeroNet(ObjectBuilder& b) {
    b.accessor("embedDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return ev::fromDouble(d && d->net ? d->net->embed_dim() : 0);
    });
    b.accessor("trunkDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return ev::fromDouble(d && d->net ? d->net->trunk_dim() : 0);
    });
    b.accessor("policyLogits", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return ev::fromDouble(d && d->net ? d->net->policy_logits() : 0);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return ev::fromDouble(d && d->net ? d->net->num_params() : 0);
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return netForward(d ? d->net.get() : nullptr, a, "SingleHeroNet");
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return netBackward(d ? d->net.get() : nullptr, a, "SingleHeroNet");
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        if (d && d->net) d->net->zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        if (d && d->net) {
            d->net->sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        }
        return ev::undefined();
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        return netSave(d ? d->net.get() : nullptr);
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapSingleHeroNet(self);
        if (!d || a.empty()) return ev::undefined();
        return netLoad(d->net.get(), a[0], "SingleHeroNet");
    });
}

// ── PolicyValueNet ────────────────────────────────────────────────────────

void decoratePolicyValueNet(ObjectBuilder& b) {
    b.accessor("inDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return ev::fromDouble(d && d->net ? d->net->in_dim() : 0);
    });
    b.accessor("numActions", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return ev::fromDouble(d && d->net ? d->net->num_actions() : 0);
    });
    b.accessor("trunkDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return ev::fromDouble(d && d->net ? d->net->trunk_dim() : 0);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return ev::fromDouble(d && d->net ? d->net->num_params() : 0);
    });
    b.accessor("numHeads", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return ev::fromDouble(d && d->net ? d->net->num_heads() : 0);
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (!d || !d->net) return ev::fromUtf8("cpu");
        return ev::fromUtf8(d->net->device() != brotensor::Device::CPU ? "gpu" : "cpu");
    });
    b.def("headSizes", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (!d || !d->net) return makeIntArrayValue({});
        return intArrayMethod(d->net->head_sizes());
    });
    b.def("headOffsets", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (!d || !d->net) return makeIntArrayValue({});
        return intArrayMethod(d->net->head_offsets());
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return netForward(d ? d->net.get() : nullptr, a, "PolicyValueNet");
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return netBackward(d ? d->net.get() : nullptr, a, "PolicyValueNet");
    });
    b.def("forwardBatched", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return netForwardBatched(d ? d->net.get() : nullptr, a, "PolicyValueNet");
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (d && d->net) d->net->zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (d && d->net) {
            d->net->sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        }
        return ev::undefined();
    });
    b.def("to", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return netTo(d ? d->net.get() : nullptr, a);
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        return netSave(d ? d->net.get() : nullptr);
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapPolicyValueNet(self);
        if (!d || a.empty()) return ev::undefined();
        return netLoad(d->net.get(), a[0], "PolicyValueNet");
    });
}

// ── SingleHeroNetTX ───────────────────────────────────────────────────────

void decorateHeroNetTx(ObjectBuilder& b) {
    b.accessor("inDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return ev::fromDouble(d && d->net ? d->net->in_dim() : 0);
    });
    b.accessor("numActions", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return ev::fromDouble(d && d->net ? d->net->num_actions() : 0);
    });
    b.accessor("numHeads", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return ev::fromDouble(d && d->net ? d->net->num_heads() : 0);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return ev::fromDouble(d && d->net ? d->net->num_params() : 0);
    });
    b.accessor("device", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (!d || !d->net) return ev::fromUtf8("cpu");
        return ev::fromUtf8(d->net->device() != brotensor::Device::CPU ? "gpu" : "cpu");
    });
    b.def("headSizes", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (!d || !d->net) return makeIntArrayValue({});
        return intArrayMethod(d->net->head_sizes());
    });
    b.def("headOffsets", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (!d || !d->net) return makeIntArrayValue({});
        return intArrayMethod(d->net->head_offsets());
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return netForward(d ? d->net.get() : nullptr, a, "SingleHeroNetTX");
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return netBackward(d ? d->net.get() : nullptr, a, "SingleHeroNetTX");
    });
    b.def("forwardBatched", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return netForwardBatched(d ? d->net.get() : nullptr, a, "SingleHeroNetTX");
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (d && d->net) d->net->zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (d && d->net) {
            d->net->sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        }
        return ev::undefined();
    });
    b.def("adamStep", 5, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (d && d->net) {
            d->net->adam_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)),
                              static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3)),
                              i32At(a, 4));
        }
        return ev::undefined();
    });
    b.def("to", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return netTo(d ? d->net.get() : nullptr, a);
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapHeroNetTx(self);
        return netSave(d ? d->net.get() : nullptr);
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapHeroNetTx(self);
        if (!d || a.empty()) return ev::undefined();
        return netLoad(d->net.get(), a[0], "SingleHeroNetTX");
    });
}

// ── WeightsHandle ─────────────────────────────────────────────────────────

void decorateWeightsHandle(ObjectBuilder& b) {
    b.def("publish", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapWeightsHandle(self);
        if (!d || !d->handle || a.size() < 2) return ev::throwTypeError("publish(blob, version)");
        RawBytes raw = rawBytes(a[0]);
        if (!raw.data) return ev::throwTypeError("expected TypedArray");
        std::vector<uint8_t> blob(raw.data, raw.data + raw.size);
        d->handle->publish(std::move(blob), readSeedArg(a[1], 0));
        return ev::undefined();
    });
    b.def("snapshot", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapWeightsHandle(self);
        if (!d || !d->handle) return ev::null();
        uint64_t version = 0;
        auto sp = d->handle->snapshot(&version);
        if (!sp) return ev::null();
        ObjectBuilder o;
        o.set("blob", makeUint8Array(sp->data(), sp->size()));
        o.set("version", makeBigIntValue(version));
        return o.get();
    });
    b.def("version", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapWeightsHandle(self);
        return makeBigIntValue(d && d->handle ? d->handle->version() : 0);
    });
}

} // namespace

void ensureAINnNetClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_singleHeroNetClass.init("AISingleHeroNet", decorateSingleHeroNet);
    g_policyValueNetClass.init("AIPolicyValueNet", decoratePolicyValueNet);
    g_heroNetTxClass.init("AISingleHeroNetTX", decorateHeroNetTx);
    g_weightsHandleClass.init("AIWeightsHandle", decorateWeightsHandle);
}

void installAINnNets(ObjectBuilder& nnNs) {
    ensureAINnNetClassesInstalled();

    nnNs.def("createSingleHeroNet", 1, [](Value, std::span<const Value> a) -> Value {
        nn::SingleHeroNet::Config cfg{};
        Value opts = argAt(a, 0);
        if (ev::isObject(opts)) {
            ev::Persistent root(opts);
            Value encV = ev::getProperty(root.get(), "enc");
            if (ev::isObject(encV)) {
                cfg.enc.hidden = getIntProp(encV, "hidden", cfg.enc.hidden);
                cfg.enc.embed_dim = getIntProp(encV, "embedDim", cfg.enc.embed_dim);
            }
            cfg.trunk_hidden = getIntProp(root.get(), "trunkHidden", cfg.trunk_hidden);
            cfg.value_hidden = getIntProp(root.get(), "valueHidden", cfg.value_hidden);
            cfg.seed = readSeedArg(ev::getProperty(root.get(), "seed"), cfg.seed);
        }
        auto net = std::make_shared<nn::SingleHeroNet>();
        try {
            net->init(cfg);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        auto cell = std::make_unique<HostSingleHeroNet>();
        cell->net = std::move(net);
        return g_singleHeroNetClass.createInstance(std::move(cell));
    });

    nnNs.def("createPolicyValueNet", 1, [](Value, std::span<const Value> a) -> Value {
        nn::PolicyValueNet::Config cfg{};
        Value opts = argAt(a, 0);
        if (ev::isObject(opts)) {
            ev::Persistent root(opts);
            cfg.in_dim = getIntProp(root.get(), "inDim", cfg.in_dim);
            cfg.num_actions = getIntProp(root.get(), "numActions", cfg.num_actions);
            cfg.value_hidden = getIntProp(root.get(), "valueHidden", cfg.value_hidden);
            auto hidden = readIntArrayProp(root.get(), "hidden");
            if (!hidden.empty()) cfg.hidden = std::move(hidden);
            cfg.head_sizes = readIntArrayProp(root.get(), "headSizes");
            cfg.seed = readSeedArg(ev::getProperty(root.get(), "seed"), cfg.seed);
        }
        const bool actionsOk = cfg.num_actions > 0 || !cfg.head_sizes.empty();
        if (cfg.in_dim <= 0 || !actionsOk || cfg.hidden.empty() || cfg.value_hidden <= 0) {
            return ev::throwTypeError(
                "createPolicyValueNet({inDim,numActions|headSizes,hidden:[...],valueHidden,seed?}) — "
                "inDim, hidden[], valueHidden are required, plus one of numActions or headSizes");
        }
        auto net = std::make_shared<nn::PolicyValueNet>();
        try {
            net->init(cfg);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        auto cell = std::make_unique<HostPolicyValueNet>();
        cell->net = std::move(net);
        return g_policyValueNetClass.createInstance(std::move(cell));
    });

    nnNs.def("createSingleHeroNetTX", 1, [](Value, std::span<const Value> a) -> Value {
        nn::SingleHeroNetTX::Config cfg{};
        Value opts = argAt(a, 0);
        if (ev::isObject(opts)) {
            ev::Persistent root(opts);
            cfg.self_hidden = getIntProp(root.get(), "selfHidden", cfg.self_hidden);
            cfg.slot_proj = getIntProp(root.get(), "slotProj", cfg.slot_proj);
            cfg.d_model = getIntProp(root.get(), "dModel", cfg.d_model);
            cfg.d_ff = getIntProp(root.get(), "dFf", cfg.d_ff);
            cfg.num_heads = getIntProp(root.get(), "numHeads", cfg.num_heads);
            cfg.num_blocks = getIntProp(root.get(), "numBlocks", cfg.num_blocks);
            cfg.trunk_hidden = getIntProp(root.get(), "trunkHidden", cfg.trunk_hidden);
            cfg.value_hidden = getIntProp(root.get(), "valueHidden", cfg.value_hidden);
            cfg.seed = readSeedArg(ev::getProperty(root.get(), "seed"), cfg.seed);
        }
        auto net = std::make_shared<nn::SingleHeroNetTX>();
        try {
            net->init(cfg);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        auto cell = std::make_unique<HostHeroNetTx>();
        cell->net = std::move(net);
        return g_heroNetTxClass.createInstance(std::move(cell));
    });

    nnNs.def("createWeightsHandle", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostWeightsHandle>();
        cell->handle = std::make_shared<nn::WeightsHandle>();
        return g_weightsHandleClass.createInstance(std::move(cell));
    });
}

void installAINeural(ObjectBuilder& game) {
    ObjectBuilder nnNs;
    installAINnCircuits(nnNs);
    installAINnNets(nnNs);
    installAINnOps(nnNs);
    game.set("nn", nnNs.get());
}

} // namespace brogameagent::api

#else  // !BROGAMEAGENT_HAS_NN

namespace brogameagent::api {

// The layer behind bro.ai.game.nn is compiled out: the namespace still
// exists and says so, exactly as the other compiled-out features do.
void installAINeural(ObjectBuilder& game) {
    ObjectBuilder nnNs;
    nnNs.set("available", false);
    game.set("nn", nnNs.get());
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
