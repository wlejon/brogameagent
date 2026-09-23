// bro.ai.game.nn — the circuit half: Linear, Relu, Tanh, DeepSetsEncoder,
// ValueHead, FactoredPolicyHead.
//
// Ported from the pre-transition qjsbind classes in ai_nn_bindings.cpp. The
// one deliberate change is the tensor type: the old AITensor class is gone,
// so every place the old surface took one now takes a bro.tensor GpuTensor
// handle or a Float32Array (viewed in place) — see tensorArg().

#include "host_ai_nn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <brotensor/api.h>

#include <exception>

namespace brogameagent::api {

HostClass g_linearClass;
HostClass g_reluClass;
HostClass g_tanhClass;
HostClass g_deepSetsClass;
HostClass g_valueHeadClass;
HostClass g_factoredHeadClass;

namespace {

constexpr uint64_t kDefaultSeed = 0xC0DE1234ULL;

/// brotensor throws std::runtime_error on a shape/device mismatch; an
/// uncaught throw would unwind through the compiled JS frame, so every call
/// into a circuit goes through this.
template <typename Fn>
Value guarded(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

/// The `save()` body every circuit shares.
template <typename T>
Value saveCircuit(T& c) {
    std::vector<uint8_t> bytes;
    try {
        c.save_to(bytes);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return makeUint8Array(bytes.data(), bytes.size());
}

/// The `load(bytes)` body every circuit shares.
template <typename T>
Value loadCircuit(T& c, Value arg) {
    RawBytes raw = rawBytes(arg);
    if (!raw.data) return ev::throwTypeError("expected TypedArray");
    size_t off = 0;
    try {
        c.load_from(raw.data, off, raw.size);
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

/// A parameter tensor handed back to JS. The pre-transition binding copied
/// the tensor into a fresh AITensor (`new TensorData{ d->l.W() }`), so the
/// value JS sees is a snapshot, not an alias; a GpuTensor built the same way
/// keeps that.
Value paramTensor(const brotensor::Tensor& t) {
    Value v = brotensor::api::createGpuTensorValue(t);
    if (!ev::isObject(v)) {
        return ev::throwError("bro.tensor natives are not registered in this realm");
    }
    return v;
}

// ── Linear ────────────────────────────────────────────────────────────────

void decorateLinear(ObjectBuilder& b) {
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return ev::fromUtf8(d ? d->l.name() : "");
    });
    b.accessor("inDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return ev::fromDouble(d ? d->l.in_dim() : 0);
    });
    b.accessor("outDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return ev::fromDouble(d ? d->l.out_dim() : 0);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return ev::fromDouble(d ? d->l.num_params() : 0);
    });
    b.accessor("W", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return d ? paramTensor(d->l.W()) : ev::undefined();
    });
    b.accessor("b", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return d ? paramTensor(d->l.b()) : ev::undefined();
    });
    b.accessor("dW", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return d ? paramTensor(d->l.dW()) : ev::undefined();
    });
    b.accessor("dB", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return d ? paramTensor(d->l.dB()) : ev::undefined();
    });

    b.def("init", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapLinear(self);
        if (!d) return ev::throwTypeError("Linear.init: bad receiver");
        uint64_t s = readSeedArg(argAt(a, 2), kDefaultSeed);
        const int in = dimAt(a, 0, "Linear.init: in"), out = dimAt(a, 1, "Linear.init: out");
        try {
            d->l.init(in, out, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeBigIntValue(s);
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapLinear(self);
        auto [x, y] = tensorArgs<2>(a, {0, 1});
        if (!d || !x || !y) return ev::throwTypeError("Linear.forward(x,y) expects Tensors");
        return guarded([&] { d->l.forward(*x.ptr, *y.ptr); });
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapLinear(self);
        auto [dY, dX] = tensorArgs<2>(a, {0, 1});
        if (!d || !dY || !dX) return ev::throwTypeError("Linear.backward(dY,dX) expects Tensors");
        return guarded([&] { d->l.backward(*dY.ptr, *dX.ptr); });
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        if (d) d->l.zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapLinear(self);
        if (!d) return ev::undefined();
        return guarded([&] {
            d->l.sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        });
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapLinear(self);
        return d ? saveCircuit(d->l) : ev::undefined();
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapLinear(self);
        if (!d || a.empty()) return ev::undefined();
        return loadCircuit(d->l, a[0]);
    });
}

// ── Relu / Tanh ───────────────────────────────────────────────────────────

template <typename Host, typename Member, Host* (*Unwrap)(Value), Member Host::*Field>
void decorateActivation(ObjectBuilder& b, const char* label) {
    b.accessor("name", [label](Value self, std::span<const Value>) -> Value {
        auto* d = Unwrap(self);
        return ev::fromUtf8(d ? (d->*Field).name() : label);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = Unwrap(self);
        return ev::fromDouble(d ? (d->*Field).num_params() : 0);
    });
    b.def("forward", 2, [label](Value self, std::span<const Value> a) -> Value {
        auto* d = Unwrap(self);
        auto [x, y] = tensorArgs<2>(a, {0, 1});
        if (!d || !x || !y) return ev::throwTypeError(std::string(label) + ".forward(x,y)");
        return guarded([&] { (d->*Field).forward(*x.ptr, *y.ptr); });
    });
    b.def("backward", 2, [label](Value self, std::span<const Value> a) -> Value {
        auto* d = Unwrap(self);
        auto [dY, dX] = tensorArgs<2>(a, {0, 1});
        if (!d || !dY || !dX) return ev::throwTypeError(std::string(label) + ".backward(dY,dX)");
        return guarded([&] { (d->*Field).backward(*dY.ptr, *dX.ptr); });
    });
    // Present so an activation drops into the same training loop as a
    // parameterised circuit; both were no-ops before the transition too.
    b.def("zeroGrad", 0, [](Value, std::span<const Value>) -> Value { return ev::undefined(); });
    b.def("sgdStep", 2, [](Value, std::span<const Value>) -> Value { return ev::undefined(); });
}

// ── DeepSetsEncoder ───────────────────────────────────────────────────────

void decorateDeepSets(ObjectBuilder& b) {
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapDeepSets(self);
        return ev::fromUtf8(d ? d->e.name() : "");
    });
    b.accessor("outDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapDeepSets(self);
        return ev::fromDouble(d ? d->e.out_dim() : 0);
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapDeepSets(self);
        return ev::fromDouble(d ? d->e.num_params() : 0);
    });
    b.def("init", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapDeepSets(self);
        if (!d) return ev::throwTypeError("DeepSetsEncoder.init: bad receiver");
        nn::DeepSetsEncoder::Config cfg{};
        // a[0] is a rooted slot, current across the first read's allocation;
        // a local copy of it would not be.
        if (!a.empty() && ev::isObject(a[0])) {
            cfg.hidden = getCountProp(a[0], "hidden", cfg.hidden);
            cfg.embed_dim = getCountProp(a[0], "embedDim", cfg.embed_dim);
        }
        uint64_t s = readSeedArg(argAt(a, 1), kDefaultSeed);
        try {
            d->e.init(cfg, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeBigIntValue(s);
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapDeepSets(self);
        auto [x, y] = tensorArgs<2>(a, {0, 1});
        if (!d || !x || !y) return ev::throwTypeError("DeepSetsEncoder.forward(x,y)");
        return guarded([&] { d->e.forward(*x.ptr, *y.ptr); });
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapDeepSets(self);
        auto [dY, dX] = tensorArgs<2>(a, {0, 1});
        if (!d || !dY || !dX) return ev::throwTypeError("DeepSetsEncoder.backward(dY,dX)");
        return guarded([&] { d->e.backward(*dY.ptr, *dX.ptr); });
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapDeepSets(self);
        if (d) d->e.zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapDeepSets(self);
        if (!d) return ev::undefined();
        return guarded([&] {
            d->e.sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        });
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapDeepSets(self);
        return d ? saveCircuit(d->e) : ev::undefined();
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapDeepSets(self);
        if (!d || a.empty()) return ev::undefined();
        return loadCircuit(d->e, a[0]);
    });
}

// ── ValueHead ─────────────────────────────────────────────────────────────

void decorateValueHead(ObjectBuilder& b) {
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapValueHead(self);
        return ev::fromUtf8(d ? d->v.name() : "");
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapValueHead(self);
        return ev::fromDouble(d ? d->v.num_params() : 0);
    });
    b.def("init", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapValueHead(self);
        if (!d) return ev::throwTypeError("ValueHead.init: bad receiver");
        uint64_t s = readSeedArg(argAt(a, 2), kDefaultSeed);
        const int in = dimAt(a, 0, "ValueHead.init: in"), hidden = dimAt(a, 1, "ValueHead.init: hidden");
        try {
            d->v.init(in, hidden, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeBigIntValue(s);
    });
    b.def("forward", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapValueHead(self);
        TensorArg e = tensorArg(argAt(a, 0));
        if (!d || !e) return ev::throwTypeError("ValueHead.forward(embed)");
        float v = 0.0f;
        try {
            d->v.forward(*e.ptr, v);
        } catch (const std::exception& ex) {
            return ev::throwError(ex.what());
        }
        return ev::fromDouble(v);
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapValueHead(self);
        TensorArg de = tensorArg(argAt(a, 1));
        if (!d || !de) return ev::throwTypeError("ValueHead.backward(dValue,dEmbed)");
        return guarded([&] { d->v.backward(static_cast<float>(numAt(a, 0)), *de.ptr); });
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapValueHead(self);
        if (d) d->v.zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapValueHead(self);
        if (!d) return ev::undefined();
        return guarded([&] {
            d->v.sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        });
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapValueHead(self);
        return d ? saveCircuit(d->v) : ev::undefined();
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapValueHead(self);
        if (!d || a.empty()) return ev::undefined();
        return loadCircuit(d->v, a[0]);
    });
}

// ── FactoredPolicyHead ────────────────────────────────────────────────────

void decorateFactoredHead(ObjectBuilder& b) {
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFactoredHead(self);
        return ev::fromUtf8(d ? d->h.name() : "");
    });
    b.accessor("numParams", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFactoredHead(self);
        return ev::fromDouble(d ? d->h.num_params() : 0);
    });
    b.accessor("totalLogits", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFactoredHead(self);
        return ev::fromDouble(d ? d->h.total_logits() : 0);
    });
    b.def("init", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFactoredHead(self);
        if (!d) return ev::throwTypeError("FactoredPolicyHead.init: bad receiver");
        uint64_t s = readSeedArg(argAt(a, 1), kDefaultSeed);
        const int in = dimAt(a, 0, "FactoredPolicyHead.init: in");
        try {
            d->h.init(in, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeBigIntValue(s);
    });
    b.def("forward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFactoredHead(self);
        auto [e, l] = tensorArgs<2>(a, {0, 1});
        if (!d || !e || !l) return ev::throwTypeError("FactoredPolicyHead.forward(embed,logits)");
        return guarded([&] { d->h.forward(*e.ptr, *l.ptr); });
    });
    b.def("backward", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFactoredHead(self);
        auto [dl, de] = tensorArgs<2>(a, {0, 1});
        if (!d || !dl || !de) return ev::throwTypeError("FactoredPolicyHead.backward(dLogits,dEmbed)");
        return guarded([&] { d->h.backward(*dl.ptr, *de.ptr); });
    });
    b.def("zeroGrad", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFactoredHead(self);
        if (d) d->h.zero_grad();
        return ev::undefined();
    });
    b.def("sgdStep", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFactoredHead(self);
        if (!d) return ev::undefined();
        return guarded([&] {
            d->h.sgd_step(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        });
    });
    b.def("save", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFactoredHead(self);
        return d ? saveCircuit(d->h) : ev::undefined();
    });
    b.def("load", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFactoredHead(self);
        if (!d || a.empty()) return ev::undefined();
        return loadCircuit(d->h, a[0]);
    });
}

} // namespace

void ensureAINnCircuitClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_linearClass.init("AILinear", decorateLinear);
    g_reluClass.init("AIRelu", [](ObjectBuilder& b) {
        decorateActivation<HostRelu, nn::Relu, unwrapRelu, &HostRelu::r>(b, "Relu");
    });
    g_tanhClass.init("AITanh", [](ObjectBuilder& b) {
        decorateActivation<HostTanh, nn::Tanh, unwrapTanh, &HostTanh::t>(b, "Tanh");
    });
    g_deepSetsClass.init("AIDeepSetsEncoder", decorateDeepSets);
    g_valueHeadClass.init("AIValueHead", decorateValueHead);
    g_factoredHeadClass.init("AIFactoredPolicyHead", decorateFactoredHead);
}

void installAINnCircuits(ObjectBuilder& nnNs) {
    ensureAINnCircuitClassesInstalled();

    // createTensor(rows, cols=1) — the old AITensor factory, now answering a
    // bro.tensor GpuTensor so there is exactly one tensor type in the stack.
    nnNs.def("createTensor", 2, [](Value, std::span<const Value> a) -> Value {
        const int r = a.empty() ? 0 : dimAt(a, 0, "createTensor: rows");
        const int c = a.size() >= 2 ? dimAt(a, 1, "createTensor: cols") : 1;
        Value v = brotensor::api::createGpuTensorValue(brotensor::Tensor::mat(r, c));
        if (!ev::isObject(v)) {
            return ev::throwError("bro.tensor natives are not registered in this realm");
        }
        return v;
    });

    nnNs.def("createLinear", 3, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostLinear>();
        if (a.size() >= 2) {
            uint64_t s = a.size() >= 3 ? readSeedArg(a[2], kDefaultSeed) : kDefaultSeed;
            const int in = dimAt(a, 0, "createLinear: in"), out = dimAt(a, 1, "createLinear: out");
            try {
                cell->l.init(in, out, s);
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        }
        return g_linearClass.createInstance(std::move(cell));
    });
    nnNs.def("createRelu", 0, [](Value, std::span<const Value>) -> Value {
        return g_reluClass.createInstance(std::make_unique<HostRelu>());
    });
    nnNs.def("createTanh", 0, [](Value, std::span<const Value>) -> Value {
        return g_tanhClass.createInstance(std::make_unique<HostTanh>());
    });
    nnNs.def("createDeepSetsEncoder", 2, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostDeepSets>();
        nn::DeepSetsEncoder::Config cfg{};
        // a[0] is a rooted slot, current across the first read's allocation;
        // a local copy of it would not be.
        if (!a.empty() && ev::isObject(a[0])) {
            cfg.hidden = getCountProp(a[0], "hidden", cfg.hidden);
            cfg.embed_dim = getCountProp(a[0], "embedDim", cfg.embed_dim);
        }
        uint64_t s = a.size() >= 2 ? readSeedArg(a[1], kDefaultSeed) : kDefaultSeed;
        try {
            cell->e.init(cfg, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return g_deepSetsClass.createInstance(std::move(cell));
    });
    nnNs.def("createValueHead", 3, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostValueHead>();
        if (a.size() >= 2) {
            uint64_t s = a.size() >= 3 ? readSeedArg(a[2], kDefaultSeed) : kDefaultSeed;
            const int in = dimAt(a, 0, "createValueHead: in");
            const int hidden = dimAt(a, 1, "createValueHead: hidden");
            try {
                cell->v.init(in, hidden, s);
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        }
        return g_valueHeadClass.createInstance(std::move(cell));
    });
    nnNs.def("createFactoredPolicyHead", 2, [](Value, std::span<const Value> a) -> Value {
        auto cell = std::make_unique<HostFactoredHead>();
        if (!a.empty()) {
            uint64_t s = a.size() >= 2 ? readSeedArg(a[1], kDefaultSeed) : kDefaultSeed;
            const int in = dimAt(a, 0, "createFactoredPolicyHead: in");
            try {
                cell->h.init(in, s);
            } catch (const std::exception& e) {
                return ev::throwError(e.what());
            }
        }
        return g_factoredHeadClass.createInstance(std::move(cell));
    });
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
