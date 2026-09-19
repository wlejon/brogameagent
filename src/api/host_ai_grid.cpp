// bro.ai.game.grid — the observation/curriculum half: ObsWindow,
// FrameStack, FailureTape, BestCrop, PotentialShaper, StallDetector and the
// behaviour-cloning generator.
//
// Ported from ai_grid_bindings.cpp.

#include "host_ai_grid_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <algorithm>
#include <brogameagent/generic_mcts.h>

namespace brogameagent::api {

HostClass g_obsWindowClass;
HostClass g_frameStackClass;
HostClass g_failureTapeClass;
HostClass g_bestCropClass;
HostClass g_shaperClass;
HostClass g_stallDetectorClass;

namespace {

Value callJs(const ev::Persistent& fn, Value thisV, std::span<const Value> args, bool* ok) {
    if (ok) *ok = false;
    if (!ev::isFunction(fn.get())) return ev::undefined();
    ev::CallResult r = ev::call(fn.get(), thisV, args);
    if (r.thrown) return ev::undefined();
    if (ok) *ok = true;
    return r.value;
}

// ── ObsWindow ─────────────────────────────────────────────────────────────

void decorateObsWindow(ObjectBuilder& b) {
    b.accessor("outDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapObsWindow(self);
        return ev::fromDouble(d && d->win ? d->win->out_dim() : 0);
    });
    b.def("layout", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapObsWindow(self);
        ObjectBuilder o;
        if (!d || !d->win) return o.get();
        const auto& L = d->win->layout();
        o.set("cols", static_cast<double>(L.cols));
        o.set("rows", static_cast<double>(L.rows));
        o.set("tileOffset", static_cast<double>(L.tile_offset));
        o.set("tileChannels", static_cast<double>(L.tile_channels));
        o.set("tileSize", static_cast<double>(L.tile_size));
        o.set("layers", hostArrayOf(L.layers.size(), [&](size_t i) {
            ObjectBuilder li;
            li.set("offset", static_cast<double>(L.layers[i].offset));
            li.set("channels", static_cast<double>(L.layers[i].channels));
            li.set("size", static_cast<double>(L.layers[i].size));
            return li.get();
        }));
        o.set("selfOffset", static_cast<double>(L.self_offset));
        o.set("selfSize", static_cast<double>(L.self_size));
        o.set("total", static_cast<double>(L.total));
        return o.get();
    });
    b.def("build", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapObsWindow(self);
        if (!d || !d->win || a.size() < 2) return ev::undefined();
        std::vector<float> selfBlock;
        if (a.size() >= 3) selfBlock = readFloats(a[2]);
        std::vector<float> out(static_cast<size_t>(d->win->out_dim()), 0.0f);
        d->win->build(i32At(a, 0), i32At(a, 1),
                      selfBlock.empty() ? nullptr : selfBlock.data(), selfBlock.size(),
                      out.data());
        return makeFloat32Array(out.data(), out.size());
    });
}

Value createObsWindow(std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) {
        return ev::throwTypeError("createObsWindow(opts): expected an object");
    }
    ev::Persistent opts(a[0]);

    grid::ObsWindowSpec spec{};
    Value specV = ev::getProperty(opts.get(), "spec");
    ev::Persistent sp(ev::isObject(specV) ? specV : opts.get());
    spec.cols_behind = getIntProp(sp.get(), "colsBehind", 0);
    spec.cols_ahead = getIntProp(sp.get(), "colsAhead", 0);
    spec.rows_up = getIntProp(sp.get(), "rowsUp", 0);
    spec.rows_down = getIntProp(sp.get(), "rowsDown", 0);
    spec.tile_channels = getIntProp(sp.get(), "tileChannels", 1);
    spec.self_block_size = getIntProp(sp.get(), "selfBlockSize", 0);

    auto cell = std::make_unique<HostObsWindow>();
    HostObsWindow* d = cell.get();

    Value tileV = ev::getProperty(opts.get(), "tile");
    if (ev::isObject(tileV)) {
        ev::Persistent tile(tileV);
        spec.tile_channels = getIntProp(tile.get(), "channels", spec.tile_channels);
        spec.tile_normalize = readFloats(ev::getProperty(tile.get(), "normalize"));
        spec.oob_tile = readFloats(ev::getProperty(tile.get(), "oob"));
        Value sf = ev::getProperty(tile.get(), "sample");
        if (ev::isFunction(sf)) d->tileFn.set(sf);
    }

    grid::TileSampleFn tileFn;
    if (ev::isFunction(d->tileFn.get())) {
        const int TC = spec.tile_channels;
        tileFn = [d, TC](int col, int row, float* out) -> bool {
            Value args[2] = {ev::fromDouble(col), ev::fromDouble(row)};
            bool ok = false;
            Value r = callJs(d->tileFn, ev::undefined(), std::span<const Value>(args, 2), &ok);
            if (!ok) return false;
            if (ev::isBool(r)) {
                const float v = ev::toBool(r) ? 1.0f : 0.0f;
                for (int i = 0; i < TC; ++i) out[i] = v;
                return true;
            }
            if (ev::isNumber(r)) {
                const float v = static_cast<float>(ev::toDouble(r));
                for (int i = 0; i < TC; ++i) out[i] = v;
                return true;
            }
            std::vector<float> vec = readFloats(r);
            const int n = std::min<int>(TC, static_cast<int>(vec.size()));
            for (int i = 0; i < n; ++i) out[i] = vec[static_cast<size_t>(i)];
            for (int i = n; i < TC; ++i) out[i] = 0.0f;
            return !vec.empty();
        };
    } else {
        tileFn = [](int, int, float*) { return false; };
    }

    std::vector<grid::EntityLayerSpec> layers;
    Value layersV = ev::getProperty(opts.get(), "layers");
    if (ev::isObject(layersV)) {
        ev::Persistent arr(layersV);
        Value lenV = ev::getProperty(arr.get(), "length");
        const uint32_t n = (ev::isUndefined(lenV) || ev::isObject(lenV))
                               ? 0u
                               : static_cast<uint32_t>(ev::toDouble(lenV));
        for (uint32_t i = 0; i < n; ++i) {
            Value loV = ev::getElement(arr.get(), i);
            if (!ev::isObject(loV)) continue;
            ev::Persistent lo(loV);
            grid::EntityLayerSpec L;
            L.channels = getIntProp(lo.get(), "channels", 1);
            L.overwrite = getBoolProperty(lo.get(), "overwrite", false);
            L.normalize = readFloats(ev::getProperty(lo.get(), "normalize"));

            Value enumV = ev::getProperty(lo.get(), "enumerate");
            Value sampV = ev::getProperty(lo.get(), "sample");
            d->enumerateFns.emplace_back(ev::isFunction(enumV) ? enumV : ev::undefined());
            d->sampleFns.emplace_back(ev::isFunction(sampV) ? sampV : ev::undefined());
            const size_t idx = d->enumerateFns.size() - 1;
            const int chan = L.channels;

            L.enumerate_fn = [d, idx]() -> size_t {
                bool ok = false;
                Value r = callJs(d->enumerateFns[idx], ev::undefined(), {}, &ok);
                if (!ok || ev::isObject(r)) return 0;
                const double n2 = ev::toDouble(r);
                return n2 > 0 ? static_cast<size_t>(n2) : 0;
            };
            L.sample_fn = [d, idx, chan](size_t i) -> grid::EntityCell {
                grid::EntityCell c;
                Value arg = ev::fromDouble(static_cast<double>(i));
                bool ok = false;
                Value r = callJs(d->sampleFns[idx], ev::undefined(),
                                 std::span<const Value>(&arg, 1), &ok);
                if (!ok || !ev::isObject(r)) return c;
                ev::Persistent ro(r);
                c.col = getIntProp(ro.get(), "col", 0);
                c.row = getIntProp(ro.get(), "row", 0);
                Value vals = ev::getProperty(ro.get(), "values");
                if (!ev::isUndefined(vals) && !ev::isNull(vals)) {
                    c.values = readFloats(vals);
                } else {
                    // Single 'value' field shortcut.
                    c.values.assign(static_cast<size_t>(chan), 0.0f);
                    if (chan > 0) {
                        c.values[0] =
                            static_cast<float>(getDoubleProperty(ro.get(), "value", 1.0));
                    }
                }
                return c;
            };

            layers.push_back(std::move(L));
        }
    }

    d->win = std::make_unique<grid::ObsWindow>(spec, tileFn, std::move(layers));
    return g_obsWindowClass.createInstance(std::move(cell));
}

// ── FrameStack ────────────────────────────────────────────────────────────

void decorateFrameStack(ObjectBuilder& b) {
    b.accessor("outDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        return ev::fromDouble(d && d->fs ? d->fs->out_dim() : 0);
    });
    b.accessor("innerDim", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        return ev::fromDouble(d && d->fs ? d->fs->inner_dim() : 0);
    });
    b.accessor("k", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        return ev::fromDouble(d && d->fs ? d->fs->k() : 0);
    });
    b.accessor("filled", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        return ev::fromDouble(d && d->fs ? d->fs->filled() : 0);
    });
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        if (d && d->fs) d->fs->reset();
        return ev::undefined();
    });
    b.def("push", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFrameStack(self);
        if (!d || !d->fs || a.empty()) return ev::undefined();
        std::vector<float> v = readFloats(a[0]);
        if (static_cast<int>(v.size()) < d->fs->inner_dim()) {
            v.resize(static_cast<size_t>(d->fs->inner_dim()), 0.0f);
        }
        d->fs->push(v.data());
        return ev::undefined();
    });
    b.def("read", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFrameStack(self);
        if (!d || !d->fs) return makeFloat32Array(nullptr, 0);
        std::vector<float> v = d->fs->read();
        return makeFloat32Array(v.data(), v.size());
    });
}

// ── FailureTape ───────────────────────────────────────────────────────────

void decorateFailureTape(ObjectBuilder& b) {
    b.accessor("size", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFailureTape(self);
        return ev::fromDouble(d && d->tape ? d->tape->size() : 0);
    });
    b.accessor("capacity", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFailureTape(self);
        return ev::fromDouble(d && d->tape ? d->tape->capacity() : 0);
    });
    b.def("recordFailure", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFailureTape(self);
        if (!d || !d->tape || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        d->tape->record_failure(readFailureTail(a[0]));
        return ev::undefined();
    });
    b.def("multipliers", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFailureTape(self);
        if (!d || !d->tape || a.size() < 2) return makeFloat32Array(nullptr, 0);
        std::vector<float> m = d->tape->multipliers(strAt(a, 0), i32At(a, 1));
        return makeFloat32Array(m.data(), m.size());
    });
    b.def("applyPriors", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapFailureTape(self);
        if (!d || !d->tape || a.size() < 2) return makeFloat32Array(nullptr, 0);
        std::vector<float> prior = readFloats(a[1]);
        d->tape->apply_priors(strAt(a, 0), prior.data(), static_cast<int>(prior.size()));
        return makeFloat32Array(prior.data(), prior.size());
    });
    b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapFailureTape(self);
        if (d && d->tape) d->tape->clear();
        return ev::undefined();
    });
}

// ── BestCrop ──────────────────────────────────────────────────────────────

void decorateBestCrop(ObjectBuilder& b) {
    b.accessor("size", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapBestCrop(self);
        return ev::fromDouble(d && d->pool ? d->pool->size() : 0);
    });
    b.accessor("capacity", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapBestCrop(self);
        return ev::fromDouble(d && d->pool ? d->pool->capacity() : 0);
    });
    b.def("push", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapBestCrop(self);
        if (!d || !d->pool || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        ev::Persistent opts(a[0]);
        std::any snap{JsSnapshot(ev::getProperty(opts.get(), "snapshot"))};
        std::vector<int> prefix = readIntArrayValue(ev::getProperty(opts.get(), "prefix"));
        const float score = static_cast<float>(getDoubleProperty(opts.get(), "score", 0.0));
        const int depth = getIntProp(opts.get(), "depth", 0);
        d->pool->push(std::move(snap), std::move(prefix), score, depth);
        return ev::undefined();
    });
    b.def("seed", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapBestCrop(self);
        ObjectBuilder o;
        if (!d || !d->pool || d->pool->empty()) return o.get();
        grid::BestSeed seed = d->pool->seed(d->rng);
        Value snap = ev::null();
        if (seed.snapshot.has_value()) {
            if (auto* h = std::any_cast<JsSnapshot>(&seed.snapshot)) snap = h->get();
        }
        o.set("snapshot", snap);
        o.set("prefix", makeIntArrayValue(seed.prefix));
        return o.get();
    });
    b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapBestCrop(self);
        if (d && d->pool) d->pool->clear();
        return ev::undefined();
    });
}

// ── PotentialShaper / StallDetector ───────────────────────────────────────

void decorateShaper(ObjectBuilder& b) {
    b.accessor("gamma", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapShaper(self);
        return ev::fromDouble(d && d->sh ? d->sh->gamma() : 0.0);
    });
    b.def("reset", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapShaper(self);
        if (d && d->sh) d->sh->reset(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });
    b.def("step", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapShaper(self);
        if (!d || !d->sh) return ev::fromDouble(0.0);
        return ev::fromDouble(d->sh->step(static_cast<float>(numAt(a, 0))));
    });
}

void decorateStallDetector(ObjectBuilder& b) {
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapStallDetector(self);
        if (d && d->det) d->det->reset();
        return ev::undefined();
    });
    b.def("tick", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapStallDetector(self);
        if (!d || !d->det) return ev::fromBool(false);
        return ev::fromBool(d->det->tick(static_cast<float>(numAt(a, 0))));
    });
}

// ── generateBC ────────────────────────────────────────────────────────────

/// The five JS env methods generateBC drives, rooted for the call's duration.
struct JsEnv {
    ev::Persistent obj, snapshot, restore, step, legal, observe;
    int numActions = 0;
};

bool readJsEnv(Value envV, JsEnv& out) {
    if (!ev::isObject(envV)) return false;
    out.obj.set(envV);
    out.numActions = getIntProp(out.obj.get(), "numActions", 0);
    auto method = [&](const char* n) {
        Value f = ev::getProperty(out.obj.get(), n);
        return ev::isFunction(f) ? f : ev::undefined();
    };
    out.snapshot.set(method("snapshot"));
    out.restore.set(method("restore"));
    out.step.set(method("step"));
    out.legal.set(method("legalActions"));
    out.observe.set(method("observe"));
    return ev::isFunction(out.snapshot.get()) && ev::isFunction(out.restore.get()) &&
           ev::isFunction(out.step.get()) && ev::isFunction(out.legal.get()) &&
           ev::isFunction(out.observe.get()) && out.numActions > 0;
}

bgm::GenericEnv buildEnvFromJs(JsEnv* cb) {
    bgm::GenericEnv env;
    env.num_actions = cb->numActions;
    env.snapshot_fn = [cb]() -> std::any {
        bool ok = false;
        Value r = callJs(cb->snapshot, cb->obj.get(), {}, &ok);
        if (!ok) return std::any{};
        return std::any{JsSnapshot(r)};
    };
    env.restore_fn = [cb](const std::any& s) {
        if (!s.has_value()) return;
        const auto* h = std::any_cast<JsSnapshot>(&s);
        if (!h) return;
        Value arg = h->get();
        callJs(cb->restore, cb->obj.get(), std::span<const Value>(&arg, 1), nullptr);
    };
    env.step_fn = [cb](int action) -> bgm::GenericStepResult {
        Value arg = ev::fromDouble(action);
        bool ok = false;
        Value r = callJs(cb->step, cb->obj.get(), std::span<const Value>(&arg, 1), &ok);
        bgm::GenericStepResult out{};
        if (!ok || !ev::isObject(r)) return out;
        ev::Persistent ro(r);
        out.reward = static_cast<float>(getDoubleProperty(ro.get(), "reward", 0.0));
        out.done = getBoolProperty(ro.get(), "done", false);
        return out;
    };
    env.legal_actions_fn = [cb]() -> std::vector<int> {
        bool ok = false;
        Value r = callJs(cb->legal, cb->obj.get(), {}, &ok);
        if (!ok) return {};
        return readIntArrayValue(r);
    };
    env.observe_fn = [cb]() -> std::vector<float> {
        bool ok = false;
        Value r = callJs(cb->observe, cb->obj.get(), {}, &ok);
        if (!ok) return {};
        return readFloats(r);
    };
    return env;
}

Value generateBC(std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) {
        return ev::throwTypeError("generateBC(opts): expected object");
    }
    ev::Persistent opts(a[0]);

    JsEnv cb;
    if (!readJsEnv(ev::getProperty(opts.get(), "env"), cb)) {
        return ev::throwTypeError(
            "generateBC: env missing snapshot/restore/step/legalActions/observe/numActions");
    }
    bgm::GenericEnv env = buildEnvFromJs(&cb);

    ev::Persistent heuristic(ev::getProperty(opts.get(), "heuristic"));
    if (!ev::isFunction(heuristic.get())) {
        return ev::throwTypeError("generateBC: heuristic must be a function");
    }
    grid::HeuristicPolicyFn policy = [&heuristic](const std::vector<float>& obs,
                                                  const std::vector<int>& legal) -> int {
        Value args[2] = {makeFloat32Array(obs.data(), obs.size()), makeIntArrayValue(legal)};
        bool ok = false;
        Value r = callJs(heuristic, ev::undefined(), std::span<const Value>(args, 2), &ok);
        if (!ok || ev::isObject(r)) return -1;
        return static_cast<int>(ev::toDouble(r));
    };

    grid::BCConfig cfg;
    cfg.min_return = static_cast<float>(getDoubleProperty(opts.get(), "minReturn", cfg.min_return));
    cfg.rollout_horizon = getIntProp(opts.get(), "rolloutHorizon", cfg.rollout_horizon);
    cfg.gamma = static_cast<float>(getDoubleProperty(opts.get(), "gamma", cfg.gamma));
    cfg.clip_value = getBoolProperty(opts.get(), "clipValue", cfg.clip_value);

    std::vector<std::any> starts;
    Value startsV = ev::getProperty(opts.get(), "starts");
    if (ev::isObject(startsV)) {
        ev::Persistent arr(startsV);
        Value lenV = ev::getProperty(arr.get(), "length");
        const uint32_t n = (ev::isUndefined(lenV) || ev::isObject(lenV))
                               ? 0u
                               : static_cast<uint32_t>(ev::toDouble(lenV));
        starts.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            starts.push_back(std::any{JsSnapshot(ev::getElement(arr.get(), i))});
        }
    }

    auto sits = grid::generate_bc_situations(env, policy, starts, cfg);
    return hostArrayOf(sits.size(),
                       [&](size_t i) { return makeGenericSituationObject(sits[i]); });
}

} // namespace

// ---------------------------------------------------------------------------
// Shared readers
// ---------------------------------------------------------------------------

std::vector<float> readFloats(Value v) {
    std::vector<float> out;
    readFloatVector(v, out);
    return out;
}

std::string readStringPropOr(Value obj, const char* key, const char* def) {
    if (!ev::isObject(obj)) return def ? def : "";
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    if (!ev::isString(v)) return def ? def : "";
    return ev::toUtf8(v);
}

std::vector<grid::FailureStep> readFailureTail(Value arr) {
    std::vector<grid::FailureStep> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent root(arr);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return out;
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(root.get(), i);
        if (!ev::isObject(e)) continue;
        ev::Persistent eo(e);
        grid::FailureStep s;
        s.sig = readStringPropOr(eo.get(), "sig", "");
        s.action = getIntProp(eo.get(), "action", -1);
        out.push_back(std::move(s));
    }
    return out;
}

learn::GenericSituation situationFromValue(Value v) {
    learn::GenericSituation s;
    if (!ev::isObject(v)) return s;
    ev::Persistent root(v);
    s.obs = readFloats(ev::getProperty(root.get(), "obs"));
    s.policy_target = readFloats(ev::getProperty(root.get(), "policyTarget"));
    s.action_mask = readFloats(ev::getProperty(root.get(), "actionMask"));
    s.value_target = static_cast<float>(getDoubleProperty(root.get(), "valueTarget", 0.0));
    return s;
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

void ensureAIGridClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_obsWindowClass.init("AIGridObsWindow", decorateObsWindow);
    g_frameStackClass.init("AIGridFrameStack", decorateFrameStack);
    g_failureTapeClass.init("AIGridFailureTape", decorateFailureTape);
    g_bestCropClass.init("AIGridBestCrop", decorateBestCrop);
    g_shaperClass.init("AIGridPotentialShaper", decorateShaper);
    g_stallDetectorClass.init("AIGridStallDetector", decorateStallDetector);
}

void installAIGridCore(ObjectBuilder& gridNs) {
    ensureAIGridClassesInstalled();

    gridNs.def("createObsWindow", 1, [](Value, std::span<const Value> a) -> Value {
        return createObsWindow(a);
    });

    gridNs.def("createFrameStack", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isObject(a[0])) {
            return ev::throwTypeError("createFrameStack(opts): expected object");
        }
        const int innerDim = getIntProp(a[0], "innerDim", 0);
        const int k = getIntProp(a[0], "k", 1);
        if (innerDim <= 0 || k <= 0) {
            return ev::throwTypeError("createFrameStack: innerDim and k must be > 0");
        }
        auto cell = std::make_unique<HostFrameStack>();
        cell->fs = std::make_unique<grid::FrameStack>(innerDim, k);
        return g_frameStackClass.createInstance(std::move(cell));
    });

    gridNs.def("createFailureTape", 1, [](Value, std::span<const Value> a) -> Value {
        grid::FailureTapeConfig cfg;
        if (!a.empty() && ev::isObject(a[0])) {
            cfg.tape_depth = getIntProp(a[0], "tapeDepth", cfg.tape_depth);
            cfg.ring_capacity = getIntProp(a[0], "ringCapacity", cfg.ring_capacity);
            cfg.penalty = static_cast<float>(getDoubleProperty(a[0], "penalty", cfg.penalty));
            cfg.floor = static_cast<float>(getDoubleProperty(a[0], "floor", cfg.floor));
        }
        auto cell = std::make_unique<HostFailureTape>();
        cell->tape = std::make_unique<grid::FailureTape>(cfg);
        return g_failureTapeClass.createInstance(std::move(cell));
    });

    gridNs.def("createBestCrop", 1, [](Value, std::span<const Value> a) -> Value {
        grid::BestCropConfig cfg;
        uint64_t seed = 0xC0DE1234ULL;
        if (!a.empty() && ev::isObject(a[0])) {
            cfg.capacity = getIntProp(a[0], "capacity", cfg.capacity);
            cfg.depth_bonus =
                static_cast<float>(getDoubleProperty(a[0], "depthBonus", cfg.depth_bonus));
            cfg.age_decay = static_cast<float>(getDoubleProperty(a[0], "ageDecay", cfg.age_decay));
            cfg.seed_top_k = getIntProp(a[0], "seedTopK", cfg.seed_top_k);
            seed = getU64Property(a[0], "seed", seed);
        }
        auto cell = std::make_unique<HostBestCrop>();
        cell->pool = std::make_unique<grid::BestCrop>(cfg);
        cell->rng.seed(seed);
        return g_bestCropClass.createInstance(std::move(cell));
    });

    gridNs.def("createPotentialShaper", 1, [](Value, std::span<const Value> a) -> Value {
        float gamma = 0.99f;
        if (!a.empty() && ev::isObject(a[0])) {
            gamma = static_cast<float>(getDoubleProperty(a[0], "gamma", 0.99));
        }
        auto cell = std::make_unique<HostShaper>();
        cell->sh = std::make_unique<grid::PotentialShaper>(gamma);
        return g_shaperClass.createInstance(std::move(cell));
    });

    gridNs.def("createStallDetector", 1, [](Value, std::span<const Value> a) -> Value {
        float eps = 0.0f;
        int patience = 60;
        if (!a.empty() && ev::isObject(a[0])) {
            eps = static_cast<float>(getDoubleProperty(a[0], "epsilon", 0.0));
            patience = getIntProp(a[0], "patience", 60);
        }
        auto cell = std::make_unique<HostStallDetector>();
        cell->det = std::make_unique<grid::StallDetector>(eps, patience);
        return g_stallDetectorClass.createInstance(std::move(cell));
    });

    gridNs.def("generateBC", 1, [](Value, std::span<const Value> a) -> Value {
        return generateBC(a);
    });
}

void installAIGrid(ObjectBuilder& game) {
    ObjectBuilder gridNs;
    installAIGridCore(gridNs);
    installAIGridRecording(gridNs);
    game.set("grid", gridNs.get());
}

} // namespace brogameagent::api

#else  // !BROGAMEAGENT_HAS_NN

#include "host_ai_mcts_shared.h"

namespace brogameagent::api {

void installAIGrid(ObjectBuilder& game) {
    ObjectBuilder gridNs;
    gridNs.set("available", false);
    game.set("grid", gridNs.get());
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
