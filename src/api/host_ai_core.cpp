#include "host_ai_internal.h"
#include <limits>

namespace brogameagent::api {

// ---------------------------------------------------------------------------
// Host Classes Instantiation
// ---------------------------------------------------------------------------

HostClass g_navGridClass;
HostClass g_navMeshClass;
HostClass g_agentClass;
HostClass g_agentBindingClass;
HostClass g_hexNavClass;
HostClass g_worldClass;
HostClass g_genericMctsClass;
HostClass g_mctsClass;
HostClass g_decoupledMctsClass;
HostClass g_teamMctsClass;
HostClass g_optionClass;
HostClass g_optionMctsClass;

// ---------------------------------------------------------------------------
// HexNav Wrapper
// ---------------------------------------------------------------------------

namespace {

bool idAt(std::span<const Value> a, size_t i, std::string& out) {
    Value v = argAt(a, i);
    if (ev::isObject(v) || ev::isSymbol(v)) return false;
    if (ev::isUndefined(v)) { out = "undefined"; return true; }
    out = ev::toUtf8(v);
    return true;
}

double costAt(std::span<const Value> a, size_t i) {
    if (!hasArg(a, i)) return std::numeric_limits<double>::infinity();
    Value v = a[i];
    if (ev::isObject(v)) return std::numeric_limits<double>::infinity();
    return ev::toDouble(v);
}

template <typename T>
const T* viewAt(std::span<const Value> a, size_t i, bronze::ElementKind kind, size_t& count) {
    count = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(argAt(a, i));
    if (!info || info.elementKind != kind) return nullptr;
    count = info.elementCount;
    return reinterpret_cast<const T*>(info.data);
}

const double* doublesAt(std::span<const Value> a, size_t i, size_t& count, std::vector<double>& tmp) {
    count = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(argAt(a, i));
    if (!info) return nullptr;
    if (info.elementKind == ev::elements::Float64) {
        count = info.elementCount;
        return reinterpret_cast<const double*>(info.data);
    }
    if (info.elementKind == ev::elements::Float32) {
        const float* f = reinterpret_cast<const float*>(info.data);
        tmp.assign(f, f + info.elementCount);
        count = tmp.size();
        return tmp.data();
    }
    return nullptr;
}

Value typedArrayOf(bronze::ElementKind kind, const void* data, size_t count, size_t elemSize) {
    ev::Persistent view(ev::createTypedArray(kind, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    ev::fillTypedArray(view.get(), std::span<const uint8_t>(
                                       static_cast<const uint8_t*>(data), count * elemSize));
    return view.get();
}

} // namespace

void decorateHexNavProto(ObjectBuilder& b) {
    b.accessor("size", [](Value self, std::span<const Value>) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        return ev::fromDouble(h && h->nav ? h->nav->size() : 0);
    }, nullptr);

    b.def("setStepCosts", 2, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("setStepCosts(id, costs)");
        size_t n = 0;
        std::vector<double> tmp;
        const double* t = doublesAt(a, 1, n, tmp);
        if (!t) return ev::throwTypeError("setStepCosts: costs must be Float64Array or Float32Array");
        if (!h->nav->setStepCosts(id, t, n))
            return ev::throwRangeError("setStepCosts: costs must hold size*size*6 entries");
        return ev::fromBool(true);
    });

    b.def("updateStepCosts", 3, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("updateStepCosts(id, cells, values)");
        size_t nc = 0, nv = 0;
        std::vector<double> tmp;
        const double* vals = doublesAt(a, 2, nv, tmp);
        const int32_t* cells = viewAt<int32_t>(a, 1, ev::elements::Int32, nc);
        if (!cells || !vals || nv != nc * 6)
            return ev::throwTypeError("updateStepCosts: cells Int32Array, values Float64Array (6 per cell)");
        return ev::fromBool(h->nav->updateStepCosts(id, cells, nc, vals));
    });

    b.def("hasStepCosts", 1, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::fromBool(false);
        return ev::fromBool(h->nav->hasStepCosts(id));
    });

    b.def("setClearance", 2, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("setClearance(id, table)");
        size_t n = 0;
        const uint8_t* t = viewAt<uint8_t>(a, 1, ev::elements::Uint8, n);
        if (!t || !h->nav->setClearance(id, t, n))
            return ev::throwRangeError("setClearance: table must be Uint8Array of size*size entries");
        return ev::fromBool(true);
    });

    b.def("buildClearance", 6, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id))
            return ev::throwTypeError("buildClearance(id, radius, passable, elevation, floors, crushFloors)");
        const size_t cells = static_cast<size_t>(h->nav->cells());
        const int32_t radius = i32At(a, 1);
        const int32_t crush = i32At(a, 5);
        size_t np = 0, ne = 0, nf = 0;
        const uint8_t* passable = viewAt<uint8_t>(a, 2, ev::elements::Uint8, np);
        const int8_t* elevation = viewAt<int8_t>(a, 3, ev::elements::Int8, ne);
        const int16_t* floors = viewAt<int16_t>(a, 4, ev::elements::Int16, nf);
        if (!passable || !elevation || !floors || np != cells || ne != cells || nf != cells)
            return ev::throwTypeError("buildClearance: passable, elevation and floors must each hold size*size entries");
        const std::vector<uint8_t>& out =
            h->nav->buildClearance(id, radius, passable, elevation, floors, crush);
        return typedArrayOf(ev::elements::Uint8, out.data(), out.size(), 1);
    });

    b.def("hasClearance", 1, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::fromBool(false);
        return ev::fromBool(h->nav->hasClearance(id));
    });

    b.def("findPath", 6, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("findPath(id, x0, y0, x1, y1, maxCost?)");
        std::vector<int32_t> path;
        const bool ok = h->nav->findPath(id, i32At(a, 1), i32At(a, 2), i32At(a, 3), i32At(a, 4),
                                         costAt(a, 5), path);
        if (!ok) return ev::null();
        return makeInt32Array(path.data(), path.size());
    });

    b.def("findPathRadius", 7, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id, clr;
        if (!h || !h->nav || !idAt(a, 0, id) || !idAt(a, 1, clr))
            return ev::throwTypeError("findPathRadius(id, clearanceId, x0, y0, x1, y1, maxCost?)");
        std::vector<int32_t> path;
        const bool ok = h->nav->findPathRadius(id, clr, i32At(a, 2), i32At(a, 3), i32At(a, 4),
                                               i32At(a, 5), costAt(a, 6), path);
        if (!ok) return ev::null();
        return makeInt32Array(path.data(), path.size());
    });

    b.def("movementField", 4, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("movementField(id, x0, y0, maxCost?)");
        std::vector<float> cost;
        std::vector<int32_t> parent;
        if (!h->nav->movementField(id, i32At(a, 1), i32At(a, 2), costAt(a, 3), cost, parent))
            return ev::null();
        ObjectBuilder o;
        o.set("cost", makeFloat32Array(cost.data(), cost.size()));
        o.set("parent", makeInt32Array(parent.data(), parent.size()));
        return o.get();
    });
}

Value aiCreateHexNav(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createHexNav({ size }) requires options");
    ev::Persistent opts(a[0]);
    int32_t size = static_cast<int32_t>(getDoubleProperty(opts.get(), "size", 0));
    if (size <= 0 || size > 4096) return ev::throwRangeError("createHexNav: size must be 1..4096");
    auto* h = new HostHexNav();
    h->nav = std::make_unique<brogameagent::HexNav>(size);
    return g_hexNavClass.make(h, [](void* p) { delete static_cast<HostHexNav*>(p); });
}

// ---------------------------------------------------------------------------
// World Wrapper
// ---------------------------------------------------------------------------

void decorateWorldProto(ObjectBuilder& b) {
    b.def("addAgent", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::undefined();
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::undefined();
        w->world.addAgent(&ag->agent);
        w->roster.push_back({&ag->agent, ev::Persistent(a[0])});
        return ev::undefined();
    });

    b.def("removeAgent", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::undefined();
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::undefined();
        w->world.removeAgent(&ag->agent);
        for (size_t i = 0; i < w->roster.size(); ++i) {
            if (w->roster[i].agent == &ag->agent) {
                w->roster.erase(w->roster.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        return ev::undefined();
    });

    b.def("addObstacle", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        w->world.addObstacle(parseAABB(a[0]));
        return ev::undefined();
    });

    b.def("tick", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return ev::undefined();
        w->world.tick(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("step", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return ev::undefined();
        w->world.tick(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setAvoidance", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::undefined();
        Value v = a[0];
        if (ev::isBool(v)) {
            w->world.setAvoidanceEnabled(ev::toBool(v));
            return ev::undefined();
        }
        if (!ev::isObject(v)) return ev::throwTypeError("setAvoidance(bool | {enabled?, navGrid?})");
        ev::Persistent opts(v);
        const bool enabled = getBoolProperty(opts.get(), "enabled", true);
        Value gv = ev::getProperty(opts.get(), "navGrid");
        if (ev::isObject(gv)) {
            HostNavGrid* ng = unwrapNavGrid(gv);
            if (!ng || !ng->grid)
                return ev::throwTypeError("setAvoidance: navGrid must be a createNavGrid() object");
            w->world.clearAvoidanceObstacles();
            for (const brogameagent::AABB& box : ng->grid->obstacles())
                w->world.addAvoidanceObstacle(box);
        }
        w->world.setAvoidanceEnabled(enabled);
        return ev::undefined();
    });

    b.accessor("avoidanceEnabled", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        return ev::fromBool(w && w->world.avoidanceEnabled());
    }, nullptr);

    b.def("nearestEnemy", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::null();
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::null();
        auto* enemy = w->world.nearestEnemy(ag->agent);
        if (!enemy) return ev::null();
        for (const auto& r : w->roster) {
            if (r.agent == enemy) return r.value.get();
        }
        return ev::null();
    });

    b.def("enemiesInRange", 2, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return hostArrayOf(0, [](size_t) { return ev::null(); });
        float range = static_cast<float>(numAt(a, 1));
        auto enemies = w->world.enemiesInRange(ag->agent, range);
        std::vector<Value> out;
        for (auto* enemy : enemies) {
            for (const auto& r : w->roster) {
                if (r.agent == enemy) {
                    out.push_back(r.value.get());
                    break;
                }
            }
        }
        return hostArrayOf(out.size(), [&](size_t i) { return out[i]; });
    });

    b.accessor("agentCount", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        return ev::fromDouble(w ? static_cast<double>(w->world.agents().size()) : 0.0);
    }, nullptr);
}

Value aiCreateWorld(Value, std::span<const Value>) {
    auto* w = new HostWorld();
    return g_worldClass.make(w, [](void* p) { delete static_cast<HostWorld*>(p); },
                             ev::Finalize::Deferred);
}

// ---------------------------------------------------------------------------
// Installation & Module Assembly
// ---------------------------------------------------------------------------

void ensureAIClassesInstalled() {
    static bool installed = false;
    if (installed) return;
    installed = true;

    g_navGridClass.install("NavGrid", 0, nullptr, decorateNavGridProto);
    g_navMeshClass.install("NavMesh", 0, nullptr, decorateNavMeshProto);
    g_agentClass.install("Agent", 0, nullptr, decorateAgentProto);
    g_agentBindingClass.install("AgentBinding", 0, nullptr, decorateAgentBindingProto);
    g_hexNavClass.install("HexNav", 0, nullptr, decorateHexNavProto);
    g_worldClass.install("World", 0, nullptr, decorateWorldProto);

    // Aliases with AI prefix
    g_navGridClass.alias("AINavGrid");
    g_navMeshClass.alias("AINavMesh");
    g_agentClass.alias("AIAgent");
    g_agentBindingClass.alias("AIAgentBinding");
    g_hexNavClass.alias("AIHexNav");
    g_worldClass.alias("AIWorld");
}

Value makeAiGameValue() {
    ObjectBuilder b;

    // Constructors
    b.set("NavGrid", g_navGridClass.constructor());
    b.set("NavMesh", g_navMeshClass.constructor());
    b.set("Agent", g_agentClass.constructor());
    b.set("AgentBinding", g_agentBindingClass.constructor());
    b.set("HexNav", g_hexNavClass.constructor());
    b.set("World", g_worldClass.constructor());
    b.set("GenericMcts", g_genericMctsClass.constructor());
    b.set("Mcts", g_mctsClass.constructor());
    b.set("DecoupledMcts", g_decoupledMctsClass.constructor());
    b.set("TeamMcts", g_teamMctsClass.constructor());

    // Navigation factories
    b.def("createNavGrid", 1, aiCreateNavGrid);
    b.def("bakeNavMesh", 1, aiBakeNavMesh);
    b.def("loadNavMesh", 1, aiLoadNavMesh);
    b.def("buildFromMesh", 3, aiBuildFromMesh);

#if BROGAMEAGENT_HAS_NAVMESH
    b.set("navMeshAvailable", ev::fromBool(true));
#else
    b.set("navMeshAvailable", ev::fromBool(false));
#endif

    // Agent & World factories
    b.def("createAgent", 1, aiCreateAgent);
    b.def("createAgentBinding", 1, aiCreateAgentBinding);
    b.def("createWorld", 0, aiCreateWorld);
    b.def("createHexNav", 1, aiCreateHexNav);

    // Perception
    installPerception(b);

    // Perception sub-object for namespace parity
    ObjectBuilder perception;
    installPerception(perception);
    b.set("perception", perception.get());

    // Steering
    ObjectBuilder steer;
    installSteering(steer);
    b.set("steer", steer.get());
    b.set("steering", steer.get());

    // MCTS
    installAIMcts(b);

    return b.get();
}

Value makeBroAiValue() {
    ObjectBuilder ai;
    {
        ev::Persistent game(makeAiGameValue());
        ai.set("game", game.get());
    }
    return ai.get();
}

} // namespace brogameagent::api
