// `bro.ai.game` for the compiled realm: the ORCA World, the HexNav navigator,
// and the game object that hangs the factories and the perception helpers
// together.

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

// ---------------------------------------------------------------------------
// HexNav Wrapper
// ---------------------------------------------------------------------------

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
        // The radius sizes a disk of 3r(r+1)+1 offsets; past the grid's own
        // extent (size <= 4096) it only grows the loop.
        const int32_t radius = static_cast<int32_t>(intAt(a, 1, 0, 8192, "buildClearance: radius"));
        const int32_t crush = i32At(a, 5, "buildClearance: crushFloors");
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

    b.def("components", 2, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id, clr;
        if (!h || !h->nav || !idAt(a, 0, id)) return ev::throwTypeError("components(id, clearanceId?)");
        if (a.size() > 1 && !ev::isUndefined(a[1]) && !ev::isNull(a[1]) && !idAt(a, 1, clr)) clr.clear();
        const std::vector<int32_t>& labels = h->nav->components(id, clr);
        return makeInt32Array(labels.data(), labels.size());
    });

    decorateHexNavExtras(b);  // field() — host_ai_world_extra.cpp
}

Value aiCreateHexNav(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) return ev::throwTypeError("createHexNav({ size }) requires options");
    ev::Persistent opts(a[0]);
    const int32_t size = getI32Property(opts.get(), "size", 0, 1, 4096);
    if (size <= 0) return ev::throwRangeError("createHexNav: size must be 1..4096");
    auto* h = new HostHexNav();
    h->nav = std::make_unique<brogameagent::HexNav>(size);
    return g_hexNavClass.make(h, [](void* p) { delete static_cast<HostHexNav*>(p); });
}

// ---------------------------------------------------------------------------
// World Wrapper
// ---------------------------------------------------------------------------

Value worldAgentValue(Value worldSelf, const brogameagent::Agent* agent) {
    HostWorld* w = unwrapWorld(worldSelf);
    if (!w) return ev::undefined();
    const HostWorld::Roster* r = w->rosterEntry(agent);
    if (!r) return ev::undefined();
    const uint32_t key = r->key;
    ev::Persistent agents(ev::getProperty(worldSelf, "_agents"));
    if (!ev::isObject(agents.get())) return ev::undefined();
    Value v = ev::getElement(agents.get(), key);
    // `_agents` is reachable from JS; answer only the wrapper of this agent.
    HostAgent* h = unwrapAgent(v);
    return (h && &h->agent == agent) ? v : ev::undefined();
}

namespace {

/// `self._agents`, created on first use. `self` must be rooted by the caller.
Value worldAgentsTable(ev::Persistent& self) {
    Value t = ev::getProperty(self.get(), "_agents");
    if (ev::isObject(t)) return t;
    ev::Persistent fresh(ev::createObject());
    self.set(ev::setProperty(self.get(), "_agents", fresh.get()));
    return fresh.get();
}

/// Wrappers for `agents` (those on the roster), in order.
Value worldAgentArray(Value self, const std::vector<const brogameagent::Agent*>& agents) {
    ev::Persistent selfP(self);
    std::vector<const brogameagent::Agent*> onRoster;
    HostWorld* w = unwrapWorld(selfP.get());
    for (const auto* a : agents) {
        if (w && w->rosterEntry(a)) onRoster.push_back(a);
    }
    return hostArrayOf(onRoster.size(),
                       [&](size_t i) { return worldAgentValue(selfP.get(), onRoster[i]); });
}

} // namespace

void decorateWorldProto(ObjectBuilder& b) {
    b.def("addAgent", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::undefined();
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::undefined();
        w->world.addAgent(&ag->agent);
        if (w->rosterEntry(&ag->agent)) return ev::undefined();
        ev::Persistent selfP(self), agentP(a[0]);
        const uint32_t key = w->nextRosterKey++;
        ev::Persistent table(worldAgentsTable(selfP));
        ev::setElement(table.get(), key, agentP.get());
        w->roster.push_back({&ag->agent, key});
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
                const uint32_t key = w->roster[i].key;
                w->roster.erase(w->roster.begin() + static_cast<std::ptrdiff_t>(i));
                ev::Persistent selfP(self);
                ev::Persistent table(ev::getProperty(selfP.get(), "_agents"));
                if (ev::isObject(table.get())) ev::deleteProperty(table.get(), std::to_string(key));
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
        ActiveWorldScope scope(w, self);
        w->world.tick(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("step", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return ev::undefined();
        ActiveWorldScope scope(w, self);
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
        Value v = worldAgentValue(self, enemy);
        return ev::isUndefined(v) ? ev::null() : v;
    });

    b.def("enemiesInRange", 2, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return hostArrayOf(0, [](size_t) { return ev::null(); });
        float range = static_cast<float>(numAt(a, 1));
        auto enemies = w->world.enemiesInRange(ag->agent, range);
        std::vector<const brogameagent::Agent*> found(enemies.begin(), enemies.end());
        return worldAgentArray(self, found);
    });

    b.def("alliesInRange", 2, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 2) return hostArrayOf(0, [](size_t) { return ev::null(); });
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return hostArrayOf(0, [](size_t) { return ev::null(); });
        float range = static_cast<float>(numAt(a, 1));
        float rangeSq = range * range;
        auto allies = w->world.alliesOf(ag->agent);
        std::vector<const brogameagent::Agent*> found;
        for (auto* al : allies) {
            float dx = al->x() - ag->agent.x();
            float dz = al->z() - ag->agent.z();
            if (dx * dx + dz * dz <= rangeSq) found.push_back(al);
        }
        return worldAgentArray(self, found);
    });

    b.accessor("damageEvents", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        const auto& evts = w->world.events();
        return hostArrayOf(evts.size(), [&](size_t i) {
            const auto& e = evts[i];
            ObjectBuilder obj;
            obj.set("sourceId", ev::fromDouble(e.attackerId));
            obj.set("attackerId", ev::fromDouble(e.attackerId));
            obj.set("targetId", ev::fromDouble(e.targetId));
            obj.set("amount", ev::fromDouble(e.amount));
            obj.set("kind", ev::fromUtf8(damageKindStr(e.kind)));
            obj.set("killed", ev::fromBool(e.killed));
            return obj.get();
        });
    }, nullptr);

    b.accessor("events", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        const auto& evs = w->world.events();
        return hostArrayOf(evs.size(), [&](size_t i) {
            const auto& e = evs[i];
            ObjectBuilder obj;
            obj.set("sourceId", ev::fromDouble(e.attackerId));
            obj.set("attackerId", ev::fromDouble(e.attackerId));
            obj.set("targetId", ev::fromDouble(e.targetId));
            obj.set("amount", ev::fromDouble(e.amount));
            obj.set("kind", ev::fromUtf8(damageKindStr(e.kind)));
            obj.set("killed", ev::fromBool(e.killed));
            return obj.get();
        });
    }, nullptr);

    b.def("resolveAttack", 2, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 2) return ev::fromBool(false);
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::fromBool(false);
        int targetId = i32At(a, 1, "resolveAttack: targetId");
        return ev::fromBool(w->world.resolveAttack(ag->agent, targetId));
    });

    b.def("resolveAbility", 3, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 3) return ev::fromBool(false);
        HostAgent* ag = unwrapAgent(a[0]);
        if (!ag) return ev::fromBool(false);
        ActiveWorldScope scope(w, self);
        int slot = i32At(a, 1, "resolveAbility: slot");
        int targetId = i32At(a, 2, "resolveAbility: targetId");
        return ev::fromBool(w->world.resolveAbility(ag->agent, slot, targetId));
    });

    b.def("dealDamage", 4, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 3) return ev::fromDouble(0.0);
        HostAgent* att = unwrapAgent(a[0]);
        HostAgent* tgt = unwrapAgent(a[1]);
        if (!att || !tgt) return ev::fromDouble(0.0);
        float amount = static_cast<float>(numAt(a, 2));
        std::string kindStr = "physical";
        if (a.size() >= 4 && ev::isString(a[3])) kindStr = ev::toUtf8(a[3]);
        float dealt = w->world.dealDamage(att->agent, tgt->agent, amount, parseDamageKind(kindStr.c_str()));
        return ev::fromDouble(dealt);
    });

    b.def("clearEvents", 0, [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (w) w->world.clearEvents();
        return ev::undefined();
    });

    b.def("spawnProjectile", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty() || !ev::isObject(a[0])) return ev::fromDouble(-1);
        ev::Persistent opts(a[0]);
        brogameagent::Projectile p;
        p.ownerId = getI32Property(opts.get(), "ownerId", -1);
        p.teamId = getI32Property(opts.get(), "teamId", 0);
        p.targetId = getI32Property(opts.get(), "targetId", -1);
        p.x = static_cast<float>(getDoubleProperty(opts.get(), "x", 0));
        p.z = static_cast<float>(getDoubleProperty(opts.get(), "z", 0));
        p.vx = static_cast<float>(getDoubleProperty(opts.get(), "vx", 0));
        p.vz = static_cast<float>(getDoubleProperty(opts.get(), "vz", 0));
        p.speed = static_cast<float>(getDoubleProperty(opts.get(), "speed", 20));
        p.radius = static_cast<float>(getDoubleProperty(opts.get(), "radius", 0.3));
        p.damage = static_cast<float>(getDoubleProperty(opts.get(), "damage", 0));
        p.remainingLife = static_cast<float>(getDoubleProperty(opts.get(), "remainingLife", 2));
        p.splashRadius = static_cast<float>(getDoubleProperty(opts.get(), "splashRadius", 0));
        p.maxHits = getI32Property(opts.get(), "maxHits", 0, 0);

        Value kindVal = ev::getProperty(opts.get(), "kind");
        if (ev::isString(kindVal)) p.kind = parseDamageKind(ev::toUtf8(kindVal).c_str());

        Value modeVal = ev::getProperty(opts.get(), "mode");
        if (ev::isString(modeVal)) {
            std::string m = ev::toUtf8(modeVal);
            if (m == "pierce") p.mode = brogameagent::ProjectileMode::Pierce;
            else if (m == "aoe") p.mode = brogameagent::ProjectileMode::AoE;
        }

        int id = w->world.spawnProjectile(p);
        return ev::fromDouble(id);
    });

    b.accessor("projectiles", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return hostArrayOf(0, [](size_t) { return ev::null(); });
        const auto& projs = w->world.projectiles();
        std::vector<const brogameagent::Projectile*> alive;
        for (const auto& p : projs) if (p.alive) alive.push_back(&p);
        return hostArrayOf(alive.size(), [&](size_t i) {
            const auto& p = *alive[i];
            ObjectBuilder obj;
            obj.set("id", ev::fromDouble(p.id));
            obj.set("ownerId", ev::fromDouble(p.ownerId));
            obj.set("teamId", ev::fromDouble(p.teamId));
            obj.set("x", ev::fromDouble(p.x));
            obj.set("z", ev::fromDouble(p.z));
            obj.set("vx", ev::fromDouble(p.vx));
            obj.set("vz", ev::fromDouble(p.vz));
            obj.set("speed", ev::fromDouble(p.speed));
            obj.set("damage", ev::fromDouble(p.damage));
            obj.set("alive", ev::fromBool(p.alive));
            const char* modeStr = "single";
            if (p.mode == brogameagent::ProjectileMode::Pierce) modeStr = "pierce";
            else if (p.mode == brogameagent::ProjectileMode::AoE) modeStr = "aoe";
            obj.set("mode", ev::fromUtf8(modeStr));
            return obj.get();
        });
    }, nullptr);

    b.def("snapshot", 0, [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w) return ev::null();
        auto snap = w->world.snapshot();
        ObjectBuilder obj;
        Value agentsArr = hostArrayOf(snap.agents.size(), [&](size_t i) {
            const auto& a = snap.agents[i];
            ObjectBuilder ao;
            ao.set("id", ev::fromDouble(a.id));
            ao.set("x", ev::fromDouble(a.x));
            ao.set("z", ev::fromDouble(a.z));
            ao.set("vx", ev::fromDouble(a.vx));
            ao.set("vz", ev::fromDouble(a.vz));
            ao.set("yaw", ev::fromDouble(a.yaw));
            ao.set("aimYaw", ev::fromDouble(a.aimYaw));
            ao.set("aimPitch", ev::fromDouble(a.aimPitch));
            ao.set("speed", ev::fromDouble(a.speed));
            ao.set("radius", ev::fromDouble(a.radius));
            ao.set("hp", ev::fromDouble(a.unit.hp));
            ao.set("maxHp", ev::fromDouble(a.unit.maxHp));
            ao.set("mana", ev::fromDouble(a.unit.mana));
            ao.set("teamId", ev::fromDouble(a.unit.teamId));
            ao.set("hasTarget", ev::fromBool(a.hasTarget));
            ao.set("targetX", ev::fromDouble(a.targetX));
            ao.set("targetZ", ev::fromDouble(a.targetZ));
            return ao.get();
        });
        obj.set("agents", agentsArr);
        obj.set("nextProjectileId", ev::fromDouble(snap.nextProjectileId));
        return obj.get();
    });

    b.def("restore", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        ev::Persistent root(a[0]);
        brogameagent::WorldSnapshot snap;
        ev::Persistent agentsArr(ev::getProperty(root.get(), "agents"));
        if (ev::isObject(agentsArr.get())) {
            Value lenV = ev::getProperty(agentsArr.get(), "length");
            if (ev::isNumber(lenV)) {
                const uint32_t n = toLength(lenV);
                for (uint32_t i = 0; i < n; i++) {
                    // Rooted: every getDoubleProperty below allocates.
                    ev::Persistent aoP(ev::getElement(agentsArr.get(), static_cast<uint32_t>(i)));
                    if (ev::isObject(aoP.get())) {
                        brogameagent::AgentSnapshot as;
                        as.id = getI32Property(aoP.get(), "id", 0);
                        as.x = static_cast<float>(getDoubleProperty(aoP.get(),"x", 0));
                        as.z = static_cast<float>(getDoubleProperty(aoP.get(),"z", 0));
                        as.vx = static_cast<float>(getDoubleProperty(aoP.get(),"vx", 0));
                        as.vz = static_cast<float>(getDoubleProperty(aoP.get(),"vz", 0));
                        as.yaw = static_cast<float>(getDoubleProperty(aoP.get(),"yaw", 0));
                        as.aimYaw = static_cast<float>(getDoubleProperty(aoP.get(),"aimYaw", 0));
                        as.aimPitch = static_cast<float>(getDoubleProperty(aoP.get(),"aimPitch", 0));
                        as.speed = static_cast<float>(getDoubleProperty(aoP.get(),"speed", 6));
                        as.radius = static_cast<float>(getDoubleProperty(aoP.get(),"radius", 0.4));
                        as.unit.hp = static_cast<float>(getDoubleProperty(aoP.get(),"hp", 100));
                        as.unit.maxHp = static_cast<float>(getDoubleProperty(aoP.get(),"maxHp", 100));
                        as.unit.mana = static_cast<float>(getDoubleProperty(aoP.get(),"mana", 0));
                        as.unit.teamId = getI32Property(aoP.get(), "teamId", 0);
                        as.unit.id = as.id;
                        as.hasTarget = getBoolProperty(aoP.get(),"hasTarget", false);
                        as.targetX = static_cast<float>(getDoubleProperty(aoP.get(),"targetX", 0));
                        as.targetZ = static_cast<float>(getDoubleProperty(aoP.get(),"targetZ", 0));
                        snap.agents.push_back(as);
                    }
                }
            }
        }
        snap.nextProjectileId = getI32Property(root.get(), "nextProjectileId", 1);
        w->world.restore(snap);
        return ev::undefined();
    });

    b.accessor("agentCount", [](Value self, std::span<const Value>) -> Value {
        HostWorld* w = unwrapWorld(self);
        return ev::fromDouble(w ? static_cast<double>(w->world.agents().size()) : 0.0);
    }, nullptr);

    // findById() / registerAbility() / seed() — host_ai_world_extra.cpp
    decorateWorldExtras(b);
}

Value aiCreateWorld(Value, std::span<const Value>) {
    auto* w = new HostWorld();
    return g_worldClass.make(w, [](void* p) { delete static_cast<HostWorld*>(p); },
                             ev::Finalize::Deferred);
}

// ---------------------------------------------------------------------------
// Perception Helpers
// ---------------------------------------------------------------------------

Value aiCanSee(Value, std::span<const Value> a) {
    if (a.size() < 8) return ev::fromBool(false);
    std::vector<brogameagent::AABB> boxes = parseAABBArray(a[7]);
    const bool r = brogameagent::canSee(
        {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))},
        {static_cast<float>(numAt(a, 2)), static_cast<float>(numAt(a, 3))},
        static_cast<float>(numAt(a, 4)), static_cast<float>(numAt(a, 5)),
        static_cast<float>(numAt(a, 6)), boxes.data(), static_cast<int>(boxes.size()));
    return ev::fromBool(r);
}

Value aiComputeLeadAim(Value, std::span<const Value> a) {
    if (a.size() < 10) return ev::null();
    float v[10];
    for (size_t i = 0; i < 10; ++i) v[i] = static_cast<float>(numAt(a, i));
    brogameagent::LeadAimResult r =
        brogameagent::computeLeadAim(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
    ObjectBuilder o;
    o.set("yaw", ev::fromDouble(r.aim.yaw));
    o.set("pitch", ev::fromDouble(r.aim.pitch));
    o.set("valid", ev::fromBool(r.valid));
    o.set("timeToHit", ev::fromDouble(r.timeToHit));
    return o.get();
}

Value gameComputeAim(Value, std::span<const Value> a) {
    if (a.empty()) return ev::null();
    float ox = 0, oy = 0, oz = 0, tx = 0, ty = 0, tz = 0;
    if (a.size() >= 6 && !ev::isObject(a[0]) && !ev::isObject(a[1])) {
        ox = static_cast<float>(numAt(a, 0)); oy = static_cast<float>(numAt(a, 1)); oz = static_cast<float>(numAt(a, 2));
        tx = static_cast<float>(numAt(a, 3)); ty = static_cast<float>(numAt(a, 4)); tz = static_cast<float>(numAt(a, 5));
    } else if (a.size() >= 2) {
        auto p1 = parseVec3(a[0]), p2 = parseVec3(a[1]);
        ox = p1.x; oy = p1.y; oz = p1.z; tx = p2.x; ty = p2.y; tz = p2.z;
    } else {
        return ev::null();
    }
    brogameagent::AimResult aim = brogameagent::computeAim(ox, oy, oz, tx, ty, tz);
    ObjectBuilder o;
    o.set("valid", ev::fromBool(true));
    o.set("yaw", ev::fromDouble(aim.yaw));
    o.set("pitch", ev::fromDouble(aim.pitch));
    return o.get();
}

// ---------------------------------------------------------------------------
// Class Registration
// ---------------------------------------------------------------------------

void ensureAIClassesInstalled() {
    // Once per THREAD: a class's constructor and prototype are the
    // installing thread's (host_class.h), so a Worker realm installs its own.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_navGridClass.install("AINavGrid", 0, nullptr, decorateNavGridProto);
    g_navMeshClass.install("AINavMesh", 0, nullptr, decorateNavMeshProto);
    g_agentClass.install("AIAgent", 0, nullptr, decorateAgentProto);
    g_agentBindingClass.install("AIAgentBinding", 0, nullptr, decorateAgentBindingProto);
    g_unitClass.install("AIUnit", 0, nullptr, decorateUnitProto);
    g_hexNavClass.install("AIHexNav", 0, nullptr, decorateHexNavProto);
    g_worldClass.install("AIWorld", 0, nullptr, decorateWorldProto);
    ensureAIMctsClassesInstalled();
    ensureAIExtrasClassesInstalled();
}

// ---------------------------------------------------------------------------
// Namespace Mounting
// ---------------------------------------------------------------------------

Value makeAiGameValue() {
    ensureAIClassesInstalled();
    ObjectBuilder b;
    b.def("createNavGrid", 1, aiCreateNavGrid);
    b.def("createHexNav", 1, aiCreateHexNav);
    b.def("bakeNavMesh", 1, aiBakeNavMesh);
    b.def("loadNavMesh", 1, aiLoadNavMesh);
#if BROGAMEAGENT_HAS_NAVMESH
    b.set("navMeshAvailable", ev::fromBool(true));
#else
    b.set("navMeshAvailable", ev::fromBool(false));
#endif
    b.def("createAgent", 1, aiCreateAgent);
    b.def("createWorld", 0, aiCreateWorld);
    b.def("hasLineOfSight", 5, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 5) return ev::fromBool(false);
        float fx = static_cast<float>(numAt(a, 0)), fz = static_cast<float>(numAt(a, 1));
        float tx = static_cast<float>(numAt(a, 2)), tz = static_cast<float>(numAt(a, 3));
        if (auto* ng = unwrapNavGrid(a[4])) return ev::fromBool(ng->grid && ng->grid->hasGridLOS({fx, fz}, {tx, tz}));
        auto boxes = parseAABBArray(a[4]);
        return ev::fromBool(brogameagent::hasLineOfSight({fx, fz}, {tx, tz}, boxes.data(), static_cast<int>(boxes.size())));
    });
    b.def("canSee", 8, aiCanSee);
    b.def("computeAim", 6, gameComputeAim);
    b.def("computeLeadAim", 10, aiComputeLeadAim);
    installAIExtras(b);
    installAIMcts(b);
    installRegisterCapability(b);
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
