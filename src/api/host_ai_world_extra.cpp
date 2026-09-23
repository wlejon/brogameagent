// The tail of the pre-transition `bro.ai.game` surface: HexNav.field(),
// Agent.applyAction(), and World.findById() / registerAbility() / seed().
//
// These lived in the old ai_bindings.cpp next to the methods they belong to;
// they are here rather than in host_ai_game.cpp / host_ai_agent.cpp only to
// keep those files well inside the file-size ceiling.

#include "host_ai_internal.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace brogameagent::api {

namespace {

/// The table id argument, in the loose form every other HexNav method takes:
/// anything but an object or a symbol stringifies.
bool hexIdAt(std::span<const Value> a, size_t i, std::string& out) {
    Value v = argAt(a, i);
    if (ev::isObject(v) || ev::isSymbol(v)) return false;
    if (ev::isUndefined(v)) { out = "undefined"; return true; }
    out = ev::toUtf8(v);
    return true;
}

/// An optional Uint8Array mask of exactly `cells` entries. Absent (undefined /
/// null) is fine and yields a null pointer; a present-but-wrong value fails.
/// Matches the old hexNavMask().
bool hexMask(Value v, size_t cells, const uint8_t*& out) {
    out = nullptr;
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    ev::TypedArrayInfo info = ev::typedArrayInfo(v);
    if (!info || info.elementKind != ev::elements::Uint8 || info.elementCount != cells) {
        return false;
    }
    out = info.data;
    return true;
}

Value makeFloat64Array(const double* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Float64, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(double)));
    }
    return view.get();
}

} // namespace

// ---------------------------------------------------------------------------
// HexNav.field()
// ---------------------------------------------------------------------------

void decorateHexNavExtras(ObjectBuilder& b) {
    // field(id, { seeds, blocked?, aura?, auraMult?, quantum?, ring? })
    //   → { dist: Float64Array, parent: Int32Array, pops }
    //
    // The multi-seed Dijkstra "how far is every cell from the nearest seed"
    // field, bucketed on a `ring`-slot quantised heap.
    b.def("field", 2, [](Value self, std::span<const Value> a) -> Value {
        HostHexNav* h = unwrapHexNav(self);
        std::string id;
        if (!h || !h->nav || a.size() < 2 || !ev::isObject(a[1]) || !hexIdAt(a, 0, id)) {
            return ev::throwTypeError(
                "field(id, { seeds, blocked?, aura?, auraMult?, quantum?, ring? })");
        }
        const size_t cells = static_cast<size_t>(h->nav->cells());

        // Every JS read happens FIRST: a typed array's data pointer is a
        // snapshot the collector may abandon at the next allocation, so the
        // three views are taken back to back once nothing else will run.
        ev::Persistent opts(a[1]);
        ev::Persistent seedsV(ev::getProperty(opts.get(), "seeds"));
        ev::Persistent blockedV(ev::getProperty(opts.get(), "blocked"));
        ev::Persistent auraV(ev::getProperty(opts.get(), "aura"));
        const double auraMult = getDoubleProperty(opts.get(), "auraMult", 1.0);
        const double quantum = getDoubleProperty(opts.get(), "quantum", 0.25);
        // The ring is a bucket array the search allocates: clamp before the
        // cast (NaN or a huge double does not convert to int).
        const double ringD = getDoubleProperty(opts.get(), "ring", 64.0);
        const int ring = ringD >= 1.0 ? static_cast<int>(std::min(ringD, 1048576.0)) : 0;

        ev::TypedArrayInfo seedsInfo = ev::typedArrayInfo(seedsV.get());
        const int32_t* seeds = nullptr;
        size_t nSeeds = 0;
        if (seedsInfo && seedsInfo.elementKind == ev::elements::Int32) {
            seeds = reinterpret_cast<const int32_t*>(seedsInfo.data);
            nSeeds = seedsInfo.elementCount;
        }
        // No seeds is not an error: it builds an all-Infinity field.
        const uint8_t* blocked = nullptr;
        const uint8_t* aura = nullptr;
        const bool okB = hexMask(blockedV.get(), cells, blocked);
        const bool okA = hexMask(auraV.get(), cells, aura);
        if (!okB || !okA) {
            return ev::throwTypeError(
                "field: blocked and aura must be Uint8Arrays of size*size entries");
        }

        std::vector<double> dist;
        std::vector<int32_t> parent;
        const size_t pops = h->nav->targetField(id, seeds, nSeeds, blocked, aura,
                                                auraMult, quantum, ring, dist, parent);

        ObjectBuilder o;
        o.set("dist", makeFloat64Array(dist.data(), dist.size()));
        o.set("parent", makeInt32Array(parent.data(), parent.size()));
        o.set("pops", ev::fromDouble(static_cast<double>(pops)));
        return o.get();
    });
}

// ---------------------------------------------------------------------------
// Agent.applyAction()
// ---------------------------------------------------------------------------

void decorateAgentExtras(ObjectBuilder& b) {
    // applyAction(action, dt): the continuous-control entry point a policy /
    // network drives, as opposed to the path-following update(dt).
    b.def("applyAction", 2, [](Value self, std::span<const Value> a) -> Value {
        HostAgent* h = unwrapAgent(self);
        if (!h || h->destroyed) return ev::undefined();
        if (a.size() < 2) return ev::throwTypeError("applyAction(action, dt)");
        if (!ev::isObject(a[0])) return ev::throwTypeError("action must be an object");
        const float dt = static_cast<float>(numAt(a, 1));
        h->agent.applyAction(parseAgentAction(a[0]), dt);
        return ev::undefined();
    });
}

// ---------------------------------------------------------------------------
// World.findById() / registerAbility() / seed()
// ---------------------------------------------------------------------------

void decorateWorldExtras(ObjectBuilder& b) {
    // findById(unitId) → the AIAgent wrapper on this world's roster, or null.
    // The old body walked the wrapper's `__agents` array for pointer
    // identity; the roster HostWorld already keeps is the same list.
    b.def("findById", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::null();
        const int id = i32At(a, 0, "findById: id");
        const brogameagent::Agent* found = w->world.findById(id);
        if (!found) return ev::null();
        Value v = worldAgentValue(self, found);
        return ev::isUndefined(v) ? ev::null() : v;
    });

    // registerAbility(abilityId, { cooldown = 1, manaCost = 0, range = 0, fn })
    //
    // fn(caster, world, targetId) fires from the native cast path — including
    // from inside an MCTS rollout on a cloned World, where `caster` is a
    // clone's Agent and so has no wrapper: the old body passed undefined
    // there too.
    b.def("registerAbility", 2, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.size() < 2) return ev::undefined();
        const int abilityId = i32At(a, 0, "registerAbility: abilityId");
        if (!ev::isObject(a[1])) return ev::throwTypeError("spec must be an object");

        ev::Persistent selfP(self);
        ev::Persistent spec(a[1]);
        brogameagent::AbilitySpec s;
        s.cooldown = static_cast<float>(getDoubleProperty(spec.get(), "cooldown", 1.0));
        s.manaCost = static_cast<float>(getDoubleProperty(spec.get(), "manaCost", 0.0));
        s.range = static_cast<float>(getDoubleProperty(spec.get(), "range", 0.0));

        // The callback lives on the world handle as `_abilities[abilityId]`
        // (HostWorld), so an ability closing over whatever owns the world
        // does not pin it. Registering without a fn clears an earlier one.
        ev::Persistent fn(ev::getProperty(spec.get(), "fn"));
        const bool hasFn = ev::isFunction(fn.get());
        {
            ev::Persistent table(ev::getProperty(selfP.get(), "_abilities"));
            if (!ev::isObject(table.get())) {
                table.set(ev::createObject());
                selfP.set(ev::setProperty(selfP.get(), "_abilities", table.get()));
            }
            ev::setProperty(table.get(), std::to_string(abilityId),
                            hasFn ? fn.get() : ev::undefined());
        }
        if (hasFn) {
            HostWorld* worldHost = w;
            std::weak_ptr<int> life = w->life;
            const std::thread::id owner = std::this_thread::get_id();
            s.fn = [worldHost, life, abilityId, owner](brogameagent::Agent& caster,
                                                       brogameagent::World& /*world*/,
                                                       int targetId) {
                // A cloned World (an MCTS rollout) carries this fn with it and
                // can outlive the wrapper it was registered on.
                if (life.expired()) return;
                // rootParallelSearch rolls clones out on worker threads, and
                // bronze's runtime (and every Persistent) is per thread.
                if (std::this_thread::get_id() != owner) return;
                // Only a World method (ActiveWorldScope) gives the handle
                // the callback lives on.
                ev::Persistent worldVal(worldHost->activeSelf.get());
                if (!ev::isObject(worldVal.get())) return;
                ev::Persistent table(ev::getProperty(worldVal.get(), "_abilities"));
                if (!ev::isObject(table.get())) return;
                ev::Persistent callee(ev::getProperty(table.get(), std::to_string(abilityId)));
                if (!ev::isFunction(callee.get())) return;
                ev::Persistent casterVal(worldAgentValue(worldVal.get(), &caster));
                const Value args[3] = {
                    casterVal.get(), worldVal.get(), ev::fromDouble(targetId),
                };
                ev::call(callee.get(), ev::undefined(), std::span<const Value>(args, 3));
            };
        }

        w->world.registerAbility(abilityId, std::move(s));
        return ev::undefined();
    });

    // seed(s): reseed the world RNG (crit rolls, spread, dodge).
    b.def("seed", 1, [](Value self, std::span<const Value> a) -> Value {
        HostWorld* w = unwrapWorld(self);
        if (!w || a.empty()) return ev::undefined();
        w->world.seed(u64At(a, 0, "seed"));
        return ev::undefined();
    });
}

} // namespace brogameagent::api
