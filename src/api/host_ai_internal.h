#pragma once

// Shared internal declarations and helpers for the bronze host AI & navigation module.

#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"
#include "arg_reader.h"

#include <brogameagent/brogameagent.h>
#include <brogameagent/nav_grid.h>
#include <brogameagent/nav_mesh.h>
#include <brogameagent/agent.h>
#include <brogameagent/avoidance.h>
#include <brogameagent/hex_nav.h>
#include <brogameagent/world.h>
#include <brogameagent/perception.h>
#include <brogameagent/steering.h>
#include <brogameagent/mcts.h>
#include <brogameagent/generic_mcts.h>
#include <brogameagent/unit.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace brogameagent::api {

inline constexpr uint32_t kHostNavGridTag        = 0x4E564744u;  // 'NVGD'
inline constexpr uint32_t kHostNavMeshTag        = 0x4E564D53u;  // 'NVMS'
inline constexpr uint32_t kHostAgentTag          = 0x41474E54u;  // 'AGNT'
inline constexpr uint32_t kHostAgentBindingTag   = 0x41424E44u;  // 'ABND'
inline constexpr uint32_t kHostHexNavTag         = 0x48584E56u;  // 'HXNV'
inline constexpr uint32_t kHostWorldTag          = 0x41574C44u;  // 'AWLD'
inline constexpr uint32_t kHostGenericMctsTag    = 0x474D4354u;  // 'GMCT'
inline constexpr uint32_t kHostClassicMctsTag    = 0x4D435453u;  // 'MCTS'
inline constexpr uint32_t kHostDecoupledMctsTag  = 0x444D4354u;  // 'DMCT'
inline constexpr uint32_t kHostTeamMctsTag       = 0x544D4354u;  // 'TMCT'
inline constexpr uint32_t kHostOptionTag         = 0x4F50544Eu;  // 'OPTN'
inline constexpr uint32_t kHostOptionMctsTag     = 0x4F4D4354u;  // 'OMCT'
inline constexpr uint32_t kHostUnitTag           = 0x4149554Eu;  // 'AIUN'
inline constexpr uint32_t kHostAgentSnapshotTag  = 0x4149534Eu;  // 'AISN'
inline constexpr uint32_t kHostWorldSnapshotTag  = 0x41495753u;  // 'AIWS'
inline constexpr uint32_t kHostVecSimTag         = 0x41495653u;  // 'AIVS'
inline constexpr uint32_t kHostRewardTrackerTag  = 0x41495254u;  // 'AIRT'

// ---------------------------------------------------------------------------
// Payload Structures
// ---------------------------------------------------------------------------

struct HostNavGrid {
    uint32_t tag = kHostNavGridTag;
    std::unique_ptr<brogameagent::NavGrid> grid;
};

struct HostNavMesh {
    uint32_t tag = kHostNavMeshTag;
    std::shared_ptr<brogameagent::NavMesh> mesh;
};

struct HostAgent {
    uint32_t tag = kHostAgentTag;
    brogameagent::Agent agent;
    std::shared_ptr<brogameagent::NavMesh> navMesh;

    // Navigation route tracking
    bool navActive = false;
    std::vector<bromath::Vec3> navPath;
    std::vector<uint8_t> navPathFlags;
    int navWaypoint = 0;
    float navY = 0.0f;
    bool destroyed = false;

    /// Expires with this HostAgent. A Unit proxy or AgentBinding keeps its
    /// agent alive through a `_agent` property on its own handle (an edge,
    /// not a root, so `agent.u = agent.unit` is an ordinary cycle); this
    /// token is what stops it dereferencing a collected agent if that
    /// property is deleted from JS.
    std::shared_ptr<int> life = std::make_shared<int>(1);
};

struct HostUnit {
    uint32_t tag = kHostUnitTag;
    HostAgent* owner = nullptr;
    std::weak_ptr<int> ownerLife;
    brogameagent::Agent* agent() const {
        return (owner && !ownerLife.expired()) ? &owner->agent : nullptr;
    }
};

struct HostAgentBinding {
    uint32_t tag = kHostAgentBindingTag;
    HostAgent* agentHost = nullptr;  // read through host(); alive while `_agent` is
    std::weak_ptr<int> agentLife;
    std::shared_ptr<brogameagent::NavMesh> navMesh;
    const brogameagent::NavGrid* navGrid = nullptr;

    bool active = false;
    bool partial = false;
    bool onLink = false;
    std::vector<bromath::Vec3> path;
    std::vector<uint8_t> flags;
    int currentWaypoint = 0;
    bromath::Vec3 goal{0.0f, 0.0f, 0.0f};
    float yOffset = 0.0f;
    float repathInterval = 0.0f;
    float timeSinceRepath = 0.0f;
    uint32_t meshGen = 0;

    brogameagent::Agent* agent() const {
        return agentHost ? &agentHost->agent : nullptr;
    }

    bool navigateTo(bromath::Vec3 target, bromath::Vec3 extents, bool requireFullPath);
    void stopNavigation();
    void step(float dt);
};

struct HostHexNav {
    uint32_t tag = kHostHexNavTag;
    std::unique_ptr<brogameagent::HexNav> nav;
};

// The world's JS-side state lives on its handle, where the collector traces
// it: the AIAgent wrappers as `_agents[key]` and the registerAbility
// callbacks as `_abilities[abilityId]`. Rooting either natively made an
// agent or ability that refers back to the world (`agent.world = world`, an
// ability closing over the game that owns the world) pin the world forever.
// An Agent deregisters itself from its World when destroyed, so the world
// needs no root to stay memory-safe.
struct HostWorld {
    uint32_t tag = kHostWorldTag;
    brogameagent::World world;
    struct Roster {
        const brogameagent::Agent* agent = nullptr;
        uint32_t key = 0;  // the wrapper is self._agents[key]
    };
    std::vector<Roster> roster;
    uint32_t nextRosterKey = 0;

    /// The world handle a World method is running on, rooted only for the
    /// span of an ActiveWorldScope (undefined otherwise). Ability dispatch
    /// reads `_abilities` and `_agents` through it; outside a scope (a world
    /// ticked by a host that never went through a JS method) no JS ability
    /// callback runs.
    ev::Persistent activeSelf;

    /// Expires when this HostWorld is destroyed. An AbilitySpec::fn survives
    /// a World copy (an MCTS rollout clones the World), so a clone can still
    /// hold the callback after the wrapper it came from is gone; the callback
    /// checks this token instead of dereferencing a dangling host.
    std::shared_ptr<int> life = std::make_shared<int>(1);

    const Roster* rosterEntry(const brogameagent::Agent* agent) const {
        for (const auto& r : roster) {
            if (r.agent == agent) return &r;
        }
        return nullptr;
    }
};

/// The AIAgent wrapper the world handle `worldSelf` keeps for `agent` (its
/// `_agents` entry), or undefined when the agent is not on its roster.
/// ALLOCATES; `worldSelf` must be current.
Value worldAgentValue(Value worldSelf, const brogameagent::Agent* agent);

/// `self` must be current (taken before any allocation in the caller). Both
/// it and the value it shadows are held in Persistents, so nested dispatch
/// restores a current address rather than a pre-collection one.
struct ActiveWorldScope {
    HostWorld* w = nullptr;
    ev::Persistent prev;
    ActiveWorldScope(HostWorld* world, Value self) : w(world) {
        if (w) {
            prev.set(w->activeSelf.get());
            w->activeSelf.set(self);
        }
    }
    ~ActiveWorldScope() {
        if (w) {
            w->activeSelf.set(prev.get());
        }
    }
    ActiveWorldScope(const ActiveWorldScope&) = delete;
    ActiveWorldScope& operator=(const ActiveWorldScope&) = delete;
};

// ---------------------------------------------------------------------------
// Host Classes
// ---------------------------------------------------------------------------

extern HostClass g_navGridClass;
extern HostClass g_navMeshClass;
extern HostClass g_agentClass;
extern HostClass g_agentBindingClass;
extern HostClass g_unitClass;
extern HostClass g_agentSnapshotClass;
extern HostClass g_worldSnapshotClass;
extern HostClass g_vecSimClass;
extern HostClass g_rewardTrackerClass;
extern HostClass g_hexNavClass;
extern HostClass g_worldClass;
extern HostClass g_genericMctsClass;
extern HostClass g_mctsClass;
extern HostClass g_decoupledMctsClass;
extern HostClass g_teamMctsClass;
extern HostClass g_optionClass;
extern HostClass g_optionMctsClass;

// ---------------------------------------------------------------------------
// Unwrap Helpers
// ---------------------------------------------------------------------------

inline HostUnit* unwrapUnit(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostUnit*>(ptr);
    return (h->tag == kHostUnitTag) ? h : nullptr;
}

inline HostNavGrid* unwrapNavGrid(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostNavGrid*>(ptr);
    return (h->tag == kHostNavGridTag) ? h : nullptr;
}

inline HostNavMesh* unwrapNavMesh(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostNavMesh*>(ptr);
    return (h->tag == kHostNavMeshTag) ? h : nullptr;
}

inline HostAgent* unwrapAgent(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostAgent*>(ptr);
    return (h->tag == kHostAgentTag) ? h : nullptr;
}

inline HostAgentBinding* unwrapAgentBinding(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostAgentBinding*>(ptr);
    if (h->tag != kHostAgentBindingTag) return nullptr;
    // Every binding method comes through here, so a collected agent is
    // forgotten before anything can reach it.
    if (h->agentHost && h->agentLife.expired()) h->agentHost = nullptr;
    return h;
}

inline HostHexNav* unwrapHexNav(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostHexNav*>(ptr);
    return (h->tag == kHostHexNavTag) ? h : nullptr;
}

inline HostWorld* unwrapWorld(Value v) {
    if (!ev::isObject(v)) return nullptr;
    void* ptr = ev::handleData(v);
    if (!ptr) return nullptr;
    auto* h = static_cast<HostWorld*>(ptr);
    return (h->tag == kHostWorldTag) ? h : nullptr;
}

// ---------------------------------------------------------------------------
// Value Builders & Readers
// ---------------------------------------------------------------------------

inline Value makeVec2Value(float x, float z) {
    ObjectBuilder b;
    b.set("x", ev::fromDouble(x));
    b.set("z", ev::fromDouble(z));
    return b.get();
}

inline Value makeVec3Value(float x, float y, float z) {
    ObjectBuilder b;
    b.set("x", ev::fromDouble(x));
    b.set("y", ev::fromDouble(y));
    b.set("z", ev::fromDouble(z));
    return b.get();
}

inline Value makeSteeringOutput(const brogameagent::SteeringOutput& s) {
    ObjectBuilder o;
    o.set("fx", ev::fromDouble(s.fx));
    o.set("fz", ev::fromDouble(s.fz));
    return o.get();
}

inline Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make) {
    ev::CallResult parsed = ev::parseJson("[]");
    if (parsed.thrown || !ev::isObject(parsed.value)) return ev::undefined();
    ev::Persistent arr(parsed.value);
    if (count == 0) return arr.get();

    ev::Persistent push(ev::getProperty(arr.get(), "push"));
    if (!ev::isFunction(push.get())) return arr.get();

    for (size_t i = 0; i < count; ++i) {
        Value v = make(i);
        ev::call(push.get(), arr.get(), std::span<const Value>(&v, 1));
    }
    return arr.get();
}

inline Value makePathArray(const std::vector<bromath::Vec3>& pts) {
    return hostArrayOf(pts.size(), [&](size_t i) {
        return makeVec3Value(pts[i].x, pts[i].y, pts[i].z);
    });
}

inline Value makePathArray(const std::vector<bromath::Vec2>& pts) {
    return hostArrayOf(pts.size(), [&](size_t i) {
        return makeVec2Value(pts[i].x, pts[i].y);
    });
}

inline Value makeFloat32Array(const float* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(float)));
    }
    return view.get();
}

inline Value makeInt32Array(const int32_t* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Int32, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(int32_t)));
    }
    return view.get();
}

inline Value makeUint32Array(const uint32_t* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Uint32, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data), count * sizeof(uint32_t)));
    }
    return view.get();
}

inline double getDoubleProperty(Value obj, const char* key, double def = 0.0) {
    if (!ev::isObject(obj)) return def;
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    if (!ev::isNumber(v)) return def;
    double d = ev::toDouble(v);
    return (!std::isfinite(d)) ? def : d;
}

inline uint64_t getU64Property(Value obj, const char* key, uint64_t def = 0) {
    if (!ev::isObject(obj)) return def;
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toUint64(v);
}

inline bool getBoolProperty(Value obj, const char* key, bool def = false) {
    if (!ev::isObject(obj)) return def;
    ev::Persistent root(obj);
    Value v = ev::getProperty(root.get(), key);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toBool(v);
}

inline bromath::Vec2 parseVec2(Value v, bromath::Vec2 def = {0.0f, 0.0f}) {
    if (!ev::isObject(v)) return def;
    ev::Persistent root(v);
    Value xV = ev::getProperty(root.get(), "x");
    Value zV = ev::getProperty(root.get(), "z");
    if (ev::isUndefined(zV)) zV = ev::getProperty(root.get(), "y");
    if (ev::isNumber(xV) || ev::isNumber(zV)) {
        float x = ev::isNumber(xV) ? static_cast<float>(ev::toDouble(xV)) : def.x;
        float z = ev::isNumber(zV) ? static_cast<float>(ev::toDouble(zV)) : def.y;
        return {x, z};
    }
    Value e0 = ev::getElement(root.get(), 0);
    Value e1 = ev::getElement(root.get(), 1);
    if (!ev::isUndefined(e0) && !ev::isUndefined(e1)) {
        float x = ev::isNumber(e0) ? static_cast<float>(ev::toDouble(e0)) : def.x;
        float z = ev::isNumber(e1) ? static_cast<float>(ev::toDouble(e1)) : def.y;
        return {x, z};
    }
    return def;
}

inline bromath::Vec3 parseVec3(Value v, bromath::Vec3 def = {0.0f, 0.0f, 0.0f}) {
    if (!ev::isObject(v)) return def;
    ev::Persistent root(v);
    Value xV = ev::getProperty(root.get(), "x");
    Value yV = ev::getProperty(root.get(), "y");
    Value zV = ev::getProperty(root.get(), "z");
    if (ev::isNumber(xV) || ev::isNumber(yV) || ev::isNumber(zV)) {
        float x = ev::isNumber(xV) ? static_cast<float>(ev::toDouble(xV)) : def.x;
        float y = ev::isNumber(yV) ? static_cast<float>(ev::toDouble(yV)) : def.y;
        float z = ev::isNumber(zV) ? static_cast<float>(ev::toDouble(zV)) : def.z;
        return {x, y, z};
    }
    Value e0 = ev::getElement(root.get(), 0);
    Value e1 = ev::getElement(root.get(), 1);
    Value e2 = ev::getElement(root.get(), 2);
    if (!ev::isUndefined(e0) && !ev::isUndefined(e1) && !ev::isUndefined(e2)) {
        float x = ev::isNumber(e0) ? static_cast<float>(ev::toDouble(e0)) : def.x;
        float y = ev::isNumber(e1) ? static_cast<float>(ev::toDouble(e1)) : def.y;
        float z = ev::isNumber(e2) ? static_cast<float>(ev::toDouble(e2)) : def.z;
        return {x, y, z};
    }
    return def;
}

inline brogameagent::AABB parseAABB(Value v) {
    brogameagent::AABB box{0.0f, 0.0f, 0.5f, 0.5f};
    if (!ev::isObject(v)) return box;
    ev::Persistent root(v);

    Value minXV = ev::getProperty(root.get(), "minX");
    Value minZV = ev::getProperty(root.get(), "minZ");
    Value maxXV = ev::getProperty(root.get(), "maxX");
    Value maxZV = ev::getProperty(root.get(), "maxZ");
    if (ev::isNumber(minXV) && ev::isNumber(minZV) && ev::isNumber(maxXV) && ev::isNumber(maxZV)) {
        float x0 = static_cast<float>(ev::toDouble(minXV));
        float z0 = static_cast<float>(ev::toDouble(minZV));
        float x1 = static_cast<float>(ev::toDouble(maxXV));
        float z1 = static_cast<float>(ev::toDouble(maxZV));
        box.cx = 0.5f * (x0 + x1);
        box.cz = 0.5f * (z0 + z1);
        box.hw = 0.5f * std::abs(x1 - x0);
        box.hd = 0.5f * std::abs(z1 - z0);
        return box;
    }

    Value hwV = ev::getProperty(root.get(), "hw");
    Value hdV = ev::getProperty(root.get(), "hd");
    Value cxV = ev::getProperty(root.get(), "cx");
    Value czV = ev::getProperty(root.get(), "cz");
    Value xV = ev::getProperty(root.get(), "x");
    Value zV = ev::getProperty(root.get(), "z");
    Value wV = ev::getProperty(root.get(), "width");
    Value dV = ev::getProperty(root.get(), "depth");

    if (ev::isNumber(hwV) || ev::isNumber(hdV)) {
        box.hw = ev::isNumber(hwV) ? static_cast<float>(ev::toDouble(hwV)) : 0.5f;
        box.hd = ev::isNumber(hdV) ? static_cast<float>(ev::toDouble(hdV)) : 0.5f;
        box.cx = ev::isNumber(cxV) ? static_cast<float>(ev::toDouble(cxV)) : (ev::isNumber(xV) ? static_cast<float>(ev::toDouble(xV)) : 0.0f);
        box.cz = ev::isNumber(czV) ? static_cast<float>(ev::toDouble(czV)) : (ev::isNumber(zV) ? static_cast<float>(ev::toDouble(zV)) : 0.0f);
        return box;
    }

    if (ev::isNumber(wV) && ev::isNumber(dV)) {
        float w = static_cast<float>(ev::toDouble(wV));
        float d = static_cast<float>(ev::toDouble(dV));
        float x = ev::isNumber(xV) ? static_cast<float>(ev::toDouble(xV)) : 0.0f;
        float z = ev::isNumber(zV) ? static_cast<float>(ev::toDouble(zV)) : 0.0f;
        box.cx = ev::isNumber(cxV) ? static_cast<float>(ev::toDouble(cxV)) : (x + 0.5f * w);
        box.cz = ev::isNumber(czV) ? static_cast<float>(ev::toDouble(czV)) : (z + 0.5f * d);
        box.hw = 0.5f * w;
        box.hd = 0.5f * d;
        return box;
    }

    return box;
}

inline std::vector<brogameagent::AABB> parseAABBArray(Value v) {
    std::vector<brogameagent::AABB> result;
    if (!ev::isObject(v)) return result;
    ev::Persistent root(v);
    Value lenV = ev::getProperty(root.get(), "length");
    if (!ev::isNumber(lenV)) return result;
    uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
    result.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value el = ev::getElement(root.get(), i);
        if (ev::isObject(el)) {
            result.push_back(parseAABB(el));
        }
    }
    return result;
}

inline brogameagent::DamageKind parseDamageKind(const char* str) {
    if (str && strcmp(str, "magical") == 0) return brogameagent::DamageKind::Magical;
    if (str && strcmp(str, "true") == 0) return brogameagent::DamageKind::True;
    return brogameagent::DamageKind::Physical;
}

inline const char* damageKindStr(brogameagent::DamageKind k) {
    switch (k) {
        case brogameagent::DamageKind::Magical: return "magical";
        case brogameagent::DamageKind::True: return "true";
        default: return "physical";
    }
}

inline brogameagent::AgentAction parseAgentAction(Value obj) {
    brogameagent::AgentAction a;
    if (!ev::isObject(obj)) return a;
    ev::Persistent root(obj);
    a.moveX = static_cast<float>(getDoubleProperty(root.get(), "moveX", 0.0));
    a.moveZ = static_cast<float>(getDoubleProperty(root.get(), "moveZ", 0.0));
    a.aimYaw = static_cast<float>(getDoubleProperty(root.get(), "aimYaw", 0.0));
    a.aimPitch = static_cast<float>(getDoubleProperty(root.get(), "aimPitch", 0.0));
    a.attackTargetId = static_cast<int>(getDoubleProperty(root.get(), "attackTargetId", -1.0));
    a.useAbilityId = static_cast<int>(getDoubleProperty(root.get(), "useAbilityId",
                     getDoubleProperty(root.get(), "abilitySlot", -1.0)));
    return a;
}

/// A JS `length` as a loop bound: 0 for a non-number, NaN or negative
/// length (an array-like object can say anything), clamped to 2^32 - 1.
inline uint32_t toLength(Value lenV) {
    if (!ev::isNumber(lenV)) return 0;
    const double d = ev::toDouble(lenV);
    if (!(d > 0.0)) return 0;
    return d >= 4294967295.0 ? 0xFFFFFFFFu : static_cast<uint32_t>(d);
}

/// What to reserve() for `n` elements an array-like claims to have: a
/// `{ length: 1e9 }` must not become a multi-gigabyte allocation up front.
inline size_t reserveHint(uint32_t n) {
    return std::min<size_t>(n, size_t{1} << 16);
}

/// A number as a uint32 element: 0 for NaN / negative, clamped at the top
/// (a plain static_cast of either is undefined behaviour).
inline uint32_t toU32Clamped(double d) {
    if (!(d > 0.0)) return 0u;
    return d >= 4294967295.0 ? 0xFFFFFFFFu : static_cast<uint32_t>(d);
}

/// A Float32Array is copied as-is; any other typed array or array-like is
/// read element by element (so an Int32Array's values, not its bit patterns).
inline bool readFloatVector(Value v, std::vector<float>& out) {
    if (ev::isUndefined(v) || ev::isNull(v)) return false;
    if (auto info = ev::typedArrayInfo(v)) {
        if (info.data && info.elementKind == ev::elements::Float32) {
            const float* fp = reinterpret_cast<const float*>(info.data);
            out.assign(fp, fp + info.elementCount);
            return true;
        }
    }
    if (!ev::isObject(v)) return false;
    ev::Persistent root(v);
    Value lenV = ev::getProperty(root.get(), "length");
    if (!ev::isNumber(lenV)) return false;
    const uint32_t len = toLength(lenV);
    out.clear();
    out.reserve(reserveHint(len));
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(root.get(), i);
        double d = ev::isNumber(e) ? ev::toDouble(e) : 0.0;
        out.push_back(static_cast<float>(d));
    }
    return true;
}

/// A Uint32Array is copied as-is and a Uint16Array / Uint8Array widened; any
/// other typed array or array-like is read element by element.
inline bool readU32Vector(Value v, std::vector<uint32_t>& out) {
    if (ev::isUndefined(v) || ev::isNull(v)) return false;
    if (auto info = ev::typedArrayInfo(v)) {
        if (info.data && info.elementKind == ev::elements::Uint32) {
            const uint32_t* up = reinterpret_cast<const uint32_t*>(info.data);
            out.assign(up, up + info.elementCount);
            return true;
        }
        if (info.data && info.elementKind == ev::elements::Uint16) {
            const uint16_t* up = reinterpret_cast<const uint16_t*>(info.data);
            out.assign(up, up + info.elementCount);
            return true;
        }
        if (info.data && info.elementKind == ev::elements::Uint8) {
            const uint8_t* up = reinterpret_cast<const uint8_t*>(info.data);
            out.assign(up, up + info.elementCount);
            return true;
        }
    }
    if (!ev::isObject(v)) return false;
    ev::Persistent root(v);
    Value lenV = ev::getProperty(root.get(), "length");
    if (!ev::isNumber(lenV)) return false;
    const uint32_t len = toLength(lenV);
    out.clear();
    out.reserve(reserveHint(len));
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(root.get(), i);
        out.push_back(ev::isNumber(e) ? toU32Clamped(ev::toDouble(e)) : 0u);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prototypes & Factory Declarations
// ---------------------------------------------------------------------------

// NavGrid (host_ai_navgrid.cpp)
void decorateNavGridProto(ObjectBuilder& b);
Value makeNavGridHandle(std::unique_ptr<brogameagent::NavGrid> grid);
Value aiCreateNavGrid(Value self, std::span<const Value> a);

// NavMesh (host_ai_navmesh.cpp)
void decorateNavMeshProto(ObjectBuilder& b);
Value makeNavMeshHandle(std::shared_ptr<brogameagent::NavMesh> mesh);
Value aiBakeNavMesh(Value self, std::span<const Value> a);
Value aiLoadNavMesh(Value self, std::span<const Value> a);
Value aiBuildFromMesh(Value self, std::span<const Value> a);

// Agent & AgentBinding (host_ai_agent.cpp)
void decorateAgentProto(ObjectBuilder& b);
void decorateAgentBindingProto(ObjectBuilder& b);
Value makeAgentHandle(HostAgent* h);
Value makeAgentBindingHandle(HostAgentBinding* h);
/// A binding handle whose `_agent` is `agent` (unless undefined). ALLOCATES.
Value makeAgentBindingHandle(HostAgentBinding* h, const ev::Persistent& agent);
Value aiCreateAgent(Value self, std::span<const Value> a);
Value aiCreateAgentBinding(Value self, std::span<const Value> a);
void applyAgentAvoidance(Value opts, brogameagent::Agent& agent);
// bro.ai.game.registerCapability (host_ai_capability.cpp).
void installRegisterCapability(ObjectBuilder& b);

// AIUnit (host_ai_unit.cpp)
void decorateUnitProto(ObjectBuilder& b);
Value makeUnitHandle(HostAgent* owner, Value agentVal = ev::undefined());

// Extras (host_ai_extras.cpp)
void ensureAIExtrasClassesInstalled();
void installAIExtras(ObjectBuilder& game);

// MCTS (host_ai_mcts.cpp)
void ensureAIMctsClassesInstalled();
void installAIMcts(ObjectBuilder& b);

// Game & HexNav & World (host_ai_game.cpp)
void decorateHexNavProto(ObjectBuilder& b);
void decorateWorldProto(ObjectBuilder& b);
Value aiCreateHexNav(Value self, std::span<const Value> a);
Value aiCreateWorld(Value self, std::span<const Value> a);
void ensureAIClassesInstalled();

// The tail of the pre-transition HexNav / Agent / World surface
// (host_ai_world_extra.cpp). Split out so host_ai_game.cpp stays well under
// the file-size ceiling.
void decorateHexNavExtras(ObjectBuilder& b);
void decorateAgentExtras(ObjectBuilder& b);
void decorateWorldExtras(ObjectBuilder& b);

Value makeAiGameValue();
Value makeBroAiValue();

} // namespace brogameagent::api
