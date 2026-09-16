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
};

struct HostAgentBinding {
    uint32_t tag = kHostAgentBindingTag;
    HostAgent* agentHost = nullptr;
    ev::Persistent agentRef;
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

struct HostWorld {
    uint32_t tag = kHostWorldTag;
    brogameagent::World world;
    struct Roster {
        const brogameagent::Agent* agent = nullptr;
        ev::Persistent value;
    };
    std::vector<Roster> roster;
};

// ---------------------------------------------------------------------------
// Host Classes
// ---------------------------------------------------------------------------

extern HostClass g_navGridClass;
extern HostClass g_navMeshClass;
extern HostClass g_agentClass;
extern HostClass g_agentBindingClass;
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
    return (h->tag == kHostAgentBindingTag) ? h : nullptr;
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
    if (ev::isUndefined(v) || ev::isNull(v) || ev::isObject(v)) return def;
    double d = ev::toDouble(v);
    return std::isnan(d) ? def : d;
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
    if (!ev::isUndefined(xV) || !ev::isUndefined(zV)) {
        float x = (!ev::isUndefined(xV) && !ev::isObject(xV)) ? static_cast<float>(ev::toDouble(xV)) : def.x;
        float z = (!ev::isUndefined(zV) && !ev::isObject(zV)) ? static_cast<float>(ev::toDouble(zV)) : def.y;
        return {x, z};
    }
    Value e0 = ev::getElement(root.get(), 0);
    Value e1 = ev::getElement(root.get(), 1);
    if (!ev::isUndefined(e0) && !ev::isUndefined(e1)) {
        float x = !ev::isObject(e0) ? static_cast<float>(ev::toDouble(e0)) : def.x;
        float z = !ev::isObject(e1) ? static_cast<float>(ev::toDouble(e1)) : def.y;
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
    if (!ev::isUndefined(xV) || !ev::isUndefined(yV) || !ev::isUndefined(zV)) {
        float x = (!ev::isUndefined(xV) && !ev::isObject(xV)) ? static_cast<float>(ev::toDouble(xV)) : def.x;
        float y = (!ev::isUndefined(yV) && !ev::isObject(yV)) ? static_cast<float>(ev::toDouble(yV)) : def.y;
        float z = (!ev::isUndefined(zV) && !ev::isObject(zV)) ? static_cast<float>(ev::toDouble(zV)) : def.z;
        return {x, y, z};
    }
    Value e0 = ev::getElement(root.get(), 0);
    Value e1 = ev::getElement(root.get(), 1);
    Value e2 = ev::getElement(root.get(), 2);
    if (!ev::isUndefined(e0) && !ev::isUndefined(e1) && !ev::isUndefined(e2)) {
        float x = !ev::isObject(e0) ? static_cast<float>(ev::toDouble(e0)) : def.x;
        float y = !ev::isObject(e1) ? static_cast<float>(ev::toDouble(e1)) : def.y;
        float z = !ev::isObject(e2) ? static_cast<float>(ev::toDouble(e2)) : def.z;
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
    if (!ev::isUndefined(minXV) && !ev::isUndefined(minZV) && !ev::isUndefined(maxXV) && !ev::isUndefined(maxZV)) {
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

    if (!ev::isUndefined(hwV) || !ev::isUndefined(hdV)) {
        box.hw = !ev::isUndefined(hwV) ? static_cast<float>(ev::toDouble(hwV)) : 0.5f;
        box.hd = !ev::isUndefined(hdV) ? static_cast<float>(ev::toDouble(hdV)) : 0.5f;
        box.cx = !ev::isUndefined(cxV) ? static_cast<float>(ev::toDouble(cxV)) : (!ev::isUndefined(xV) ? static_cast<float>(ev::toDouble(xV)) : 0.0f);
        box.cz = !ev::isUndefined(czV) ? static_cast<float>(ev::toDouble(czV)) : (!ev::isUndefined(zV) ? static_cast<float>(ev::toDouble(zV)) : 0.0f);
        return box;
    }

    if (!ev::isUndefined(wV) && !ev::isUndefined(dV)) {
        float w = static_cast<float>(ev::toDouble(wV));
        float d = static_cast<float>(ev::toDouble(dV));
        float x = !ev::isUndefined(xV) ? static_cast<float>(ev::toDouble(xV)) : 0.0f;
        float z = !ev::isUndefined(zV) ? static_cast<float>(ev::toDouble(zV)) : 0.0f;
        box.cx = !ev::isUndefined(cxV) ? static_cast<float>(ev::toDouble(cxV)) : (x + 0.5f * w);
        box.cz = !ev::isUndefined(czV) ? static_cast<float>(ev::toDouble(czV)) : (z + 0.5f * d);
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
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return result;
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

inline bool readFloatVector(Value v, std::vector<float>& out) {
    if (ev::isUndefined(v) || ev::isNull(v)) return false;
    if (auto info = ev::typedArrayInfo(v)) {
        if (info.data && (info.bytesPerElement == sizeof(float) || info.bytesPerElement == 0)) {
            const float* fp = reinterpret_cast<const float*>(info.data);
            out.assign(fp, fp + info.elementCount);
            return true;
        }
    }
    if (!ev::isObject(v)) return false;
    ev::Persistent root(v);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return false;
    uint32_t len = static_cast<uint32_t>(ev::toDouble(lenV));
    out.clear();
    out.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(root.get(), i);
        double d = (!ev::isUndefined(e) && !ev::isObject(e)) ? ev::toDouble(e) : 0.0;
        out.push_back(static_cast<float>(d));
    }
    return true;
}

inline bool readU32Vector(Value v, std::vector<uint32_t>& out) {
    if (ev::isUndefined(v) || ev::isNull(v)) return false;
    if (auto info = ev::typedArrayInfo(v)) {
        if (info.data && (info.bytesPerElement == sizeof(uint32_t) || info.bytesPerElement == 0)) {
            const uint32_t* up = reinterpret_cast<const uint32_t*>(info.data);
            out.assign(up, up + info.elementCount);
            return true;
        }
        if (info.data && info.bytesPerElement == sizeof(uint16_t)) {
            const uint16_t* up = reinterpret_cast<const uint16_t*>(info.data);
            out.clear();
            out.reserve(info.elementCount);
            for (uint32_t i = 0; i < info.elementCount; ++i) out.push_back(up[i]);
            return true;
        }
    }
    if (!ev::isObject(v)) return false;
    ev::Persistent root(v);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return false;
    uint32_t len = static_cast<uint32_t>(ev::toDouble(lenV));
    out.clear();
    out.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(root.get(), i);
        uint32_t u = (!ev::isUndefined(e) && !ev::isObject(e)) ? static_cast<uint32_t>(ev::toDouble(e)) : 0u;
        out.push_back(u);
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
Value aiCreateAgent(Value self, std::span<const Value> a);
Value aiCreateAgentBinding(Value self, std::span<const Value> a);
void applyAgentAvoidance(Value opts, brogameagent::Agent& agent);
void installSteering(ObjectBuilder& b);
void installPerception(ObjectBuilder& b);

// MCTS (host_ai_mcts.cpp)
void ensureAIMctsClassesInstalled();
void installAIMcts(ObjectBuilder& b);

// Core & HexNav & World (host_ai_core.cpp)
void decorateHexNavProto(ObjectBuilder& b);
void decorateWorldProto(ObjectBuilder& b);
Value aiCreateHexNav(Value self, std::span<const Value> a);
Value aiCreateWorld(Value self, std::span<const Value> a);
void ensureAIClassesInstalled();
Value makeAiGameValue();
Value makeBroAiValue();

} // namespace brogameagent::api
