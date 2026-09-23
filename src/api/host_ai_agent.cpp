#include "host_ai_internal.h"
#include <cmath>

namespace brogameagent::api {

// ---------------------------------------------------------------------------
// HostAgentBinding Implementation
// ---------------------------------------------------------------------------

bool HostAgentBinding::navigateTo(bromath::Vec3 target, bromath::Vec3 extents, bool requireFullPath) {
    (void)extents;
    if (!agentHost || agentHost->destroyed) return false;
    goal = target;
    timeSinceRepath = 0.0f;
#if BROGAMEAGENT_HAS_NAVMESH
    if (navMesh) {
        meshGen = navMesh->generation();
        float startY = active ? path[static_cast<size_t>(currentWaypoint)].y : agentHost->agent.elevation();
        auto res = navMesh->findPathEx({agentHost->agent.x(), startY, agentHost->agent.z()}, target, extents, requireFullPath);
        if (res.points.empty()) {
            active = false; partial = res.partial; path.clear(); flags.clear();
            agentHost->agent.clearTarget();
            return false;
        }
        path = std::move(res.points);
        flags = std::move(res.flags);
        active = true; partial = res.partial; currentWaypoint = 0;
        onLink = (!flags.empty() && (flags[0] & brogameagent::NavMeshPath::kLinkStart) != 0);
        agentHost->agent.setTarget(path[0].x, path[0].z);
        return true;
    }
#endif
    if (navGrid) {
        auto res = navGrid->findPathEx({agentHost->agent.x(), agentHost->agent.z()}, {target.x, target.z}, requireFullPath);
        if (res.points.empty()) {
            active = false; partial = res.partial; path.clear(); flags.clear();
            agentHost->agent.clearTarget();
            return false;
        }
        path.clear(); path.reserve(res.points.size());
        for (const auto& p : res.points) path.push_back({p.x, 0.0f, p.y});
        flags.assign(path.size(), 0);
        active = true; partial = res.partial; currentWaypoint = 0; onLink = false;
        agentHost->agent.setTarget(path[0].x, path[0].z);
        return true;
    }
    active = true; partial = false; onLink = false;
    path = { target }; flags = { 0 }; currentWaypoint = 0;
    agentHost->agent.setTarget(target.x, target.z);
    return true;
}

void HostAgentBinding::stopNavigation() {
    active = false; onLink = false;
    path.clear(); flags.clear();
    if (agentHost) agentHost->agent.clearTarget();
}

void HostAgentBinding::step(float dt) {
    if (!agentHost || agentHost->destroyed) return;
    if (active && !path.empty()) {
#if BROGAMEAGENT_HAS_NAVMESH
        if (navMesh && repathInterval > 0.0f) {
            timeSinceRepath += dt;
            if (timeSinceRepath >= repathInterval || navMesh->generation() != meshGen) {
                navigateTo(goal, brogameagent::NavMesh::kDefaultExtents, false);
            }
        }
#endif
        constexpr float kAdvR = 0.75f, kArrR = 0.5f;
        while (currentWaypoint < static_cast<int>(path.size())) {
            const auto& wp = path[static_cast<size_t>(currentWaypoint)];
            float dx = wp.x - agentHost->agent.x(), dz = wp.z - agentHost->agent.z();
            float r = (currentWaypoint == static_cast<int>(path.size()) - 1) ? kArrR : kAdvR;
            if (dx * dx + dz * dz > r * r) break;
            currentWaypoint++;
        }
        if (currentWaypoint >= static_cast<int>(path.size())) {
            active = false; onLink = false;
            agentHost->agent.clearTarget();
        } else {
            const auto& wp = path[static_cast<size_t>(currentWaypoint)];
            agentHost->agent.setTarget(wp.x, wp.z);
            onLink = (currentWaypoint < static_cast<int>(flags.size()) && (flags[static_cast<size_t>(currentWaypoint)] & 0x01) != 0);
            const bromath::Vec3 from = (currentWaypoint > 0) ? path[static_cast<size_t>(currentWaypoint - 1)] : path.front();
            float sx = wp.x - from.x, sz = wp.z - from.z, segLenSq = sx * sx + sz * sz;
            if (segLenSq > 1e-6f) {
                float t = std::clamp(((agentHost->agent.x() - from.x) * sx + (agentHost->agent.z() - from.z) * sz) / segLenSq, 0.0f, 1.0f);
                agentHost->agent.setElevation(from.y + (wp.y - from.y) * t + yOffset);
            } else {
                agentHost->agent.setElevation(wp.y + yOffset);
            }
        }
    }
    agentHost->agent.update(dt);
}

// ---------------------------------------------------------------------------
// Steering Implementations
// ---------------------------------------------------------------------------

static SteeringOutput computeWander(bromath::Vec2 pos, float yaw, float radius, float dist, float jitter) {
    float cx = pos.x + std::sin(yaw) * dist, cz = pos.y - std::cos(yaw) * dist;
    static thread_local uint32_t s_seed = 123456789;
    s_seed = s_seed * 1664525u + 1013904223u;
    float normRnd = static_cast<float>(s_seed & 0xffff) / 65535.0f;
    float angle = yaw + (normRnd - 0.5f) * jitter * 6.2831853f;
    auto s = brogameagent::seek(pos, {cx + std::sin(angle) * radius, cz - std::cos(angle) * radius});
    return {s.fx, s.fz};
}

static SteeringOutput computeAvoid(bromath::Vec2 pos, bromath::Vec2 vel, const std::vector<brogameagent::AABB>& obs, float lookahead) {
    float spd = std::sqrt(vel.x * vel.x + vel.y * vel.y);
    if (spd < 1e-4f) return {0.0f, 0.0f};
    float dirX = vel.x / spd, dirZ = vel.y / spd, rayLen = spd * lookahead;
    float closestDist = 1e9f;
    bromath::Vec2 closestCenter{0, 0};
    bool found = false;
    for (const auto& box : obs) {
        float dx = box.cx - pos.x, dz = box.cz - pos.y;
        float proj = dx * dirX + dz * dirZ;
        if (proj < 0 || proj > rayLen) continue;
        float px = dx - proj * dirX, pz = dz - proj * dirZ;
        if (std::sqrt(px * px + pz * pz) < std::max(box.hw, box.hd) + 0.5f && proj < closestDist) {
            closestDist = proj; closestCenter = {box.cx, box.cz}; found = true;
        }
    }
    if (!found) return {0.0f, 0.0f};
    float sideX = -dirZ, sideZ = dirX;
    float sign = ((closestCenter.x - pos.x) * sideX + (closestCenter.y - pos.y) * sideZ >= 0.0f) ? -1.0f : 1.0f;
    return {sideX * sign * spd, sideZ * sign * spd};
}

// ---------------------------------------------------------------------------
// Agent Prototype Decoration
// ---------------------------------------------------------------------------

void decorateAgentProto(ObjectBuilder& b) {
    b.accessor("position",
        [](Value s, std::span<const Value>) -> Value {
            auto* h = unwrapAgent(s);
            return h ? makeVec3Value(h->agent.x(), h->navActive ? h->navY : h->agent.elevation(), h->agent.z()) : ev::undefined();
        },
        [](Value s, std::span<const Value> a) -> Value {
            auto* h = unwrapAgent(s);
            if (h && !a.empty() && ev::isObject(a[0])) {
                auto p = parseVec3(a[0]);
                h->agent.setPosition(p.x, p.z); h->agent.setElevation(p.y); h->navY = p.y;
            }
            return ev::undefined();
        });

    b.accessor("velocity", [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s);
        if (!h) return ev::undefined();
        auto v = h->agent.velocity();
        return makeVec3Value(v.x, 0.0f, v.y);
    }, nullptr);

    b.accessor("unit", [](Value self_, std::span<const Value>) -> Value {
        HostAgent* h = unwrapAgent(self_);
        if (!h) return ev::undefined();
        // A fresh proxy per read. The proxy roots its agent (so a detached
        // `const u = agent.unit` never outlives the agent it edits); caching
        // it in a Persistent on the agent closed a root cycle that made every
        // agent whose .unit was ever read uncollectable.
        return makeUnitHandle(h, self_);
    }, nullptr);

    auto speedGetter = [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.speed()) : ev::undefined();
    };
    auto speedSetter = [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setSpeed(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    };
    b.accessor("maxSpeed", speedGetter, speedSetter);
    b.accessor("speed", speedGetter, speedSetter);

    b.accessor("radius",
        [](Value s, std::span<const Value>) -> Value {
            auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.radius()) : ev::undefined();
        },
        [](Value s, std::span<const Value> a) -> Value {
            auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setRadius(static_cast<float>(numAt(a, 0)));
            return ev::undefined();
        });

    auto accelGetter = [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.unit().moveSpeed) : ev::undefined();
    };
    auto accelSetter = [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setMaxAccel(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    };
    b.accessor("maxAcceleration", accelGetter, accelSetter);
    b.accessor("maxAccel", accelGetter, accelSetter);

    b.accessor("x", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.x()) : ev::undefined(); }, nullptr);
    b.accessor("z", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.z()) : ev::undefined(); }, nullptr);
    b.accessor("currentWaypoint", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.currentWaypoint()) : ev::undefined(); }, nullptr);
    b.accessor("path", [](Value s, std::span<const Value>) -> Value { auto* h = unwrapAgent(s); return h ? makePathArray(h->agent.path()) : ev::undefined(); }, nullptr);
    b.accessor("atTarget", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromBool(h->agent.atTarget()) : ev::undefined(); }, nullptr);
    b.accessor("hasTarget", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromBool(h->agent.hasTarget() || h->navActive) : ev::undefined(); }, nullptr);
    b.accessor("yaw", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.yaw()) : ev::undefined(); }, nullptr);
    b.accessor("aimYaw", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.aimYaw()) : ev::undefined(); }, nullptr);
    b.accessor("aimPitch", [](Value s, std::span<const Value>) { auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.aimPitch()) : ev::undefined(); }, nullptr);

    b.accessor("elevation",
        [](Value s, std::span<const Value>) -> Value {
            auto* h = unwrapAgent(s); return h ? ev::fromDouble(h->agent.elevation()) : ev::undefined();
        },
        [](Value s, std::span<const Value> a) -> Value {
            auto* h = unwrapAgent(s);
            if (h && !a.empty()) { float y = static_cast<float>(numAt(a, 0)); h->agent.setElevation(y); h->navY = y; }
            return ev::undefined();
        });

    b.def("getPosition", 0, [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s);
        return h ? makeVec3Value(h->agent.x(), h->navActive ? h->navY : h->agent.elevation(), h->agent.z()) : ev::undefined();
    });

    b.def("getVelocity", 0, [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        auto v = h->agent.velocity(); return makeVec3Value(v.x, 0.0f, v.y);
    });

    b.def("setGoal", 3, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || h->destroyed) return ev::undefined();
        bromath::Vec3 target{0, 0, 0};
        if (a.size() >= 3) target = { static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)), static_cast<float>(numAt(a, 2)) };
        else if (a.size() >= 2 && !ev::isObject(a[0])) target = { static_cast<float>(numAt(a, 0)), 0.0f, static_cast<float>(numAt(a, 1)) };
        else if (!a.empty() && ev::isObject(a[0])) target = parseVec3(a[0]);

#if BROGAMEAGENT_HAS_NAVMESH
        if (h->navMesh) {
            float startY = h->navActive ? h->navY : h->agent.elevation();
            auto path = h->navMesh->findPathEx({h->agent.x(), startY, h->agent.z()}, target);
            if (!path.points.empty()) {
                h->navPath = std::move(path.points); h->navPathFlags = std::move(path.flags);
                h->navWaypoint = 0; h->navActive = true; h->navY = h->navPath.front().y;
                h->agent.setTarget(h->navPath[0].x, h->navPath[0].z);
            } else {
                h->navActive = false; h->navPath.clear(); h->agent.clearTarget();
            }
            return ev::undefined();
        }
#endif
        h->navActive = false; h->navPath.clear(); h->agent.setTarget(target.x, target.z);
        return ev::undefined();
    });

    b.def("setTarget", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        h->navActive = false; h->navPath.clear();
        if (a.size() >= 2) h->agent.setTarget(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        else if (!a.empty() && ev::isObject(a[0])) { auto p = parseVec2(a[0]); h->agent.setTarget(p.x, p.y); }
        return ev::undefined();
    });

    b.def("update", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || h->destroyed) return ev::undefined();
        float dt = a.empty() ? (1.0f / 60.0f) : static_cast<float>(numAt(a, 0));
        if (h->navActive && h->navMesh) {
            constexpr float kAdvR = 0.75f, kArrR = 0.5f;
            while (h->navWaypoint < static_cast<int>(h->navPath.size())) {
                const auto& wp = h->navPath[static_cast<size_t>(h->navWaypoint)];
                float dx = wp.x - h->agent.x(), dz = wp.z - h->agent.z();
                float r = (h->navWaypoint == static_cast<int>(h->navPath.size()) - 1) ? kArrR : kAdvR;
                if (dx * dx + dz * dz > r * r) break;
                h->navY = wp.y; h->navWaypoint++;
            }
            if (h->navWaypoint >= static_cast<int>(h->navPath.size())) {
                h->navActive = false; h->agent.clearTarget();
            } else {
                const auto& wp = h->navPath[static_cast<size_t>(h->navWaypoint)];
                h->agent.setTarget(wp.x, wp.z);
                const bromath::Vec3 from = (h->navWaypoint > 0) ? h->navPath[static_cast<size_t>(h->navWaypoint - 1)] : h->navPath.front();
                float sx = wp.x - from.x, sz = wp.z - from.z, segLenSq = sx * sx + sz * sz;
                if (segLenSq > 1e-6f) {
                    float t = std::clamp(((h->agent.x() - from.x) * sx + (h->agent.z() - from.z) * sz) / segLenSq, 0.0f, 1.0f);
                    h->navY = from.y + (wp.y - from.y) * t;
                } else {
                    h->navY = wp.y;
                }
            }
        }
        h->agent.update(dt);
        return ev::undefined();
    });

    auto clearFn = [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s);
        if (h && !h->destroyed) { h->navActive = false; h->navPath.clear(); h->agent.clearTarget(); }
        return ev::undefined();
    };
    b.def("stop", 0, clearFn);
    b.def("clearTarget", 0, clearFn);

    b.def("destroy", 0, [](Value s, std::span<const Value>) -> Value {
        auto* h = unwrapAgent(s);
        if (h) {
            h->destroyed = true; h->navActive = false; h->navPath.clear();
            h->agent.clearTarget(); h->agent.setNavGrid(nullptr); h->navMesh.reset();
        }
        return ev::undefined();
    });

    b.def("setPosition", 3, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        if (a.size() >= 3) {
            float x = static_cast<float>(numAt(a, 0)), y = static_cast<float>(numAt(a, 1)), z = static_cast<float>(numAt(a, 2));
            h->agent.setPosition(x, z); h->agent.setElevation(y); h->navY = y;
        } else if (a.size() >= 2) {
            h->agent.setPosition(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        } else if (!a.empty() && ev::isObject(a[0])) {
            auto p = parseVec3(a[0]);
            h->agent.setPosition(p.x, p.z); h->agent.setElevation(p.y); h->navY = p.y;
        }
        return ev::undefined();
    });

    b.def("setSpeed", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setSpeed(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setPath", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        Value list = argAt(a, 0);
        if (!ev::isObject(list) || ev::isFunction(list)) return ev::throwTypeError("setPath(waypoints: [{x,z}|[x,z], ...])");
        ev::Persistent root(list);
        Value lenV = ev::getProperty(root.get(), "length");
        if (ev::isUndefined(lenV) || ev::isObject(lenV)) return ev::throwTypeError("setPath(waypoints: [{x,z}|[x,z], ...])");
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
        std::vector<bromath::Vec2> path; path.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            Value wp = ev::getElement(root.get(), i);
            if (!ev::isObject(wp)) return ev::throwTypeError("setPath: waypoint must be {x,z} or [x,z]");
            path.push_back(parseVec2(wp));
        }
        h->navActive = false; h->navPath.clear(); h->agent.setPath(std::move(path));
        return ev::undefined();
    });

    b.def("setVelocity", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s);
        if (h && a.size() >= 2) h->agent.setVelocity(static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1)));
        return ev::undefined();
    });

    b.def("setYaw", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setYaw(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setMaxAccel", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setMaxAccel(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setMaxTurnRate", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setMaxTurnRate(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setAvoidance", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) applyAgentAvoidance(a[0], h->agent);
        return ev::undefined();
    });

    b.def("setRadius", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (h && !a.empty()) h->agent.setRadius(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setNavMesh", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        if (auto* nm = unwrapNavMesh(a[0])) h->navMesh = nm->mesh; else h->navMesh.reset();
        return ev::undefined();
    });

    b.def("setNavGrid", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        if (auto* ng = unwrapNavGrid(a[0])) h->agent.setNavGrid(ng->grid.get()); else h->agent.setNavGrid(nullptr);
        return ev::undefined();
    });

    b.def("aimAt", 4, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        float tx = 0, ty = 0, tz = 0, eyeH = 1.6f;
        if (a.size() >= 3) {
            tx = static_cast<float>(numAt(a, 0)); ty = static_cast<float>(numAt(a, 1)); tz = static_cast<float>(numAt(a, 2));
            if (a.size() >= 4) eyeH = static_cast<float>(numAt(a, 3));
        } else if (!a.empty() && ev::isObject(a[0])) {
            auto p = parseVec3(a[0]); tx = p.x; ty = p.y; tz = p.z;
            if (a.size() >= 2) eyeH = static_cast<float>(numAt(a, 1));
        }
        auto aim = h->agent.aimAt(tx, ty, tz, eyeH);
        ObjectBuilder res;
        res.set("yaw", ev::fromDouble(aim.yaw));
        res.set("pitch", ev::fromDouble(aim.pitch));
        return res.get();
    });

    b.def("seek", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        bromath::Vec2 target = (a.size() >= 2 && !ev::isObject(a[0]))
            ? bromath::Vec2{static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))} : parseVec2(a[0]);
        auto st = brogameagent::seek({h->agent.x(), h->agent.z()}, target);
        h->agent.setVelocity(st.fx * h->agent.speed(), st.fz * h->agent.speed());
        return makeSteeringOutput(st);
    });

    b.def("flee", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        bromath::Vec2 threat = (a.size() >= 2 && !ev::isObject(a[0]))
            ? bromath::Vec2{static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))} : parseVec2(a[0]);
        auto st = brogameagent::flee({h->agent.x(), h->agent.z()}, threat);
        h->agent.setVelocity(st.fx * h->agent.speed(), st.fz * h->agent.speed());
        return makeSteeringOutput(st);
    });

    b.def("arrive", 3, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        bromath::Vec2 target = parseVec2(a[0]); float slowR = 3.0f;
        if (a.size() >= 3) { target = {static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))}; slowR = static_cast<float>(numAt(a, 2)); }
        else if (a.size() >= 2) slowR = static_cast<float>(numAt(a, 1));
        auto st = brogameagent::arrive({h->agent.x(), h->agent.z()}, target, slowR);
        h->agent.setVelocity(st.fx * h->agent.speed(), st.fz * h->agent.speed());
        return makeSteeringOutput(st);
    });

    b.def("wander", 3, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        float radius = (a.size() >= 1) ? static_cast<float>(numAt(a, 0)) : 2.0f;
        float dist   = (a.size() >= 2) ? static_cast<float>(numAt(a, 1)) : 3.0f;
        float jitter = (a.size() >= 3) ? static_cast<float>(numAt(a, 2)) : 0.5f;
        auto st = computeWander({h->agent.x(), h->agent.z()}, h->agent.yaw(), radius, dist, jitter);
        h->agent.setVelocity(st.fx * h->agent.speed(), st.fz * h->agent.speed());
        return makeSteeringOutput(st);
    });

    b.def("avoid", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h || a.empty()) return ev::undefined();
        auto boxes = parseAABBArray(a[0]);
        float lookahead = (a.size() >= 2) ? static_cast<float>(numAt(a, 1)) : 1.0f;
        auto st = computeAvoid({h->agent.x(), h->agent.z()}, h->agent.velocity(), boxes, lookahead);
        h->agent.setVelocity(st.fx, st.fz);
        return makeSteeringOutput(st);
    });

    b.def("bind", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* h = unwrapAgent(s); if (!h) return ev::undefined();
        ev::Persistent agentP(s);
        auto* b = new HostAgentBinding();
        b->agentHost = h; b->agentLife = h->life; b->navMesh = h->navMesh;
        if (!a.empty() && ev::isObject(a[0])) {
            ev::Persistent root(a[0]);
            Value nmV = ev::getProperty(root.get(), "navMesh");
            if (auto* nm = unwrapNavMesh(nmV)) b->navMesh = nm->mesh;
            Value ngV = ev::getProperty(root.get(), "navGrid");
            if (auto* ng = unwrapNavGrid(ngV)) b->navGrid = ng->grid.get();
            b->yOffset = static_cast<float>(getDoubleProperty(root.get(), "yOffset", 0.0));
            b->repathInterval = static_cast<float>(getDoubleProperty(root.get(), "repathInterval", 0.0));
        }
        return attachBindingAgent(makeAgentBindingHandle(b), agentP.get());
    });

    decorateAgentExtras(b);  // applyAction() — host_ai_world_extra.cpp
}

// ---------------------------------------------------------------------------
// AgentBinding Prototype Decoration
// ---------------------------------------------------------------------------

void decorateAgentBindingProto(ObjectBuilder& b) {
    b.accessor("agent", [](Value s, std::span<const Value>) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (!bd || !bd->agentHost) return ev::undefined();
        return ev::getProperty(s, "_agent");
    }, nullptr);

    b.def("navigateTo", 3, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s); if (!bd || a.empty()) return ev::fromBool(false);
        bromath::Vec3 target = parseVec3(a[0]);
        bromath::Vec3 extents = brogameagent::NavMesh::kDefaultExtents;
        bool requireFull = false;
        if (a.size() >= 2 && ev::isObject(a[1])) {
            ev::Persistent root(a[1]);
            // Each read is consumed before the next one allocates.
            Value reqV = ev::getProperty(root.get(), "requireFullPath");
            if (!ev::isUndefined(reqV)) requireFull = ev::toBool(reqV);
            Value extV = ev::getProperty(root.get(), "extents");
            if (ev::isObject(extV)) extents = parseVec3(extV, extents);
            Value nmV = ev::getProperty(root.get(), "navMesh");
            if (auto* nm = unwrapNavMesh(nmV)) bd->navMesh = nm->mesh;
        }
        return ev::fromBool(bd->navigateTo(target, extents, requireFull));
    });

    b.def("stopNavigation", 0, [](Value s, std::span<const Value>) -> Value {
        auto* bd = unwrapAgentBinding(s); if (bd) bd->stopNavigation();
        return ev::undefined();
    });

    b.def("navigationInfo", 0, [](Value s, std::span<const Value>) -> Value {
        auto* bd = unwrapAgentBinding(s); if (!bd) return ev::null();
        ObjectBuilder info;
        info.set("active", ev::fromBool(bd->active));
        info.set("partial", ev::fromBool(bd->partial));
        info.set("onLink", ev::fromBool(bd->onLink));
        return info.get();
    });

    auto stepFn = [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (bd) { float dt = a.empty() ? (1.0f / 60.0f) : static_cast<float>(numAt(a, 0)); bd->step(dt); }
        return ev::undefined();
    };
    b.def("step", 1, stepFn);
    b.def("update", 1, stepFn);

    b.def("hold", 0, [](Value s, std::span<const Value>) -> Value {
        auto* bd = unwrapAgentBinding(s); if (bd && bd->agentHost) bd->agentHost->agent.clearTarget();
        return ev::undefined();
    });

    b.def("moveTo", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (bd && a.size() >= 2 && bd->agentHost) {
            bd->navigateTo({static_cast<float>(numAt(a, 0)), 0.0f, static_cast<float>(numAt(a, 1))},
                           brogameagent::NavMesh::kDefaultExtents, false);
        }
        return ev::undefined();
    });

    b.def("attack", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (bd && !a.empty() && bd->agentHost) {
            brogameagent::AgentAction act; act.attackTargetId = i32At(a, 0);
            bd->agentHost->agent.applyAction(act, 1.0f / 60.0f);
        }
        return ev::undefined();
    });

    b.def("cast", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (bd && !a.empty() && bd->agentHost) {
            brogameagent::AgentAction act; act.useAbilityId = i32At(a, 0);
            if (a.size() >= 2) act.attackTargetId = i32At(a, 1);
            bd->agentHost->agent.applyAction(act, 1.0f / 60.0f);
        }
        return ev::undefined();
    });

    b.def("flee", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s);
        if (bd && bd->agentHost && a.size() >= 2) {
            bromath::Vec2 threat{static_cast<float>(numAt(a, 0)), static_cast<float>(numAt(a, 1))};
            auto st = brogameagent::flee({bd->agentHost->agent.x(), bd->agentHost->agent.z()}, threat);
            bd->agentHost->agent.setVelocity(st.fx * bd->agentHost->agent.speed(), st.fz * bd->agentHost->agent.speed());
        }
        return ev::undefined();
    });

    b.def("distanceTo", 1, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s); if (!bd || !bd->agentHost || a.empty()) return ev::fromDouble(0.0);
        bromath::Vec2 p = parseVec2(a[0]);
        float dx = bd->agentHost->agent.x() - p.x, dz = bd->agentHost->agent.z() - p.y;
        return ev::fromDouble(std::sqrt(dx * dx + dz * dz));
    });

    b.def("inRange", 2, [](Value s, std::span<const Value> a) -> Value {
        auto* bd = unwrapAgentBinding(s); if (!bd || !bd->agentHost || a.empty()) return ev::fromBool(false);
        bromath::Vec2 p = parseVec2(a[0]);
        float range = (a.size() >= 2) ? static_cast<float>(numAt(a, 1)) : bd->agentHost->agent.unit().attackRange;
        float dx = bd->agentHost->agent.x() - p.x, dz = bd->agentHost->agent.z() - p.y;
        return ev::fromBool((dx * dx + dz * dz) <= range * range);
    });
}

// ---------------------------------------------------------------------------
// Factory Helpers
// ---------------------------------------------------------------------------

void applyAgentAvoidance(Value opts, brogameagent::Agent& agent) {
    brogameagent::AgentAvoidance av;
    if (ev::isBool(opts)) {
        av.enabled = ev::toBool(opts);
    } else if (ev::isObject(opts)) {
        ev::Persistent root(opts);
        av.enabled = getBoolProperty(root.get(), "enabled", av.enabled);
        av.radius = static_cast<float>(getDoubleProperty(root.get(), "radius", av.radius));
        av.maxSpeed = static_cast<float>(getDoubleProperty(root.get(), "maxSpeed", av.maxSpeed));
        av.neighborDist = static_cast<float>(getDoubleProperty(root.get(), "neighborDist", av.neighborDist));
        av.maxNeighbors = static_cast<int>(getDoubleProperty(root.get(), "maxNeighbors", av.maxNeighbors));
        av.timeHorizon = static_cast<float>(getDoubleProperty(root.get(), "timeHorizon", av.timeHorizon));
        av.timeHorizonObst = static_cast<float>(getDoubleProperty(root.get(), "timeHorizonObst", av.timeHorizonObst));
        av.height = static_cast<float>(getDoubleProperty(root.get(), "height", av.height));
        av.priority = static_cast<float>(getDoubleProperty(root.get(), "priority", av.priority));
        av.layers = static_cast<uint32_t>(getDoubleProperty(root.get(), "layers", av.layers));
        av.mask = static_cast<uint32_t>(getDoubleProperty(root.get(), "mask", av.mask));
    } else {
        return;
    }
    agent.setAvoidance(av);
}

Value makeAgentHandle(HostAgent* h) {
    return g_agentClass.make(h, [](void* p) { delete static_cast<HostAgent*>(p); });
}

Value makeAgentBindingHandle(HostAgentBinding* h) {
    return g_agentBindingClass.make(h, [](void* p) { delete static_cast<HostAgentBinding*>(p); });
}

// The binding keeps its agent alive through `_agent` on its own handle — an
// edge the collector traces — rather than a root, so an agent that keeps its
// binding (`agent.binding = agent.bind()`) is an ordinary, collectable cycle.
Value attachBindingAgent(Value binding, Value agent) {
    ev::Persistent bP(binding);
    if (!ev::isUndefined(agent)) bP.set(ev::setProperty(bP.get(), "_agent", agent));
    return bP.get();
}

Value aiCreateAgent(Value, std::span<const Value> a) {
    auto* h = new HostAgent();
    if (!a.empty() && ev::isObject(a[0])) {
        ev::Persistent root(a[0]);
        Value posV = ev::getProperty(root.get(), "position");
        bromath::Vec3 pos = parseVec3(posV);
        if (ev::isUndefined(posV)) {
            pos.x = static_cast<float>(getDoubleProperty(root.get(), "x", 0.0));
            pos.y = static_cast<float>(getDoubleProperty(root.get(), "y", getDoubleProperty(root.get(), "elevation", 0.0)));
            pos.z = static_cast<float>(getDoubleProperty(root.get(), "z", 0.0));
        }
        h->agent.setPosition(pos.x, pos.z);
        h->agent.setElevation(pos.y);
        h->navY = pos.y;
        double radius = getDoubleProperty(root.get(), "radius", 0.4);
        h->agent.setRadius(static_cast<float>(radius));
        double speed = getDoubleProperty(root.get(), "maxSpeed", getDoubleProperty(root.get(), "speed", 6.0));
        h->agent.setSpeed(static_cast<float>(speed));
        double maxAccel = getDoubleProperty(root.get(), "maxAcceleration", getDoubleProperty(root.get(), "maxAccel", -1.0));
        if (maxAccel > 0) h->agent.setMaxAccel(static_cast<float>(maxAccel));
        double maxTurnRate = getDoubleProperty(root.get(), "maxTurnRate", -1.0);
        if (maxTurnRate > 0) h->agent.setMaxTurnRate(static_cast<float>(maxTurnRate));

        h->agent.unit().id = static_cast<int>(getDoubleProperty(root.get(), "id", 0.0));
        h->agent.unit().teamId = static_cast<int>(getDoubleProperty(root.get(), "teamId", 0.0));
        double hp = getDoubleProperty(root.get(), "hp", 100.0);
        h->agent.unit().hp = static_cast<float>(hp);
        h->agent.unit().maxHp = static_cast<float>(getDoubleProperty(root.get(), "maxHp", hp));
        h->agent.unit().damage = static_cast<float>(getDoubleProperty(root.get(), "damage", 10.0));
        h->agent.unit().attackRange = static_cast<float>(getDoubleProperty(root.get(), "attackRange", 3.0));
        h->agent.unit().mana = static_cast<float>(getDoubleProperty(root.get(), "mana", 0.0));
        h->agent.unit().maxMana = static_cast<float>(getDoubleProperty(root.get(), "maxMana", 100.0));
        h->agent.unit().armor = static_cast<float>(getDoubleProperty(root.get(), "armor", 0.0));
        h->agent.unit().magicResist = static_cast<float>(getDoubleProperty(root.get(), "magicResist", 0.0));
        h->agent.unit().attacksPerSec = static_cast<float>(getDoubleProperty(root.get(), "attacksPerSec", 1.0));
        h->agent.unit().moveSpeed = static_cast<float>(speed);
        h->agent.unit().radius = static_cast<float>(radius);

        Value avoidV = ev::getProperty(root.get(), "avoidance");
        if (!ev::isUndefined(avoidV) && !ev::isNull(avoidV)) applyAgentAvoidance(avoidV, h->agent);

        Value nmV = ev::getProperty(root.get(), "navMesh");
        if (auto* nm = unwrapNavMesh(nmV)) h->navMesh = nm->mesh;

        Value ngV = ev::getProperty(root.get(), "navGrid");
        if (auto* ng = unwrapNavGrid(ngV)) h->agent.setNavGrid(ng->grid.get());
    }
    return makeAgentHandle(h);
}

Value aiCreateAgentBinding(Value, std::span<const Value> a) {
    auto* b = new HostAgentBinding();
    ev::Persistent agentP;
    if (!a.empty() && ev::isObject(a[0])) {
        ev::Persistent root(a[0]);
        agentP.set(ev::getProperty(root.get(), "agent"));
        if (auto* ag = unwrapAgent(agentP.get())) {
            b->agentHost = ag;
            b->agentLife = ag->life;
        } else {
            agentP.set(ev::undefined());
        }
        Value nmV = ev::getProperty(root.get(), "navMesh");
        if (auto* nm = unwrapNavMesh(nmV)) b->navMesh = nm->mesh;
        Value ngV = ev::getProperty(root.get(), "navGrid");
        if (auto* ng = unwrapNavGrid(ngV)) b->navGrid = ng->grid.get();
        b->yOffset = static_cast<float>(getDoubleProperty(root.get(), "yOffset", 0.0));
        b->repathInterval = static_cast<float>(getDoubleProperty(root.get(), "repathInterval", 0.0));
    }
    return attachBindingAgent(makeAgentBindingHandle(b), agentP.get());
}

} // namespace brogameagent::api
