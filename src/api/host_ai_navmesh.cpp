#include "api.h"
#include "host_ai_internal.h"

namespace brogameagent::api {

#if BROGAMEAGENT_HAS_NAVMESH

namespace {

/// The scalar bake knobs, shared by bakeNavMesh and both buildFromMesh
/// spellings. `opts` must be rooted by the caller (each read allocates).
void readBakeConfig(Value opts, brogameagent::NavMeshBakeConfig& cfg) {
    ev::Persistent root(opts);
    auto num = [&](const char* key, float def) {
        return static_cast<float>(getDoubleProperty(root.get(), key, def));
    };
    cfg.cellSize = num("cellSize", cfg.cellSize);
    cfg.cellHeight = num("cellHeight", cfg.cellHeight);
    cfg.agentRadius = num("agentRadius", cfg.agentRadius);
    cfg.agentHeight = num("agentHeight", cfg.agentHeight);
    cfg.agentMaxClimb = num("agentMaxClimb", cfg.agentMaxClimb);
    cfg.agentMaxSlopeDeg = num("agentMaxSlopeDeg", cfg.agentMaxSlopeDeg);
    cfg.regionMinSize = num("regionMinSize", cfg.regionMinSize);
    cfg.regionMergeSize = num("regionMergeSize", cfg.regionMergeSize);
    cfg.edgeMaxLen = num("edgeMaxLen", cfg.edgeMaxLen);
    cfg.edgeMaxError = num("edgeMaxError", cfg.edgeMaxError);
    cfg.detailSampleDist = num("detailSampleDist", cfg.detailSampleDist);
    cfg.detailSampleMaxError = num("detailSampleMaxError", cfg.detailSampleMaxError);
    cfg.dynamicObstacles = getBoolProperty(root.get(), "dynamicObstacles", cfg.dynamicObstacles);
    cfg.tileSize = num("tileSize", cfg.tileSize);
    cfg.maxObstacles = static_cast<int>(getDoubleProperty(root.get(), "maxObstacles", cfg.maxObstacles));
}

/// `{ requireFullPath?, extents? }`, or a bare extents vector, as findPath
/// and navigateTo take it. Every read is finished with before the next one
/// allocates.
void readPathOptions(Value opts, bromath::Vec3& extents, bool& requireFull) {
    ev::Persistent root(opts);
    Value reqV = ev::getProperty(root.get(), "requireFullPath");
    const bool hasReq = !ev::isUndefined(reqV);
    if (hasReq) requireFull = ev::toBool(reqV);
    ev::Persistent extV(ev::getProperty(root.get(), "extents"));
    if (ev::isObject(extV.get())) {
        extents = parseVec3(extV.get(), extents);
    } else if (!hasReq && ev::isUndefined(extV.get())) {
        extents = parseVec3(root.get(), extents);
    }
}

} // namespace

void decorateNavMeshProto(ObjectBuilder& b) {
    b.accessor("valid", [](Value self_, std::span<const Value>) {
        HostNavMesh* h = unwrapNavMesh(self_);
        if (!h) return ev::undefined();
        return ev::fromBool(h->mesh && h->mesh->valid());
    }, nullptr);

    b.def("findPath", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.size() < 2) return ev::null();

        bromath::Vec3 start = parseVec3(a[0]);
        bromath::Vec3 end   = parseVec3(a[1]);
        bromath::Vec3 extents = brogameagent::NavMesh::kDefaultExtents;
        bool requireFull = false;

        if (a.size() >= 3 && ev::isObject(a[2])) {
            readPathOptions(a[2], extents, requireFull);
        }

        auto res = h->mesh->findPathEx(start, end, extents, requireFull);
        if (res.points.empty()) return ev::null();

        std::vector<uint32_t> linkIndices;
        for (size_t i = 0; i < res.points.size(); ++i) {
            if (res.isLinkStart(i)) linkIndices.push_back(static_cast<uint32_t>(i));
        }

        Value arr = makeFloat32Array(&res.points[0].x, res.points.size() * 3);
        ev::Persistent p(arr);
        p.set(ev::setProperty(p.get(), "partial", ev::fromBool(res.partial)));

        Value linksVal = hostArrayOf(linkIndices.size(), [&](size_t i) {
            return ev::fromDouble(linkIndices[i]);
        });
        p.set(ev::setProperty(p.get(), "links", linksVal));

        return p.get();
    });

    b.def("findRandomPoint", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh) return ev::null();
        uint32_t seed = a.empty() ? 0 : u32At(a, 0);
        bromath::Vec3 out;
        if (!h->mesh->randomPoint(seed, out)) return ev::null();
        return makeVec3Value(out.x, out.y, out.z);
    });

    b.def("randomPoint", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh) return ev::null();
        uint32_t seed = a.empty() ? 0 : u32At(a, 0);
        bromath::Vec3 out;
        if (!h->mesh->randomPoint(seed, out)) return ev::null();
        return makeVec3Value(out.x, out.y, out.z);
    });

    b.def("closestPoint", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.empty()) return ev::null();
        bromath::Vec3 pos = parseVec3(a[0]);
        bromath::Vec3 extents = (a.size() >= 2) ? parseVec3(a[1], brogameagent::NavMesh::kDefaultExtents)
                                                : brogameagent::NavMesh::kDefaultExtents;
        bromath::Vec3 out;
        if (!h->mesh->nearestPoint(pos, out, extents)) return ev::null();
        return makeVec3Value(out.x, out.y, out.z);
    });

    b.def("samplePosition", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.empty()) return ev::null();
        bromath::Vec3 pos = parseVec3(a[0]);
        bromath::Vec3 extents = (a.size() >= 2) ? parseVec3(a[1], brogameagent::NavMesh::kDefaultExtents)
                                                : brogameagent::NavMesh::kDefaultExtents;
        bromath::Vec3 out;
        if (!h->mesh->nearestPoint(pos, out, extents)) return ev::null();
        return makeVec3Value(out.x, out.y, out.z);
    });

    b.def("nearestPoint", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.empty()) return ev::null();
        bromath::Vec3 pos = parseVec3(a[0]);
        bromath::Vec3 extents = (a.size() >= 2) ? parseVec3(a[1], brogameagent::NavMesh::kDefaultExtents)
                                                : brogameagent::NavMesh::kDefaultExtents;
        bromath::Vec3 out;
        if (!h->mesh->nearestPoint(pos, out, extents)) return ev::null();
        return makeVec3Value(out.x, out.y, out.z);
    });

    b.def("raycast", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.size() < 2) return ev::null();

        bromath::Vec3 start = parseVec3(a[0]);
        bromath::Vec3 end   = parseVec3(a[1]);
        bromath::Vec3 extents = (a.size() >= 3) ? parseVec3(a[2], brogameagent::NavMesh::kDefaultExtents)
                                                : brogameagent::NavMesh::kDefaultExtents;

        auto hit = h->mesh->raycast(start, end, extents);
        ObjectBuilder res;
        res.set("hit", ev::fromBool(hit.hit));
        res.set("t", ev::fromDouble(hit.t));
        res.set("point", makeVec3Value(hit.point.x, hit.point.y, hit.point.z));
        res.set("normal", makeVec3Value(hit.normal.x, hit.normal.y, hit.normal.z));
        return res.get();
    });

    b.def("buildFromMesh", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.size() < 2) {
            return ev::throwTypeError("buildFromMesh(positions, indices, [opts])");
        }
        // Returned at the end, after reads that allocate.
        ev::Persistent selfP(self);
        std::vector<float> verts;
        std::vector<uint32_t> idx;
        if (!readFloatVector(a[0], verts) || !readU32Vector(a[1], idx) ||
            verts.size() % 3 != 0 || idx.size() % 3 != 0) {
            return ev::throwTypeError("buildFromMesh: positions must be flat xyz triples, indices triangle list");
        }
        brogameagent::NavMeshBakeConfig cfg;
        if (a.size() >= 3 && ev::isObject(a[2])) readBakeConfig(a[2], cfg);
        bool ok = h->mesh->bake(verts.data(), verts.size() / 3, idx.data(), idx.size(), cfg);
        if (!ok) {
            return ev::throwError("buildFromMesh failed: " + h->mesh->lastError());
        }
        if (h->mesh->supportsObstacles()) {
            const auto& hooks = getNavMeshHooks();
            if (hooks.registerNavMeshForPump) hooks.registerNavMeshForPump(h->mesh);
        }
        return selfP.get();
    });

    b.def("save", 0, [](Value self_, std::span<const Value>) -> Value {
        HostNavMesh* h = unwrapNavMesh(self_);
        if (!h || !h->mesh) return ev::throwTypeError("save: invalid NavMesh");
        std::vector<uint8_t> bytes;
        if (!h->mesh->saveTo(bytes)) {
            if (h->mesh->supportsObstacles()) {
                return ev::throwTypeError("save: dynamicObstacles meshes do not serialize");
            }
            return ev::throwTypeError("save: NavMesh is not baked");
        }
        Value ab = ev::createArrayBuffer(static_cast<uint32_t>(bytes.size()));
        if (!ev::isObject(ab)) return ev::null();
        if (auto info = ev::arrayBufferInfo(ab)) {
            if (info.data && !bytes.empty()) {
                std::memcpy(info.data, bytes.data(), bytes.size());
            }
        }
        return ab;
    });

    b.accessor("supportsObstacles", [](Value self_, std::span<const Value>) {
        HostNavMesh* h = unwrapNavMesh(self_);
        return ev::fromBool(h && h->mesh && h->mesh->supportsObstacles());
    }, nullptr);

    b.accessor("obstacleCount", [](Value self_, std::span<const Value>) {
        HostNavMesh* h = unwrapNavMesh(self_);
        return ev::fromDouble(h && h->mesh ? h->mesh->obstacleCount() : 0);
    }, nullptr);

    b.accessor("obstaclesPending", [](Value self_, std::span<const Value>) {
        HostNavMesh* h = unwrapNavMesh(self_);
        return ev::fromBool(h && h->mesh && h->mesh->obstaclesPending());
    }, nullptr);

    b.accessor("generation", [](Value self_, std::span<const Value>) {
        HostNavMesh* h = unwrapNavMesh(self_);
        return ev::fromDouble(h && h->mesh ? h->mesh->generation() : 0);
    }, nullptr);

    b.def("update", 1, [](Value self_, std::span<const Value> a) {
        HostNavMesh* h = unwrapNavMesh(self_);
        if (!h) return ev::undefined();
        if (!h->mesh) return ev::fromBool(true);
        float dt = (a.empty()) ? (1.0f / 60.0f) : static_cast<float>(numAt(a, 0));
        return ev::fromBool(h->mesh->update(dt));
    });

    b.def("addObstacle", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh) return ev::null();
        if (!h->mesh->supportsObstacles()) {
            return ev::throwTypeError("addObstacle: bake the mesh with dynamicObstacles: true");
        }
        if (a.empty()) return ev::throwTypeError("addObstacle(desc)");

        uint32_t id = 0;
        if (ev::isObject(a[0])) {
            ev::Persistent desc(a[0]);
            std::string type = "cylinder";
            Value tv = ev::getProperty(desc.get(), "type");
            if (ev::isString(tv)) type = ev::toUtf8(tv);

            if (type == "cylinder" || type.empty()) {
                Value pv = ev::getProperty(desc.get(), "pos");
                bromath::Vec3 pos = parseVec3(pv);
                float radius = static_cast<float>(getDoubleProperty(desc.get(), "radius", 0.0));
                float height = static_cast<float>(getDoubleProperty(desc.get(), "height", 0.0));
                if (radius <= 0 || height <= 0) {
                    return ev::throwTypeError("addObstacle: cylinder needs {pos, radius > 0, height > 0}");
                }
                id = h->mesh->addObstacle(pos, radius, height);
            } else if (type == "box") {
                ev::Persistent minV(ev::getProperty(desc.get(), "min"));
                ev::Persistent maxV(ev::getProperty(desc.get(), "max"));
                ev::Persistent ctrV(ev::getProperty(desc.get(), "center"));
                ev::Persistent extV(ev::getProperty(desc.get(), "halfExtents"));
                if (ev::isObject(minV.get()) && ev::isObject(maxV.get())) {
                    bromath::Vec3 minPt = parseVec3(minV.get());
                    bromath::Vec3 maxPt = parseVec3(maxV.get());
                    id = h->mesh->addBoxObstacle(minPt, maxPt);
                } else if (ev::isObject(ctrV.get()) && ev::isObject(extV.get())) {
                    bromath::Vec3 center = parseVec3(ctrV.get());
                    bromath::Vec3 halfExtents = parseVec3(extV.get());
                    float yaw = static_cast<float>(getDoubleProperty(desc.get(), "yaw", 0.0));
                    id = h->mesh->addBoxObstacle(center, halfExtents, yaw);
                } else {
                    return ev::throwTypeError("addObstacle: box needs {min, max} or {center, halfExtents, yaw?}");
                }
            } else {
                return ev::throwTypeError("addObstacle: type must be 'cylinder' or 'box'");
            }
        } else if (a.size() >= 3) {
            float x = static_cast<float>(numAt(a, 0));
            float y = static_cast<float>(numAt(a, 1));
            float z = static_cast<float>(numAt(a, 2));
            float radius = (a.size() >= 4) ? static_cast<float>(numAt(a, 3)) : 0.5f;
            float height = (a.size() >= 5) ? static_cast<float>(numAt(a, 4)) : 2.0f;
            id = h->mesh->addObstacle({x, y, z}, radius, height);
        }
        if (id == 0) {
            return ev::throwError("addObstacle failed: " + h->mesh->lastError());
        }
        const auto& hooks = getNavMeshHooks();
        if (hooks.registerNavMeshForPump) hooks.registerNavMeshForPump(h->mesh);
        return ev::fromDouble(id);
    });

    b.def("removeObstacle", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* h = unwrapNavMesh(self);
        if (!h || !h->mesh || a.empty()) return ev::fromBool(false);
        uint32_t id = u32At(a, 0);
        return ev::fromBool(h->mesh->removeObstacle(id));
    });
}

Value makeNavMeshHandle(std::shared_ptr<brogameagent::NavMesh> mesh) {
    auto* h = new HostNavMesh();
    h->mesh = std::move(mesh);

    return g_navMeshClass.make(h, [](void* p) {
        delete static_cast<HostNavMesh*>(p);
    });
}

Value aiBakeNavMesh(Value, std::span<const Value> a) {
    Value opts = a.empty() ? ev::undefined() : a[0];
    if (!ev::isObject(opts)) {
        return ev::throwTypeError("bakeNavMesh(options) requires an options object");
    }

    ev::Persistent root(opts);
    std::vector<float> xyz;
    std::vector<uint32_t> indices;

    ev::Persistent posV(ev::getProperty(root.get(), "positions"));
    if (ev::isUndefined(posV.get()) || ev::isNull(posV.get())) {
        posV.set(ev::getProperty(root.get(), "vertices"));
    }
    ev::Persistent idxV(ev::getProperty(root.get(), "indices"));

    const bool hasPos = !ev::isUndefined(posV.get()) && !ev::isNull(posV.get());
    const bool hasIdx = !ev::isUndefined(idxV.get()) && !ev::isNull(idxV.get());
    if (hasPos != hasIdx) {
        return ev::throwTypeError("bakeNavMesh: positions and indices must be passed together");
    }
    if (hasPos) {
        std::vector<float> verts;
        std::vector<uint32_t> idx;
        bool okP = readFloatVector(posV.get(), verts);
        bool okI = readU32Vector(idxV.get(), idx);
        if (!okP || !okI || verts.size() % 3 != 0 || idx.size() % 3 != 0) {
            return ev::throwTypeError("bakeNavMesh: positions must be flat xyz triples and indices a triangle list");
        }
        uint32_t nVerts = static_cast<uint32_t>(verts.size() / 3);
        for (uint32_t i : idx) {
            if (i >= nVerts) {
                return ev::throwRangeError("bakeNavMesh: index out of range");
            }
        }
        xyz = std::move(verts);
        indices = std::move(idx);
    }

    const auto& hooks = getNavMeshHooks();
    if (hooks.collectGeometry) {
        std::string err;
        if (!hooks.collectGeometry(root.get(), xyz, indices, err)) {
            if (!err.empty()) return ev::throwTypeError(err);
        }
    }

    if (xyz.empty() || indices.empty()) {
        return ev::throwTypeError("bakeNavMesh: no geometry supplied");
    }

    brogameagent::NavMeshBakeConfig cfg;
    readBakeConfig(root.get(), cfg);

    Value linksArr = ev::getProperty(root.get(), "offMeshLinks");
    if (!ev::isUndefined(linksArr) && !ev::isNull(linksArr) && !ev::isObject(linksArr)) {
        return ev::throwTypeError("bakeNavMesh: offMeshLinks must be an array");
    }
    if (ev::isObject(linksArr)) {
        ev::Persistent lRoot(linksArr);
        Value lenV = ev::getProperty(lRoot.get(), "length");
        if (ev::isNumber(lenV)) {
            uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
            cfg.offMeshLinks.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                Value el = ev::getElement(lRoot.get(), i);
                if (!ev::isObject(el)) {
                    return ev::throwTypeError("bakeNavMesh: offMeshLinks entry must be an object");
                }
                ev::Persistent eRoot(el);
                ev::Persistent sv(ev::getProperty(eRoot.get(), "start"));
                ev::Persistent evVal(ev::getProperty(eRoot.get(), "end"));
                if (!ev::isObject(sv.get()) || !ev::isObject(evVal.get())) {
                    return ev::throwTypeError("bakeNavMesh: offMeshLink must have 'start' and 'end' objects");
                }
                brogameagent::NavMeshOffMeshLink link;
                link.start = parseVec3(sv.get());
                link.end   = parseVec3(evVal.get());
                link.radius = static_cast<float>(getDoubleProperty(eRoot.get(), "radius", link.radius));
                link.bidirectional = getBoolProperty(eRoot.get(), "bidirectional", link.bidirectional);
                link.userId = static_cast<uint32_t>(getDoubleProperty(eRoot.get(), "userId", 0.0));
                cfg.offMeshLinks.push_back(link);
            }
        }
    }

    auto mesh = std::make_shared<brogameagent::NavMesh>();
    bool ok = mesh->bake(xyz.data(), xyz.size() / 3, indices.data(), indices.size(), cfg);
    if (!ok) {
        return ev::throwError("bakeNavMesh failed: " + mesh->lastError());
    }

    if (mesh->supportsObstacles() && hooks.registerNavMeshForPump) {
        hooks.registerNavMeshForPump(mesh);
    }

    return makeNavMeshHandle(std::move(mesh));
}

Value aiLoadNavMesh(Value, std::span<const Value> a) {
    if (a.empty()) return ev::throwTypeError("loadNavMesh(buffer) requires a buffer");
    const uint8_t* ptr = nullptr;
    size_t len = 0;
    if (auto info = ev::arrayBufferInfo(a[0])) {
        ptr = static_cast<const uint8_t*>(info.data);
        len = info.byteLength;
    } else if (auto tinfo = ev::typedArrayInfo(a[0])) {
        ptr = static_cast<const uint8_t*>(tinfo.data);
        len = tinfo.byteLength;
    } else {
        return ev::throwTypeError("loadNavMesh: argument must be an ArrayBuffer or TypedArray");
    }

    auto mesh = std::make_shared<brogameagent::NavMesh>();
    if (!mesh->loadFrom(ptr, len)) {
        return ev::throwError("loadNavMesh failed: " + mesh->lastError());
    }

    const auto& hooks = getNavMeshHooks();
    if (mesh->supportsObstacles() && hooks.registerNavMeshForPump) {
        hooks.registerNavMeshForPump(mesh);
    }

    return makeNavMeshHandle(std::move(mesh));
}

Value aiBuildFromMesh(Value, std::span<const Value> a) {
    if (a.size() < 2) {
        return ev::throwTypeError("buildFromMesh(positions, indices, [opts])");
    }
    std::vector<float> verts;
    std::vector<uint32_t> idx;
    if (!readFloatVector(a[0], verts) || !readU32Vector(a[1], idx) ||
        verts.size() % 3 != 0 || idx.size() % 3 != 0) {
        return ev::throwTypeError("buildFromMesh: positions must be flat xyz triples, indices triangle list");
    }
    brogameagent::NavMeshBakeConfig cfg;
    if (a.size() >= 3 && ev::isObject(a[2])) readBakeConfig(a[2], cfg);
    auto mesh = std::make_shared<brogameagent::NavMesh>();
    bool ok = mesh->bake(verts.data(), verts.size() / 3, idx.data(), idx.size(), cfg);
    if (!ok) {
        return ev::throwError("buildFromMesh failed: " + mesh->lastError());
    }
    return makeNavMeshHandle(std::move(mesh));
}

#else

void decorateNavMeshProto(ObjectBuilder&) {}

Value makeNavMeshHandle(std::shared_ptr<brogameagent::NavMesh>) {
    return ev::throwError("NavMesh is not available in this build (requires recastnavigation)");
}

Value aiBakeNavMesh(Value, std::span<const Value>) {
    return ev::throwError("bakeNavMesh: NavMesh is not available in this build (requires recastnavigation)");
}

Value aiLoadNavMesh(Value, std::span<const Value>) {
    return ev::throwError("loadNavMesh: NavMesh is not available in this build (requires recastnavigation)");
}

Value aiBuildFromMesh(Value, std::span<const Value>) {
    return ev::throwError("buildFromMesh: NavMesh is not available in this build (requires recastnavigation)");
}

#endif

} // namespace brogameagent::api
