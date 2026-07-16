// NavMesh — Recast/Detour adapter. All Recast/Detour usage lives in this
// translation unit; the public header speaks brogameagent/bromath types only.

#include <brogameagent/nav_mesh.h>

#include <Recast.h>
#include <DetourAlloc.h>
#include <DetourCommon.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>
#include <DetourStatus.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <mutex>

namespace brogameagent {

namespace {

// Recast build context that collects log messages so a failed bake can be
// diagnosed from NavMesh::lastError() instead of vanishing into the void.
class LogContext : public rcContext {
public:
    LogContext() : rcContext(/*state=*/true) {}
    std::string messages;

protected:
    void doLog(rcLogCategory category, const char* msg, int len) override {
        (void)len;
        if (!messages.empty()) messages += "; ";
        switch (category) {
            case RC_LOG_WARNING: messages += "warning: "; break;
            case RC_LOG_ERROR:   messages += "error: ";   break;
            default: break;
        }
        messages += msg;
    }
};

constexpr unsigned short kPolyFlagWalk = 0x01;
constexpr int kMaxPathPolys = 256;
constexpr int kMaxStraightPoints = 256;
constexpr int kQueryNodePoolSize = 2048;

inline bromath::Vec3 toVec3(const float* p) { return {p[0], p[1], p[2]}; }

// Detour's findRandomPoint takes a stateless float(*)() — thread a seeded
// LCG through thread-local state so randomPoint() stays deterministic per
// seed and race-free across threads.
thread_local uint32_t t_rngState = 1u;
float frand01() {
    t_rngState = t_rngState * 1664525u + 1013904223u;
    return static_cast<float>(t_rngState >> 8) * (1.0f / 16777216.0f);
}

} // namespace

struct NavMesh::Impl {
    dtNavMesh* mesh = nullptr;
    dtNavMeshQuery* query = nullptr;
    std::vector<uint8_t> blob;   // raw Detour tile data — the saveTo() payload
    std::string error;
    mutable std::mutex queryMutex; // dtNavMeshQuery is stateful; serialize queries

    ~Impl() { reset(); }

    void reset() {
        if (query) { dtFreeNavMeshQuery(query); query = nullptr; }
        if (mesh) { dtFreeNavMesh(mesh); mesh = nullptr; }
        blob.clear();
    }

    // Stand up dtNavMesh + dtNavMeshQuery from `data` (a raw single-tile
    // Detour blob of `size` bytes). Takes its own dtAlloc'd copy — Detour
    // frees tile data with dtFree, so the buffer must come from dtAlloc.
    bool initFromData(const uint8_t* data, size_t size) {
        reset();
        if (!data || size == 0) { error = "empty navmesh data"; return false; }

        void* copy = dtAlloc(size, DT_ALLOC_PERM);
        if (!copy) { error = "navmesh data allocation failed"; return false; }
        std::memcpy(copy, data, size);

        mesh = dtAllocNavMesh();
        dtStatus status = mesh->init(static_cast<unsigned char*>(copy),
                                     static_cast<int>(size), DT_TILE_FREE_DATA);
        if (dtStatusFailed(status)) {
            dtFree(copy); // init did not take ownership on failure
            dtFreeNavMesh(mesh); mesh = nullptr;
            error = (status & DT_WRONG_MAGIC)   ? "navmesh data: wrong magic (not a navmesh blob)"
                  : (status & DT_WRONG_VERSION) ? "navmesh data: unsupported version"
                                                : "navmesh init failed";
            return false;
        }

        query = dtAllocNavMeshQuery();
        status = query->init(mesh, kQueryNodePoolSize);
        if (dtStatusFailed(status)) {
            reset();
            error = "navmesh query init failed";
            return false;
        }

        blob.assign(data, data + size);
        error.clear();
        return true;
    }
};

NavMesh::NavMesh() : impl_(std::make_unique<Impl>()) {}
NavMesh::~NavMesh() = default;
NavMesh::NavMesh(NavMesh&&) noexcept = default;
NavMesh& NavMesh::operator=(NavMesh&&) noexcept = default;

bool NavMesh::valid() const { return impl_->mesh != nullptr; }
const std::string& NavMesh::lastError() const { return impl_->error; }

// ─── Bake ────────────────────────────────────────────────────────────────────

bool NavMesh::bake(const float* vertices, size_t vertexCount,
                   const uint32_t* indices, size_t indexCount,
                   const NavMeshBakeConfig& config) {
    Impl& im = *impl_;
    im.error.clear();

    if (!vertices || vertexCount < 3) { im.error = "bake: need at least 3 vertices"; return false; }
    if (!indices || indexCount < 3 || indexCount % 3 != 0) {
        im.error = "bake: indexCount must be a positive multiple of 3";
        return false;
    }
    if (config.cellSize <= 0 || config.cellHeight <= 0) {
        im.error = "bake: cellSize and cellHeight must be positive";
        return false;
    }

    const int nverts = static_cast<int>(vertexCount);
    const int ntris = static_cast<int>(indexCount / 3);

    // Recast wants signed int indices.
    std::vector<int> tris(indexCount);
    for (size_t i = 0; i < indexCount; i++) {
        if (indices[i] >= vertexCount) { im.error = "bake: index out of range"; return false; }
        tris[i] = static_cast<int>(indices[i]);
    }

    rcConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.cs = config.cellSize;
    cfg.ch = config.cellHeight;
    cfg.walkableSlopeAngle = config.agentMaxSlopeDeg;
    cfg.walkableHeight = static_cast<int>(std::ceil(config.agentHeight / cfg.ch));
    cfg.walkableClimb = static_cast<int>(std::floor(config.agentMaxClimb / cfg.ch));
    cfg.walkableRadius = static_cast<int>(std::ceil(config.agentRadius / cfg.cs));
    cfg.maxEdgeLen = static_cast<int>(config.edgeMaxLen / cfg.cs);
    cfg.maxSimplificationError = config.edgeMaxError;
    cfg.minRegionArea = static_cast<int>(config.regionMinSize * config.regionMinSize);
    cfg.mergeRegionArea = static_cast<int>(config.regionMergeSize * config.regionMergeSize);
    cfg.maxVertsPerPoly = 6;
    cfg.detailSampleDist = config.detailSampleDist < 0.9f ? 0 : cfg.cs * config.detailSampleDist;
    cfg.detailSampleMaxError = cfg.ch * config.detailSampleMaxError;

    rcCalcBounds(vertices, nverts, cfg.bmin, cfg.bmax);
    rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);

    LogContext ctx;

    // Recast allocations we must free on every exit path.
    rcHeightfield* hf = nullptr;
    rcCompactHeightfield* chf = nullptr;
    rcContourSet* cset = nullptr;
    rcPolyMesh* pmesh = nullptr;
    rcPolyMeshDetail* dmesh = nullptr;
    auto fail = [&](const char* stage) {
        im.error = std::string("bake: ") + stage;
        if (!ctx.messages.empty()) im.error += " [" + ctx.messages + "]";
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(hf);
        return false;
    };

    // 1. Voxelize the walkable triangles into a heightfield.
    hf = rcAllocHeightfield();
    if (!hf || !rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height,
                                    cfg.bmin, cfg.bmax, cfg.cs, cfg.ch))
        return fail("heightfield creation failed");

    std::vector<unsigned char> triAreas(static_cast<size_t>(ntris), 0);
    rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, vertices, nverts,
                            tris.data(), ntris, triAreas.data());
    if (!rcRasterizeTriangles(&ctx, vertices, nverts, tris.data(),
                              triAreas.data(), ntris, *hf, cfg.walkableClimb))
        return fail("triangle rasterization failed");

    // 2. Filter out spans an agent can't actually stand on.
    rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
    rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
    rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);

    // 3. Compact + erode by agent radius, partition into regions.
    chf = rcAllocCompactHeightfield();
    if (!chf || !rcBuildCompactHeightfield(&ctx, cfg.walkableHeight,
                                           cfg.walkableClimb, *hf, *chf))
        return fail("compact heightfield failed");
    rcFreeHeightField(hf); hf = nullptr;

    if (!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf))
        return fail("walkable-area erosion failed");
    if (!rcBuildDistanceField(&ctx, *chf))
        return fail("distance field failed");
    if (!rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea))
        return fail("region build failed");

    // 4. Regions → contours → polygon mesh (+ detail heights).
    cset = rcAllocContourSet();
    if (!cset || !rcBuildContours(&ctx, *chf, cfg.maxSimplificationError,
                                  cfg.maxEdgeLen, *cset))
        return fail("contour build failed");

    pmesh = rcAllocPolyMesh();
    if (!pmesh || !rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *pmesh))
        return fail("poly mesh build failed");

    dmesh = rcAllocPolyMeshDetail();
    if (!dmesh || !rcBuildPolyMeshDetail(&ctx, *pmesh, *chf, cfg.detailSampleDist,
                                         cfg.detailSampleMaxError, *dmesh))
        return fail("detail mesh build failed");

    rcFreeCompactHeightfield(chf); chf = nullptr;
    rcFreeContourSet(cset); cset = nullptr;

    if (pmesh->npolys == 0)
        return fail("no walkable polygons (soup too small for the agent, or all "
                    "triangles steeper than agentMaxSlopeDeg / wound clockwise)");

    // Every generated poly is walkable — a single flag; per-area costs are
    // future work alongside tiled bakes.
    for (int i = 0; i < pmesh->npolys; i++) pmesh->flags[i] = kPolyFlagWalk;

    // 5. Poly mesh → Detour navmesh data.
    dtNavMeshCreateParams params;
    std::memset(&params, 0, sizeof(params));
    params.verts = pmesh->verts;
    params.vertCount = pmesh->nverts;
    params.polys = pmesh->polys;
    params.polyAreas = pmesh->areas;
    params.polyFlags = pmesh->flags;
    params.polyCount = pmesh->npolys;
    params.nvp = pmesh->nvp;
    params.detailMeshes = dmesh->meshes;
    params.detailVerts = dmesh->verts;
    params.detailVertsCount = dmesh->nverts;
    params.detailTris = dmesh->tris;
    params.detailTriCount = dmesh->ntris;
    params.walkableHeight = config.agentHeight;
    params.walkableRadius = config.agentRadius;
    params.walkableClimb = config.agentMaxClimb;
    rcVcopy(params.bmin, pmesh->bmin);
    rcVcopy(params.bmax, pmesh->bmax);
    params.cs = cfg.cs;
    params.ch = cfg.ch;
    params.buildBvTree = true;

    unsigned char* navData = nullptr;
    int navDataSize = 0;
    if (!dtCreateNavMeshData(&params, &navData, &navDataSize))
        return fail("Detour navmesh data creation failed");

    rcFreePolyMeshDetail(dmesh); dmesh = nullptr;
    rcFreePolyMesh(pmesh); pmesh = nullptr;

    const bool ok = im.initFromData(navData, static_cast<size_t>(navDataSize));
    dtFree(navData);
    return ok;
}

// ─── Queries ─────────────────────────────────────────────────────────────────

std::vector<bromath::Vec3> NavMesh::findPath(bromath::Vec3 start, bromath::Vec3 end,
                                             bromath::Vec3 searchExtents) const {
    Impl& im = *impl_;
    if (!im.query) return {};
    std::lock_guard<std::mutex> lock(im.queryMutex);

    const dtQueryFilter filter; // default: all flags pass, uniform cost
    const float sp[3] = {start.x, start.y, start.z};
    const float ep[3] = {end.x, end.y, end.z};
    const float ext[3] = {searchExtents.x, searchExtents.y, searchExtents.z};

    dtPolyRef startRef = 0, endRef = 0;
    float snappedStart[3], snappedEnd[3];
    im.query->findNearestPoly(sp, ext, &filter, &startRef, snappedStart);
    im.query->findNearestPoly(ep, ext, &filter, &endRef, snappedEnd);
    if (!startRef || !endRef) return {};

    dtPolyRef polys[kMaxPathPolys];
    int npolys = 0;
    dtStatus status = im.query->findPath(startRef, endRef, snappedStart, snappedEnd,
                                         &filter, polys, &npolys, kMaxPathPolys);
    if (dtStatusFailed(status) || npolys == 0) return {};
    // Partial result (goal unreachable, or corridor overflowed the poly
    // buffer): report failure rather than a silently truncated path.
    if ((status & DT_PARTIAL_RESULT) || polys[npolys - 1] != endRef) return {};

    float straight[kMaxStraightPoints * 3];
    int nstraight = 0;
    status = im.query->findStraightPath(snappedStart, snappedEnd, polys, npolys,
                                        straight, nullptr, nullptr,
                                        &nstraight, kMaxStraightPoints);
    if (dtStatusFailed(status) || nstraight == 0) return {};

    std::vector<bromath::Vec3> out;
    out.reserve(static_cast<size_t>(nstraight));
    for (int i = 0; i < nstraight; i++) out.push_back(toVec3(&straight[i * 3]));
    return out;
}

bool NavMesh::nearestPoint(bromath::Vec3 p, bromath::Vec3& out,
                           bromath::Vec3 searchExtents) const {
    Impl& im = *impl_;
    if (!im.query) return false;
    std::lock_guard<std::mutex> lock(im.queryMutex);

    const dtQueryFilter filter;
    const float pt[3] = {p.x, p.y, p.z};
    const float ext[3] = {searchExtents.x, searchExtents.y, searchExtents.z};
    dtPolyRef ref = 0;
    float nearest[3];
    dtStatus status = im.query->findNearestPoly(pt, ext, &filter, &ref, nearest);
    if (dtStatusFailed(status) || !ref) return false;
    out = toVec3(nearest);
    return true;
}

NavMeshRaycastHit NavMesh::raycast(bromath::Vec3 start, bromath::Vec3 end,
                                   bromath::Vec3 searchExtents) const {
    NavMeshRaycastHit result;
    Impl& im = *impl_;
    if (!im.query) { result.t = 0; return result; }
    std::lock_guard<std::mutex> lock(im.queryMutex);

    const dtQueryFilter filter;
    const float sp[3] = {start.x, start.y, start.z};
    const float ext[3] = {searchExtents.x, searchExtents.y, searchExtents.z};
    dtPolyRef startRef = 0;
    float snapped[3];
    im.query->findNearestPoly(sp, ext, &filter, &startRef, snapped);
    if (!startRef) { result.t = 0; return result; }

    const float ep[3] = {end.x, end.y, end.z};
    float t = 0;
    float normal[3] = {0, 0, 0};
    dtPolyRef polys[kMaxPathPolys];
    int npolys = 0;
    dtStatus status = im.query->raycast(startRef, snapped, ep, &filter,
                                        &t, normal, polys, &npolys, kMaxPathPolys);
    if (dtStatusFailed(status)) { result.t = 0; return result; }

    if (t >= FLT_MAX) {
        // Reached the end unobstructed.
        result.hit = false;
        result.t = 1.0f;
        result.point = end;
    } else {
        result.hit = true;
        result.t = t;
        result.point = {snapped[0] + (ep[0] - snapped[0]) * t,
                        snapped[1] + (ep[1] - snapped[1]) * t,
                        snapped[2] + (ep[2] - snapped[2]) * t};
        result.normal = toVec3(normal);
    }
    return result;
}

bool NavMesh::randomPoint(uint32_t seed, bromath::Vec3& out) const {
    Impl& im = *impl_;
    if (!im.query) return false;
    std::lock_guard<std::mutex> lock(im.queryMutex);

    t_rngState = seed * 2654435761u + 1u; // splash the seed; 0 stays valid
    const dtQueryFilter filter;
    dtPolyRef ref = 0;
    float pt[3];
    dtStatus status = im.query->findRandomPoint(&filter, &frand01, &ref, pt);
    if (dtStatusFailed(status) || !ref) return false;
    out = toVec3(pt);
    return true;
}

// ─── Serialization ───────────────────────────────────────────────────────────

bool NavMesh::saveTo(std::vector<uint8_t>& out) const {
    if (!valid()) return false;
    out = impl_->blob;
    return true;
}

bool NavMesh::loadFrom(const uint8_t* data, size_t size) {
    return impl_->initFromData(data, size);
}

} // namespace brogameagent
