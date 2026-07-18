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
#include <DetourTileCache.h>
#include <DetourTileCacheBuilder.h>

#include <algorithm>
#include <atomic>
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
constexpr int kExpectedLayersPerTile = 4;

// ─── dtTileCache glue ───────────────────────────────────────────────────────
// The library proper ships no compressor (fastlz lives in RecastDemo's
// contrib, which is sample code, not the library). Obstacle layers are tiny
// (a few KB per tile), so storing them uncompressed costs little and keeps
// the dependency surface at the library itself.
struct StoreCompressor final : public dtTileCacheCompressor {
    int maxCompressedSize(const int bufferSize) override { return bufferSize; }
    dtStatus compress(const unsigned char* buffer, const int bufferSize,
                      unsigned char* compressed, const int maxSize,
                      int* compressedSize) override {
        if (maxSize < bufferSize) return DT_FAILURE | DT_BUFFER_TOO_SMALL;
        std::memcpy(compressed, buffer, static_cast<size_t>(bufferSize));
        *compressedSize = bufferSize;
        return DT_SUCCESS;
    }
    dtStatus decompress(const unsigned char* compressed, const int compressedSize,
                        unsigned char* buffer, const int maxBufferSize,
                        int* bufferSize) override {
        if (maxBufferSize < compressedSize) return DT_FAILURE | DT_BUFFER_TOO_SMALL;
        std::memcpy(buffer, compressed, static_cast<size_t>(compressedSize));
        *bufferSize = compressedSize;
        return DT_SUCCESS;
    }
};

// Marks every surviving tile-cache poly walkable — mirrors the static path's
// `pmesh->flags[i] = kPolyFlagWalk` loop. Runs on every incremental tile
// rebuild, so obstacle-carved rebuilds keep the same flag scheme.
struct WalkableMeshProcess final : public dtTileCacheMeshProcess {
    void process(struct dtNavMeshCreateParams* params,
                 unsigned char* polyAreas, unsigned short* polyFlags) override {
        for (int i = 0; i < params->polyCount; i++) {
            polyFlags[i] =
                (polyAreas[i] == DT_TILECACHE_NULL_AREA) ? 0 : kPolyFlagWalk;
        }
    }
};

inline bromath::Vec3 toVec3(const float* p) { return {p[0], p[1], p[2]}; }

// Detour's findRandomPoint takes a stateless float(*)() — thread a seeded
// LCG through thread-local state so randomPoint() stays deterministic per
// seed and race-free across threads.
thread_local uint32_t t_rngState = 1u;
float frand01() {
    t_rngState = t_rngState * 1664525u + 1013904223u;
    return static_cast<float>(t_rngState >> 8) * (1.0f / 16777216.0f);
}

// Map NavMeshBakeConfig onto the shared rcConfig fields (bounds/grid are
// filled by the caller — they differ between the static and tiled paths).
rcConfig makeRcConfig(const brogameagent::NavMeshBakeConfig& config) {
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
    return cfg;
}

// One compressed tile-cache layer blob (dtAlloc'd; ownership passes to
// dtTileCache::addTile on success).
struct TileLayerData {
    unsigned char* data = nullptr;
    int size = 0;
};

// Rasterize one tile's heightfield layers into compressed tile-cache blobs.
// `baseCfg` carries the whole-soup bounds in bmin/bmax and the per-tile grid
// dims in width/height/tileSize/borderSize. `triBounds` are per-triangle XZ
// AABBs (minX, minZ, maxX, maxZ) for cheap tile culling. Appends to `out`;
// returns false on hard failure (details in ctx's log).
bool rasterizeTileLayers(rcContext* ctx,
                         const float* verts, int nverts,
                         const std::vector<int>& tris,
                         const std::vector<float>& triBounds,
                         const rcConfig& baseCfg,
                         dtTileCacheCompressor* comp,
                         int tx, int ty,
                         std::vector<TileLayerData>& out) {
    rcConfig cfg = baseCfg;
    const float tcs = static_cast<float>(baseCfg.tileSize) * baseCfg.cs;
    const float border = static_cast<float>(baseCfg.borderSize) * baseCfg.cs;
    cfg.bmin[0] = baseCfg.bmin[0] + static_cast<float>(tx) * tcs - border;
    cfg.bmin[1] = baseCfg.bmin[1];
    cfg.bmin[2] = baseCfg.bmin[2] + static_cast<float>(ty) * tcs - border;
    cfg.bmax[0] = baseCfg.bmin[0] + static_cast<float>(tx + 1) * tcs + border;
    cfg.bmax[1] = baseCfg.bmax[1];
    cfg.bmax[2] = baseCfg.bmin[2] + static_cast<float>(ty + 1) * tcs + border;

    rcHeightfield* hf = rcAllocHeightfield();
    rcCompactHeightfield* chf = nullptr;
    rcHeightfieldLayerSet* lset = nullptr;
    auto cleanup = [&]() {
        rcFreeHeightfieldLayerSet(lset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(hf);
    };

    if (!hf || !rcCreateHeightfield(ctx, *hf, cfg.width, cfg.height,
                                    cfg.bmin, cfg.bmax, cfg.cs, cfg.ch)) {
        cleanup();
        return false;
    }

    // Triangles overlapping this tile's expanded XZ bounds.
    std::vector<int> tileTris;
    const int ntris = static_cast<int>(tris.size() / 3);
    for (int t = 0; t < ntris; t++) {
        const float* b = &triBounds[static_cast<size_t>(t) * 4];
        if (b[2] < cfg.bmin[0] || b[0] > cfg.bmax[0] ||
            b[3] < cfg.bmin[2] || b[1] > cfg.bmax[2]) continue;
        tileTris.push_back(tris[static_cast<size_t>(t) * 3]);
        tileTris.push_back(tris[static_cast<size_t>(t) * 3 + 1]);
        tileTris.push_back(tris[static_cast<size_t>(t) * 3 + 2]);
    }
    if (!tileTris.empty()) {
        const int n = static_cast<int>(tileTris.size() / 3);
        std::vector<unsigned char> areas(static_cast<size_t>(n), 0);
        rcMarkWalkableTriangles(ctx, cfg.walkableSlopeAngle, verts, nverts,
                                tileTris.data(), n, areas.data());
        if (!rcRasterizeTriangles(ctx, verts, nverts, tileTris.data(),
                                  areas.data(), n, *hf, cfg.walkableClimb)) {
            cleanup();
            return false;
        }
    }

    rcFilterLowHangingWalkableObstacles(ctx, cfg.walkableClimb, *hf);
    rcFilterLedgeSpans(ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
    rcFilterWalkableLowHeightSpans(ctx, cfg.walkableHeight, *hf);

    chf = rcAllocCompactHeightfield();
    if (!chf || !rcBuildCompactHeightfield(ctx, cfg.walkableHeight,
                                           cfg.walkableClimb, *hf, *chf)) {
        cleanup();
        return false;
    }
    if (!rcErodeWalkableArea(ctx, cfg.walkableRadius, *chf)) {
        cleanup();
        return false;
    }

    lset = rcAllocHeightfieldLayerSet();
    if (!lset || !rcBuildHeightfieldLayers(ctx, *chf, cfg.borderSize,
                                           cfg.walkableHeight, *lset)) {
        cleanup();
        return false;
    }

    for (int i = 0; i < lset->nlayers; i++) {
        const rcHeightfieldLayer* layer = &lset->layers[i];
        dtTileCacheLayerHeader header;
        std::memset(&header, 0, sizeof(header));
        header.magic = DT_TILECACHE_MAGIC;
        header.version = DT_TILECACHE_VERSION;
        header.tx = tx;
        header.ty = ty;
        header.tlayer = i;
        dtVcopy(header.bmin, layer->bmin);
        dtVcopy(header.bmax, layer->bmax);
        header.width = static_cast<unsigned char>(layer->width);
        header.height = static_cast<unsigned char>(layer->height);
        header.minx = static_cast<unsigned char>(layer->minx);
        header.maxx = static_cast<unsigned char>(layer->maxx);
        header.miny = static_cast<unsigned char>(layer->miny);
        header.maxy = static_cast<unsigned char>(layer->maxy);
        header.hmin = static_cast<unsigned short>(layer->hmin);
        header.hmax = static_cast<unsigned short>(layer->hmax);

        TileLayerData tile;
        if (dtStatusFailed(dtBuildTileCacheLayer(comp, &header, layer->heights,
                                                 layer->areas, layer->cons,
                                                 &tile.data, &tile.size))) {
            cleanup();
            return false;
        }
        out.push_back(tile);
    }
    cleanup();
    return true;
}

} // namespace

struct NavMesh::Impl {
    dtNavMesh* mesh = nullptr;
    dtNavMeshQuery* query = nullptr;
    std::vector<uint8_t> blob;   // raw Detour tile data — the saveTo() payload
    std::string error;
    mutable std::mutex queryMutex; // dtNavMeshQuery is stateful; serialize queries

    // Dynamic-obstacle state (dynamicObstacles bakes only). The compressor /
    // allocator / mesh-process objects are referenced by the tile cache for
    // its whole life, so they live here and tileCache is freed first.
    dtTileCache* tileCache = nullptr;
    StoreCompressor compressor;
    dtTileCacheAlloc tcAlloc;    // default dtAlloc-backed allocator
    WalkableMeshProcess meshProcess;
    std::vector<uint32_t> liveObstacles;   // handles added and not yet removed
    std::atomic<bool> obstaclesDirty{false};   // add/remove seen, update() not yet done
    std::atomic<uint32_t> generation{0};   // bumps when the surface changes

    ~Impl() { reset(); }

    void reset() {
        if (query) { dtFreeNavMeshQuery(query); query = nullptr; }
        if (mesh) { dtFreeNavMesh(mesh); mesh = nullptr; }
        if (tileCache) { dtFreeTileCache(tileCache); tileCache = nullptr; }
        liveObstacles.clear();
        obstaclesDirty.store(false, std::memory_order_relaxed);
        // generation is deliberately NOT reset: it is monotonic per NavMesh
        // object, so a re-bake reads as a surface change to path followers.
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
        generation.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Stand up dtNavMeshQuery on an already-initialized mesh (tiled path).
    bool initQuery() {
        query = dtAllocNavMeshQuery();
        dtStatus status = query->init(mesh, kQueryNodePoolSize);
        if (dtStatusFailed(status)) {
            reset();
            error = "navmesh query init failed";
            return false;
        }
        return true;
    }

    // Tiled (dynamic-obstacle) bake: the soup becomes a grid of compressed
    // dtTileCache layers, each built into the dtNavMesh — the layout
    // dtTileCache needs to rebuild individual tiles when obstacles change.
    bool bakeTiled(const float* vertices, int nverts,
                   const std::vector<int>& tris,
                   const NavMeshBakeConfig& config) {
        reset();

        if (!(config.tileSize > 0)) {
            error = "bake: tileSize must be positive";
            return false;
        }

        rcConfig cfg = makeRcConfig(config);
        rcCalcBounds(vertices, nverts, cfg.bmin, cfg.bmax);
        int gw = 0, gh = 0;
        rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &gw, &gh);

        // Tile edge in cells. dtTileCacheLayerHeader stores layer dims in
        // unsigned char, so a tile is capped at 255 cells (border included).
        int ts = static_cast<int>(std::lround(config.tileSize / cfg.cs));
        cfg.borderSize = cfg.walkableRadius + 3;
        if (ts < 16) ts = 16;
        if (ts > 255 - 2 * cfg.borderSize) ts = 255 - 2 * cfg.borderSize;
        cfg.tileSize = ts;
        cfg.width = ts + cfg.borderSize * 2;
        cfg.height = ts + cfg.borderSize * 2;

        const int tw = (gw + ts - 1) / ts;
        const int th = (gh + ts - 1) / ts;

        // Tile cache sized for the grid (a few vertical layers per tile).
        dtTileCacheParams tcp;
        std::memset(&tcp, 0, sizeof(tcp));
        rcVcopy(tcp.orig, cfg.bmin);
        tcp.cs = cfg.cs;
        tcp.ch = cfg.ch;
        tcp.width = ts;
        tcp.height = ts;
        tcp.walkableHeight = config.agentHeight;
        tcp.walkableRadius = config.agentRadius;
        tcp.walkableClimb = config.agentMaxClimb;
        tcp.maxSimplificationError = cfg.maxSimplificationError;
        tcp.maxTiles = tw * th * kExpectedLayersPerTile;
        tcp.maxObstacles = config.maxObstacles < 1 ? 1
                         : config.maxObstacles > 0xffff ? 0xffff
                         : config.maxObstacles;

        tileCache = dtAllocTileCache();
        if (!tileCache ||
            dtStatusFailed(tileCache->init(&tcp, &tcAlloc, &compressor, &meshProcess))) {
            reset();
            error = "bake: tile cache init failed";
            return false;
        }

        // Navmesh sized for the tile grid (tile/poly ref-bit split as in the
        // Detour docs: 22 id bits shared between tiles and polys).
        dtNavMeshParams nmp;
        std::memset(&nmp, 0, sizeof(nmp));
        rcVcopy(nmp.orig, cfg.bmin);
        nmp.tileWidth = static_cast<float>(ts) * cfg.cs;
        nmp.tileHeight = static_cast<float>(ts) * cfg.cs;
        int tileBits = static_cast<int>(dtIlog2(dtNextPow2(
            static_cast<unsigned int>(tw * th * kExpectedLayersPerTile))));
        if (tileBits > 14) tileBits = 14;
        nmp.maxTiles = 1 << tileBits;
        nmp.maxPolys = 1 << (22 - tileBits);

        mesh = dtAllocNavMesh();
        if (!mesh || dtStatusFailed(mesh->init(&nmp))) {
            reset();
            error = "bake: tiled navmesh init failed";
            return false;
        }

        // Per-triangle XZ bounds for tile culling.
        const int ntris = static_cast<int>(tris.size() / 3);
        std::vector<float> triBounds(static_cast<size_t>(ntris) * 4);
        for (int t = 0; t < ntris; t++) {
            float minX = FLT_MAX, minZ = FLT_MAX, maxX = -FLT_MAX, maxZ = -FLT_MAX;
            for (int k = 0; k < 3; k++) {
                const float* v = &vertices[static_cast<size_t>(tris[static_cast<size_t>(t) * 3 + k]) * 3];
                minX = std::min(minX, v[0]); maxX = std::max(maxX, v[0]);
                minZ = std::min(minZ, v[2]); maxZ = std::max(maxZ, v[2]);
            }
            float* b = &triBounds[static_cast<size_t>(t) * 4];
            b[0] = minX; b[1] = minZ; b[2] = maxX; b[3] = maxZ;
        }

        LogContext ctx;
        auto fail = [&](const char* stage) {
            error = std::string("bake: ") + stage;
            if (!ctx.messages.empty()) error += " [" + ctx.messages + "]";
            reset();
            return false;
        };

        for (int ty = 0; ty < th; ty++) {
            for (int tx = 0; tx < tw; tx++) {
                std::vector<TileLayerData> layers;
                if (!rasterizeTileLayers(&ctx, vertices, nverts, tris, triBounds,
                                         cfg, &compressor, tx, ty, layers)) {
                    for (const TileLayerData& l : layers) dtFree(l.data);
                    return fail("tile layer rasterization failed");
                }
                for (const TileLayerData& l : layers) {
                    dtStatus st = tileCache->addTile(l.data, l.size,
                                                     DT_COMPRESSEDTILE_FREE_DATA, nullptr);
                    if (dtStatusFailed(st)) {
                        dtFree(l.data);
                        return fail("tile cache addTile failed");
                    }
                }
                if (dtStatusFailed(tileCache->buildNavMeshTilesAt(tx, ty, mesh)))
                    return fail("tile navmesh build failed");
            }
        }

        // Reject an all-unwalkable bake (same contract as the static path).
        int npolys = 0;
        const dtNavMesh* cmesh = mesh;
        for (int i = 0; i < cmesh->getMaxTiles(); i++) {
            const dtMeshTile* t = cmesh->getTile(i);
            if (t && t->header) npolys += t->header->polyCount;
        }
        if (npolys == 0)
            return fail("no walkable polygons (soup too small for the agent, or all "
                        "triangles steeper than agentMaxSlopeDeg / wound clockwise)");

        if (!initQuery()) return false;
        error.clear();
        generation.fetch_add(1, std::memory_order_relaxed);
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

    // Dynamic-obstacle bakes take the tiled tile-cache path.
    if (config.dynamicObstacles)
        return im.bakeTiled(vertices, nverts, tris, config);

    rcConfig cfg = makeRcConfig(config);
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
    return findPathEx(start, end, searchExtents).points;
}

NavMeshPath NavMesh::findPathEx(bromath::Vec3 start, bromath::Vec3 end,
                                bromath::Vec3 searchExtents,
                                bool requireFullPath) const {
    NavMeshPath out;
    Impl& im = *impl_;
    if (!im.query) return out;
    std::lock_guard<std::mutex> lock(im.queryMutex);

    const dtQueryFilter filter; // default: all flags pass, uniform cost
    const float sp[3] = {start.x, start.y, start.z};
    const float ep[3] = {end.x, end.y, end.z};
    const float ext[3] = {searchExtents.x, searchExtents.y, searchExtents.z};

    dtPolyRef startRef = 0, endRef = 0;
    float snappedStart[3], snappedEnd[3];
    im.query->findNearestPoly(sp, ext, &filter, &startRef, snappedStart);
    im.query->findNearestPoly(ep, ext, &filter, &endRef, snappedEnd);
    if (!startRef || !endRef) return out;  // snap failure: empty, !partial

    dtPolyRef polys[kMaxPathPolys];
    int npolys = 0;
    dtStatus status = im.query->findPath(startRef, endRef, snappedStart, snappedEnd,
                                         &filter, polys, &npolys, kMaxPathPolys);
    if (dtStatusFailed(status) || npolys == 0) return out;

    // Partial result: goal unreachable (or the corridor overflowed the poly
    // buffer). Clamp the goal to the closest reachable point on the last
    // corridor poly — Detour guarantees it is the poly closest to the goal —
    // and report partial so callers can distinguish clamped from complete.
    out.partial = (status & DT_PARTIAL_RESULT) != 0 || polys[npolys - 1] != endRef;
    if (out.partial && requireFullPath) return out;  // empty points, partial=true

    float target[3] = {snappedEnd[0], snappedEnd[1], snappedEnd[2]};
    if (out.partial)
        im.query->closestPointOnPoly(polys[npolys - 1], snappedEnd, target, nullptr);

    float straight[kMaxStraightPoints * 3];
    int nstraight = 0;
    status = im.query->findStraightPath(snappedStart, target, polys, npolys,
                                        straight, nullptr, nullptr,
                                        &nstraight, kMaxStraightPoints);
    if (dtStatusFailed(status) || nstraight == 0) return out;

    out.points.reserve(static_cast<size_t>(nstraight));
    for (int i = 0; i < nstraight; i++) out.points.push_back(toVec3(&straight[i * 3]));
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

// ─── Dynamic obstacles ───────────────────────────────────────────────────────

namespace {
const char* obstacleAddError(dtStatus st) {
    if (st & DT_BUFFER_TOO_SMALL)
        return "addObstacle: request queue full — pump update() and retry";
    if (st & DT_OUT_OF_MEMORY)
        return "addObstacle: out of obstacle slots (raise NavMeshBakeConfig::maxObstacles)";
    return "addObstacle failed";
}
} // namespace

bool NavMesh::supportsObstacles() const {
    return impl_->tileCache != nullptr;
}

NavMesh::ObstacleId NavMesh::addObstacle(bromath::Vec3 pos, float radius, float height) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    if (!im.tileCache) {
        im.error = "addObstacle: mesh was not baked with dynamicObstacles";
        return 0;
    }
    const float p[3] = {pos.x, pos.y, pos.z};
    dtObstacleRef ref = 0;
    dtStatus st = im.tileCache->addObstacle(p, radius, height, &ref);
    if (dtStatusFailed(st)) { im.error = obstacleAddError(st); return 0; }
    im.liveObstacles.push_back(ref);
    im.obstaclesDirty.store(true, std::memory_order_relaxed);
    return ref;
}

NavMesh::ObstacleId NavMesh::addBoxObstacle(bromath::Vec3 bmin, bromath::Vec3 bmax) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    if (!im.tileCache) {
        im.error = "addObstacle: mesh was not baked with dynamicObstacles";
        return 0;
    }
    const float lo[3] = {bmin.x, bmin.y, bmin.z};
    const float hi[3] = {bmax.x, bmax.y, bmax.z};
    dtObstacleRef ref = 0;
    dtStatus st = im.tileCache->addBoxObstacle(lo, hi, &ref);
    if (dtStatusFailed(st)) { im.error = obstacleAddError(st); return 0; }
    im.liveObstacles.push_back(ref);
    im.obstaclesDirty.store(true, std::memory_order_relaxed);
    return ref;
}

NavMesh::ObstacleId NavMesh::addBoxObstacle(bromath::Vec3 center, bromath::Vec3 halfExtents,
                                            float yawRadians) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    if (!im.tileCache) {
        im.error = "addObstacle: mesh was not baked with dynamicObstacles";
        return 0;
    }
    const float c[3] = {center.x, center.y, center.z};
    const float he[3] = {halfExtents.x, halfExtents.y, halfExtents.z};
    dtObstacleRef ref = 0;
    dtStatus st = im.tileCache->addBoxObstacle(c, he, yawRadians, &ref);
    if (dtStatusFailed(st)) { im.error = obstacleAddError(st); return 0; }
    im.liveObstacles.push_back(ref);
    im.obstaclesDirty.store(true, std::memory_order_relaxed);
    return ref;
}

bool NavMesh::removeObstacle(ObstacleId id) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    if (!im.tileCache || id == 0) return false;
    auto it = std::find(im.liveObstacles.begin(), im.liveObstacles.end(), id);
    if (it == im.liveObstacles.end()) return false; // unknown / stale / doubled
    dtStatus st = im.tileCache->removeObstacle(id);
    if (dtStatusFailed(st)) {
        im.error = "removeObstacle: request queue full — pump update() and retry";
        return false;
    }
    im.liveObstacles.erase(it);
    im.obstaclesDirty.store(true, std::memory_order_relaxed);
    return true;
}

bool NavMesh::update(float dt) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    if (!im.tileCache || !im.mesh) return true;
    bool upToDate = false;
    dtStatus st = im.tileCache->update(dt, im.mesh, &upToDate);
    if (dtStatusFailed(st)) {
        im.error = "obstacle update failed";
        return false;
    }
    if (upToDate && im.obstaclesDirty.load(std::memory_order_relaxed)) {
        // A batch of obstacle changes has fully landed: the walkable surface
        // changed, so path followers must re-plan.
        im.obstaclesDirty.store(false, std::memory_order_relaxed);
        im.generation.fetch_add(1, std::memory_order_relaxed);
    }
    return upToDate;
}

bool NavMesh::obstaclesPending() const {
    return impl_->obstaclesDirty.load(std::memory_order_relaxed);
}

int NavMesh::obstacleCount() const {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.queryMutex);
    return static_cast<int>(im.liveObstacles.size());
}

uint32_t NavMesh::generation() const {
    return impl_->generation.load(std::memory_order_relaxed);
}

// ─── Serialization ───────────────────────────────────────────────────────────

bool NavMesh::saveTo(std::vector<uint8_t>& out) const {
    // Tiled (dynamic-obstacle) meshes have no single-blob representation —
    // their state lives in the tile cache. Documented as unsupported.
    if (!valid() || impl_->tileCache) return false;
    out = impl_->blob;
    return true;
}

bool NavMesh::loadFrom(const uint8_t* data, size_t size) {
    return impl_->initFromData(data, size);
}

} // namespace brogameagent
