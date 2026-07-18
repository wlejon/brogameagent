#pragma once

#include "types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brogameagent {

/// A point-to-point traversal shortcut baked into a NavMesh — jump gaps,
/// drop ledges, ladders, teleporters (the Godot NavigationLink analog).
/// Detour traverses links automatically during path queries once baked; the
/// resulting path marks the takeoff point (NavMeshPath::kLinkStart) so
/// followers can play a jump/climb animation while moving straight to the
/// next point. Each endpoint must lie within `radius` of the (eroded)
/// walkable surface to connect — endpoints that miss are dropped silently
/// by Detour, exactly like a Godot link placed off the mesh.
struct NavMeshOffMeshLink {
    bromath::Vec3 start;
    bromath::Vec3 end;
    float radius = 0.5f;       // endpoint pickup radius (world units)
    bool bidirectional = true; // false: traversable start → end only
    /// Area id stored on the link polygon (dtOffMeshConnection). All
    /// baked polys currently share one walkable area — per-area costs are
    /// future work — so this is a forward-compat tag, not a cost knob.
    uint8_t areaId = 63;       // RC_WALKABLE_AREA
    /// Application tag carried through Detour (dtOffMeshConnection::userId).
    uint32_t userId = 0;
};

/// Bake parameters for NavMesh::bake(). Defaults are tuned for a ~0.5 m
/// radius humanoid agent in meter-scale worlds; the comments give the Recast
/// semantics so embedders can retune without reading Recast docs.
struct NavMeshBakeConfig {
    float cellSize = 0.25f;          // XZ voxel size (world units). ~agentRadius/2.
    float cellHeight = 0.2f;         // Y voxel size (world units).
    float agentRadius = 0.5f;        // walkable area is eroded by this much.
    float agentHeight = 2.0f;        // min clearance for a walkable span.
    float agentMaxClimb = 0.4f;      // max step/ledge height treated as climbable.
    float agentMaxSlopeDeg = 45.0f;  // triangles steeper than this are unwalkable.
    float regionMinSize = 8.0f;      // min region size (in cells, squared internally).
    float regionMergeSize = 20.0f;   // regions smaller than this merge into neighbors.
    float edgeMaxLen = 12.0f;        // max contour edge length (world units).
    float edgeMaxError = 1.3f;       // max contour simplification deviation (cells).
    float detailSampleDist = 6.0f;   // detail-mesh sample spacing (in cells; <0.9 = off).
    float detailSampleMaxError = 1.0f; // detail-mesh max deviation (in cellHeight units).

    // --- Dynamic obstacles (tiled tile-cache bake) ---------------------------
    // With dynamicObstacles=true the soup is baked TILED through Detour's
    // dtTileCache (compressed per-tile layers) instead of as one static tile.
    // That enables the runtime obstacle API (addObstacle/removeObstacle/
    // update) — tiles touched by an obstacle are rebuilt incrementally, no
    // full rebake. Trade-offs of the tiled path: no detail mesh (surface Y is
    // quantized to cellHeight, slightly coarser on slopes), no saveTo()
    // serialization, and no off-mesh links (bake() fails rather than dropping
    // them). Queries are identical.
    bool  dynamicObstacles = false;
    float tileSize = 16.0f;          // tile edge length (world units); clamped
                                     // to 16..255 cells per tile.
    int   maxObstacles = 128;        // obstacle slot budget for this mesh.

    // --- Off-mesh links ------------------------------------------------------
    // Baked into the Detour data (static bakes only — see NavMeshOffMeshLink).
    // Links survive saveTo()/loadFrom() since they live in the tile blob.
    // NOT supported together with dynamicObstacles: dtTileCache rebuilds
    // tiles at runtime and would drop bake-time connections, so bake() fails
    // with a clear error rather than silently losing links.
    std::vector<NavMeshOffMeshLink> offMeshLinks;
};

/// Result of a NavMesh::raycast() — the navmesh "can I walk straight there"
/// test (2D walkability ray along the mesh surface, not a physics ray).
struct NavMeshRaycastHit {
    bool hit = false;         // true if a boundary blocked the ray before `end`
    float t = 1.0f;           // hit param along [start,end]; 1 when unobstructed
    bromath::Vec3 point;      // hit position (== end when !hit)
    bromath::Vec3 normal;     // XZ wall normal at the hit (zero when !hit)
};

/// Result of NavMesh::findPathEx(): the straightened (funnel/string-pulled)
/// waypoint list, including the snapped start and the (possibly clamped) end.
struct NavMeshPath {
    /// flags bit: points[i] is the TAKEOFF of an off-mesh link — the segment
    /// from points[i] to points[i+1] traverses the link (jump/drop/teleport),
    /// not the walkable surface. Followers move straight along it; apps can
    /// watch the marker to play a jump/climb animation.
    static constexpr uint8_t kLinkStart = 0x01;

    std::vector<bromath::Vec3> points;
    std::vector<uint8_t> flags;   // per-point flag bits; same size as points

    /// True when the goal was NOT reached and the path ends at the closest
    /// reachable point instead (goal on a disconnected island, or the
    /// corridor overflowed the internal poly budget). With
    /// requireFullPath=true a partial result has EMPTY points but `partial`
    /// still reads true — so callers can tell "unreachable" (empty + partial)
    /// from "endpoint failed to snap" (empty + !partial).
    bool partial = false;

    /// Convenience: is points[i] an off-mesh link takeoff?
    bool isLinkStart(size_t i) const {
        return i < flags.size() && (flags[i] & kLinkStart) != 0;
    }
};

/// Polygon navigation mesh baked from arbitrary triangle soup — the 3D
/// counterpart to NavGrid for worlds a flat 2D grid cannot represent:
/// slopes, bridges/overpasses, multi-level interiors. Backed by
/// Recast (voxelization + region/contour/polygon build) and Detour
/// (runtime queries); neither leaks into this header.
///
/// Coordinates are y-up world space (bromath::Vec3), matching the rest of
/// the library. Input triangles must be wound counter-clockwise when viewed
/// from above (+Y normals) to be considered walkable.
///
/// Two bake modes:
///   - Static (default): the whole soup becomes one Detour tile — right for
///     baked levels up to a few hundred meters across; supports the detail
///     mesh and saveTo()/loadFrom() serialization.
///   - Dynamic-obstacle (NavMeshBakeConfig::dynamicObstacles): tiled bake via
///     dtTileCache with compressed per-tile layers. Enables the runtime
///     obstacle API — addObstacle*/removeObstacle queue changes, update()
///     rebuilds only the touched tiles (typically 1 tile per update call), and
///     generation() bumps once all pending changes have been applied so path
///     followers know to re-plan. See the "Dynamic obstacles" section below.
///
/// Thread safety: bake()/loadFrom() must not run concurrently with anything
/// else. Queries (findPath / nearestPoint / raycast / randomPoint) and the
/// obstacle calls (addObstacle*/removeObstacle/update) are serialized by an
/// internal mutex — safe to call from multiple threads, but they do not run
/// in parallel (Detour query objects are stateful).
/// For parallel pathfinding, give each worker its own NavMesh via
/// saveTo()/loadFrom() — loading is a cheap memcpy, the bake is the
/// expensive part.
///
/// World integration: World is untouched — route agents by projecting the
/// Vec3 waypoints to XZ (Vec2) and feeding them to the existing followPath
/// steering / Agent path plumbing, exactly like NavGrid paths.
class NavMesh {
public:
    NavMesh();
    ~NavMesh();
    NavMesh(NavMesh&&) noexcept;
    NavMesh& operator=(NavMesh&&) noexcept;
    NavMesh(const NavMesh&) = delete;
    NavMesh& operator=(const NavMesh&) = delete;

    // --- Bake ----------------------------------------------------------------

    /// Bake a navmesh from a triangle soup.
    /// @param vertices     xyz triples, y-up world space.
    /// @param vertexCount  number of vertices (floats / 3).
    /// @param indices      triangle indices into `vertices`, CCW from above.
    /// @param indexCount   number of indices (multiple of 3).
    /// Returns false on failure; lastError() then describes what went wrong
    /// (including Recast build log messages). A successful bake replaces any
    /// previously baked/loaded mesh.
    bool bake(const float* vertices, size_t vertexCount,
              const uint32_t* indices, size_t indexCount,
              const NavMeshBakeConfig& config = {});

    /// True once a bake() or loadFrom() has succeeded.
    bool valid() const;

    /// Human-readable diagnosis of the last failed bake/load/query-setup.
    const std::string& lastError() const;

    // --- Queries ---------------------------------------------------------------
    // All queries snap their input points to the mesh within `searchExtents`
    // (half-extents of the snap box, world units). The y extent is what
    // disambiguates stacked levels: keep it smaller than the level spacing
    // to snap to the level you mean.

    /// Default snap half-extents: 2 m horizontally, 1 m vertically. The tight
    /// vertical extent is deliberate — it makes stacked-level queries resolve
    /// to the level nearest the query point.
    static constexpr bromath::Vec3 kDefaultExtents{2.0f, 1.0f, 2.0f};

    /// Find a walkable path from start to end. Returns the straightened
    /// (funnel/string-pulled) waypoint list, including the snapped start and
    /// end points. When the goal is unreachable (disconnected island) the
    /// path CLAMPS to the closest reachable point instead of failing — use
    /// findPathEx() to detect that, or to opt back into hard-fail semantics.
    /// Empty only when either endpoint fails to snap within searchExtents.
    /// Deterministic: same mesh + inputs always yield the same waypoints.
    std::vector<bromath::Vec3> findPath(bromath::Vec3 start, bromath::Vec3 end,
                                        bromath::Vec3 searchExtents = kDefaultExtents) const;

    /// findPath() with partial-path reporting. When the goal is unreachable
    /// the result holds the path to the closest reachable point with
    /// partial=true. Pass requireFullPath=true for hard-fail semantics: a
    /// partial result then has empty points (partial stays true, see
    /// NavMeshPath). Deterministic like findPath().
    NavMeshPath findPathEx(bromath::Vec3 start, bromath::Vec3 end,
                           bromath::Vec3 searchExtents = kDefaultExtents,
                           bool requireFullPath = false) const;

    /// Snap an arbitrary point onto the navmesh. Returns false if nothing is
    /// within searchExtents.
    bool nearestPoint(bromath::Vec3 p, bromath::Vec3& out,
                      bromath::Vec3 searchExtents = kDefaultExtents) const;

    /// Walkability raycast from start toward end along the mesh: does a
    /// straight walk get there, and if not, where does it stop? Stops at
    /// obstruction boundaries and at the edge of the mesh. `start` must
    /// snap onto the mesh (hit=false, t=0 result otherwise).
    NavMeshRaycastHit raycast(bromath::Vec3 start, bromath::Vec3 end,
                              bromath::Vec3 searchExtents = kDefaultExtents) const;

    /// Uniform-ish random reachable point on the mesh (area-weighted poly
    /// pick). Deterministic per seed. Returns false when the mesh is empty.
    bool randomPoint(uint32_t seed, bromath::Vec3& out) const;

    // --- Dynamic obstacles ---------------------------------------------------
    // Available only on meshes baked with NavMeshBakeConfig::dynamicObstacles.
    // add/remove queue a change; nothing moves until update() is pumped —
    // each update() call rebuilds at most one touched tile, so changes take
    // effect over the next few pumps (update() returns true once everything
    // has been applied). Obstacles carve the walkable surface exactly like
    // baked-in geometry: findPath detours around them or fails when they
    // sever the corridor, and removing them restores the original surface.

    /// Handle to a queued/active obstacle. 0 is never a valid handle.
    using ObstacleId = uint32_t;

    /// True when this mesh was baked with dynamicObstacles and can take
    /// runtime obstacles.
    bool supportsObstacles() const;

    /// Add a cylinder obstacle (pos = center of the cylinder's BASE, i.e. on
    /// the walkable surface). Returns 0 on failure (unsupported mesh, request
    /// queue full — pump update() — or out of obstacle slots; lastError()
    /// explains).
    ObstacleId addObstacle(bromath::Vec3 pos, float radius, float height);

    /// Add an axis-aligned box obstacle. Returns 0 on failure (see above).
    ObstacleId addBoxObstacle(bromath::Vec3 bmin, bromath::Vec3 bmax);

    /// Add a Y-rotated box obstacle (center + half extents + yaw radians).
    /// Returns 0 on failure (see above).
    ObstacleId addBoxObstacle(bromath::Vec3 center, bromath::Vec3 halfExtents,
                              float yawRadians);

    /// Queue removal of an obstacle. Returns false for an unknown/stale
    /// handle or an unsupported mesh. The surface restores after update().
    bool removeObstacle(ObstacleId id);

    /// Pump pending obstacle changes: rebuilds at most one touched tile per
    /// call. Returns true when the mesh is fully up to date (nothing pending);
    /// keep calling once per frame — or loop until true for a synchronous
    /// apply. On meshes without obstacle support this is a no-op returning
    /// true. `dt` is forwarded to Detour (currently unused by it).
    bool update(float dt = 1.0f / 60.0f);

    /// True while queued obstacle changes have not yet been fully applied.
    bool obstaclesPending() const;

    /// Number of active obstacles (added and not removed; counts queued ones).
    int obstacleCount() const;

    /// Monotonic counter that bumps every time the walkable surface changes:
    /// after a successful bake()/loadFrom(), and after update() finishes
    /// applying a batch of obstacle changes. Path followers snapshot this at
    /// plan time and re-plan when it moves.
    uint32_t generation() const;

    // --- Serialization -----------------------------------------------------------
    // Baking is expensive; the baked mesh is cheap to snapshot. The blob is
    // the raw Detour tile data (self-validating magic/version header) — cache
    // it to disk and loadFrom() at startup.

    /// Serialize the baked mesh. Returns false when !valid(), and for
    /// dynamic-obstacle (tiled) meshes, which do not support serialization.
    bool saveTo(std::vector<uint8_t>& out) const;

    /// Load a mesh previously produced by saveTo(). Replaces any current
    /// mesh. Returns false (with lastError() set) on malformed data.
    bool loadFrom(const uint8_t* data, size_t size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace brogameagent
