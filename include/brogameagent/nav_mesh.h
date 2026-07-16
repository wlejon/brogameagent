#pragma once

#include "types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brogameagent {

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
};

/// Result of a NavMesh::raycast() — the navmesh "can I walk straight there"
/// test (2D walkability ray along the mesh surface, not a physics ray).
struct NavMeshRaycastHit {
    bool hit = false;         // true if a boundary blocked the ray before `end`
    float t = 1.0f;           // hit param along [start,end]; 1 when unobstructed
    bromath::Vec3 point;      // hit position (== end when !hit)
    bromath::Vec3 normal;     // XZ wall normal at the hit (zero when !hit)
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
/// Current scope: single-tile bake — the whole soup becomes one Detour tile,
/// which is right for baked levels up to a few hundred meters across.
/// Tiled/streaming bakes (huge or dynamic worlds) are future work.
///
/// Thread safety: bake()/loadFrom() must not run concurrently with anything
/// else. Queries (findPath / nearestPoint / raycast / randomPoint) are
/// serialized by an internal mutex — safe to call from multiple threads,
/// but they do not run in parallel (Detour query objects are stateful).
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
    /// end points. Empty when either endpoint fails to snap or when no
    /// COMPLETE path exists — partial paths toward unreachable goals are
    /// reported as failure, not silently truncated.
    /// Deterministic: same mesh + inputs always yield the same waypoints.
    std::vector<bromath::Vec3> findPath(bromath::Vec3 start, bromath::Vec3 end,
                                        bromath::Vec3 searchExtents = kDefaultExtents) const;

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

    // --- Serialization -----------------------------------------------------------
    // Baking is expensive; the baked mesh is cheap to snapshot. The blob is
    // the raw Detour tile data (self-validating magic/version header) — cache
    // it to disk and loadFrom() at startup.

    /// Serialize the baked mesh. Returns false when !valid().
    bool saveTo(std::vector<uint8_t>& out) const;

    /// Load a mesh previously produced by saveTo(). Replaces any current
    /// mesh. Returns false (with lastError() set) on malformed data.
    bool loadFrom(const uint8_t* data, size_t size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace brogameagent
