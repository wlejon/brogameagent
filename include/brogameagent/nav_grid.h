#pragma once

#include "types.h"
#include <vector>
#include <cstdint>

namespace brogameagent {

/// Result of NavGrid::findPathEx(): smoothed waypoints plus whether the goal
/// had to be clamped.
struct NavGridPath {
    std::vector<bromath::Vec2> points;

    /// True when the goal was NOT reached and the path ends at the closest
    /// reachable cell instead (goal blocked, out of bounds, or walled off).
    /// With requireFullPath=true a partial result has EMPTY points but
    /// `partial` still reads true — so callers can tell "unreachable" from
    /// "start invalid" (empty + !partial).
    bool partial = false;
};

/// 2D grid-based navigation mesh for flat arenas with AABB obstacles.
/// Cells are marked walkable or blocked. Pathfinding uses A* on the grid
/// with 8-directional movement, then the path is smoothed via line-of-sight
/// checks to remove unnecessary waypoints.
class NavGrid {
public:
    /// Construct a navigation grid covering the given bounds.
    /// @param minX, minZ, maxX, maxZ  World-space bounds of the navigable area.
    /// @param cellSize  Size of each grid cell (smaller = more precise, more memory).
    NavGrid(float minX, float minZ, float maxX, float maxZ, float cellSize);

    /// Mark cells overlapping an AABB obstacle as blocked.
    /// @param padding  Extra clearance around the obstacle (agent radius).
    void addObstacle(const AABB& box, float padding = 0);

    /// The raw (unpadded) obstacle boxes added so far. Retained so embedders
    /// can bridge the same walls into other systems — e.g. baking them into
    /// World avoidance obstacles so ORCA respects what A* paths around.
    const std::vector<AABB>& obstacles() const { return obstacleBoxes_; }

    /// Check if a world position is on a walkable cell.
    bool isWalkable(float x, float z) const;

    /// Find a path from start to goal using A*.
    /// When the goal is blocked, out of bounds, or unreachable the path
    /// CLAMPS to the closest reachable cell (best-heuristic node) instead of
    /// failing — use findPathEx() to detect that, or to opt back into
    /// hard-fail semantics. Empty only when the start itself is invalid
    /// (out of bounds or on a blocked cell).
    /// The returned path is smoothed (redundant waypoints removed).
    std::vector<bromath::Vec2> findPath(bromath::Vec2 from, bromath::Vec2 to) const;

    /// findPath() with partial-path reporting. When the goal is not reached
    /// the result holds the path to the closest reachable cell with
    /// partial=true. Pass requireFullPath=true for hard-fail semantics: a
    /// partial result then has empty points (partial stays true, see
    /// NavGridPath). Deterministic: ties in the closest-cell fallback break
    /// by lower path cost, then lower cell index.
    NavGridPath findPathEx(bromath::Vec2 from, bromath::Vec2 to,
                           bool requireFullPath = false) const;

    /// Line-of-sight check on the grid (Bresenham). Returns true if clear.
    bool hasGridLOS(bromath::Vec2 from, bromath::Vec2 to) const;

    // Grid dimensions
    int width() const { return width_; }
    int height() const { return height_; }
    float cellSize() const { return cellSize_; }

    // World-space bounds. Used by Agent::integrate_ to clamp continuous-
    // control motion so policy-driven agents can't walk off the navigable
    // area (rollouts and the scripted path must both respect the same box).
    float minX() const { return minX_; }
    float minZ() const { return minZ_; }
    float maxX() const { return maxX_; }
    float maxZ() const { return maxZ_; }

private:
    int toGridX(float worldX) const;
    int toGridZ(float worldZ) const;
    float toWorldX(int gx) const;
    float toWorldZ(int gz) const;
    bool inBounds(int gx, int gz) const;

    std::vector<bromath::Vec2> smoothPath(const std::vector<bromath::Vec2>& raw) const;

    float minX_, minZ_, maxX_, maxZ_;
    float cellSize_;
    int width_, height_;
    std::vector<uint8_t> grid_; // 0 = walkable, 1 = blocked
    std::vector<AABB> obstacleBoxes_; // raw boxes, retained for obstacles()

    // Scratch buffers for A* live in thread_local statics inside findPath()
    // so concurrent pathfinding across threads doesn't race.
};

} // namespace brogameagent
