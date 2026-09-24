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

/// Result of NavGrid::flowField(): per cell (row-major, index = z * width + x)
/// the cost-to-goal and the unit direction to walk.
struct NavGridField {
    int width = 0, height = 0;
    /// Cost to reach the goal; +infinity on blocked and unreached cells.
    std::vector<float> dist;
    /// Unit steering direction (world x / z); 0,0 where there is none (the
    /// goal cell, blocked and unreached cells).
    std::vector<float> dirX, dirZ;
    /// Cells the wave reached (the goal's connected region).
    int reached = 0;
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

    /// Set walkable state of a world position.
    void setWalkable(float x, float z, bool walkable);

    /// Set cell traversal cost: the price of entering the cell per unit of
    /// distance (1 = open ground, the default). cost <= 0, >= 1e6 or NaN
    /// marks the cell unwalkable; any other value also makes it walkable.
    /// findPath() and flowField() both charge it.
    void setCellCost(float x, float z, float cost);

    /// The traversal cost of the cell at a world position (1 by default;
    /// +infinity when blocked or out of bounds).
    float cellCost(float x, float z) const;

    /// One search for many units: the integration field (cost-to-goal) from
    /// `goal` over every reachable cell, and a steering direction per cell.
    /// A fast-marching (eikonal) solve over 4-neighbours, so costs approximate
    /// true straight-line distance (no octile ridges funnelling a crowd into
    /// lanes) and a wall corner is never cut. Each cell's speed cost is its
    /// cellCost, plus `extraCost[i]` when given (width*height entries, e.g. a
    /// danger map), or `costs[i]` in place of the stored costs when given
    /// (then <= 0, >= 1e6 or NaN is blocked too). An out-of-bounds or blocked
    /// goal gives an empty field (reached 0).
    NavGridField flowField(bromath::Vec2 goal, const float* extraCost = nullptr,
                           const float* costs = nullptr) const;

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
    std::vector<float> cost_;   // per-cell traversal cost, 1 = open
    std::vector<AABB> obstacleBoxes_; // raw boxes, retained for obstacles()

    // Scratch buffers for A* live in thread_local statics inside findPath()
    // so concurrent pathfinding across threads doesn't race.
};

} // namespace brogameagent
