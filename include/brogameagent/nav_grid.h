#pragma once

#include "types.h"
#include <vector>
#include <cstdint>

namespace brogameagent {

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

    /// Check if a world position is on a walkable cell.
    bool isWalkable(float x, float z) const;

    /// Find a path from start to goal using A*.
    /// Returns an empty vector if no path exists.
    /// The returned path is smoothed (redundant waypoints removed).
    std::vector<Vec2> findPath(Vec2 from, Vec2 to) const;

    /// Line-of-sight check on the grid (Bresenham). Returns true if clear.
    bool hasGridLOS(Vec2 from, Vec2 to) const;

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

    std::vector<Vec2> smoothPath(const std::vector<Vec2>& raw) const;

    float minX_, minZ_, maxX_, maxZ_;
    float cellSize_;
    int width_, height_;
    std::vector<uint8_t> grid_; // 0 = walkable, 1 = blocked

    // Scratch buffers reused across findPath() calls to avoid per-call
    // allocation. NOT thread-safe: do not call findPath() concurrently
    // on the same NavGrid.
    mutable std::vector<float> gScoreScratch_;
    mutable std::vector<int>   cameFromScratch_;
};

} // namespace brogameagent
