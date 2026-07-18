#include "brogameagent/nav_grid.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace brogameagent {

NavGrid::NavGrid(float minX, float minZ, float maxX, float maxZ, float cellSize)
    : minX_(minX), minZ_(minZ), maxX_(maxX), maxZ_(maxZ), cellSize_(cellSize)
{
    width_  = static_cast<int>(std::ceil((maxX - minX) / cellSize));
    height_ = static_cast<int>(std::ceil((maxZ - minZ) / cellSize));
    grid_.assign(width_ * height_, 0); // all walkable
}

void NavGrid::addObstacle(const AABB& box, float padding) {
    obstacleBoxes_.push_back(box);

    float x0 = box.cx - box.hw - padding;
    float x1 = box.cx + box.hw + padding;
    float z0 = box.cz - box.hd - padding;
    float z1 = box.cz + box.hd + padding;

    int gx0 = std::max(0, toGridX(x0));
    int gx1 = std::min(width_ - 1, toGridX(x1));
    int gz0 = std::max(0, toGridZ(z0));
    int gz1 = std::min(height_ - 1, toGridZ(z1));

    for (int gz = gz0; gz <= gz1; gz++) {
        for (int gx = gx0; gx <= gx1; gx++) {
            grid_[gz * width_ + gx] = 1;
        }
    }
}

bool NavGrid::isWalkable(float x, float z) const {
    int gx = toGridX(x);
    int gz = toGridZ(z);
    if (!inBounds(gx, gz)) return false;
    return grid_[gz * width_ + gx] == 0;
}

bool NavGrid::hasGridLOS(bromath::Vec2 from, bromath::Vec2 to) const {
    int x0 = toGridX(from.x), z0 = toGridZ(from.y);
    int x1 = toGridX(to.x),   z1 = toGridZ(to.y);

    int dx = std::abs(x1 - x0), dz = std::abs(z1 - z0);
    int sx = (x0 < x1) ? 1 : -1;
    int sz = (z0 < z1) ? 1 : -1;
    int err = dx - dz;

    while (true) {
        if (!inBounds(x0, z0) || grid_[z0 * width_ + x0] != 0) return false;
        if (x0 == x1 && z0 == z1) break;
        int e2 = 2 * err;
        if (e2 > -dz) { err -= dz; x0 += sx; }
        if (e2 <  dx) { err += dx; z0 += sz; }
    }
    return true;
}

// A* with 8-directional movement
std::vector<bromath::Vec2> NavGrid::findPath(bromath::Vec2 from, bromath::Vec2 to) const {
    return findPathEx(from, to).points;
}

NavGridPath NavGrid::findPathEx(bromath::Vec2 from, bromath::Vec2 to,
                                bool requireFullPath) const {
    NavGridPath result;

    int sx = toGridX(from.x), sz = toGridZ(from.y);
    if (!inBounds(sx, sz) || grid_[sz * width_ + sx] != 0)
        return result;  // invalid start: empty, !partial

    // Clamp an out-of-bounds goal onto the grid; the search then aims for
    // the nearest representable cell and the result is marked partial.
    const int rawGx = toGridX(to.x), rawGz = toGridZ(to.y);
    const int gx = std::clamp(rawGx, 0, width_ - 1);
    const int gz = std::clamp(rawGz, 0, height_ - 1);
    const bool goalClamped = gx != rawGx || gz != rawGz;
    const bool goalBlocked = grid_[gz * width_ + gx] != 0;

    if (sx == gx && sz == gz && !goalClamped && !goalBlocked) {
        result.points = {to};
        return result;
    }

    const int N = width_ * height_;
    // thread_local so concurrent findPath() across threads is safe; each
    // thread reuses (and grows) its own scratch buffers.
    thread_local std::vector<float> gScoreScratch;
    thread_local std::vector<int>   cameFromScratch;
    gScoreScratch.assign(N, 1e18f);
    cameFromScratch.assign(N, -1);
    auto& gScore   = gScoreScratch;
    auto& cameFrom = cameFromScratch;

    auto idx = [&](int x, int z) { return z * width_ + x; };
    auto heuristic = [&](int x, int z) {
        float dx = static_cast<float>(std::abs(x - gx));
        float dz = static_cast<float>(std::abs(z - gz));
        // Octile distance
        float mn = std::min(dx, dz);
        return (dx + dz) + (1.41421356f - 2.0f) * mn;
    };

    struct Node {
        float f;
        int x, z;
        bool operator>(const Node& o) const { return f > o.f; }
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    gScore[idx(sx, sz)] = 0;
    open.push({heuristic(sx, sz), sx, sz});

    // Closest-reachable fallback: the expanded node nearest the goal (by
    // heuristic; ties break by lower g then lower cell index — fully
    // deterministic). Used when the goal is blocked/clamped/walled off.
    int bestIdx = idx(sx, sz);
    float bestH = heuristic(sx, sz);
    float bestG = 0.0f;
    bool goalReached = false;

    // 8 directions: dx, dz, cost
    static constexpr int DX[] = {1, -1, 0, 0, 1, -1, 1, -1};
    static constexpr int DZ[] = {0, 0, 1, -1, 1, 1, -1, -1};
    static constexpr float COST[] = {1, 1, 1, 1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

    while (!open.empty()) {
        auto [f, cx, cz] = open.top();
        open.pop();

        if (cx == gx && cz == gz) { goalReached = true; break; }

        int ci = idx(cx, cz);
        const float h = heuristic(cx, cz);
        if (f > gScore[ci] + h + 0.01f) continue; // stale entry

        if (h < bestH ||
            (h == bestH && (gScore[ci] < bestG ||
                            (gScore[ci] == bestG && ci < bestIdx)))) {
            bestIdx = ci;
            bestH = h;
            bestG = gScore[ci];
        }

        for (int d = 0; d < 8; d++) {
            int nx = cx + DX[d], nz = cz + DZ[d];
            if (!inBounds(nx, nz)) continue;
            int ni = idx(nx, nz);
            if (grid_[ni] != 0) continue;

            // For diagonal moves, check that both cardinal neighbors are clear
            if (d >= 4) {
                if (grid_[idx(cx + DX[d], cz)] != 0) continue;
                if (grid_[idx(cx, cz + DZ[d])] != 0) continue;
            }

            float ng = gScore[ci] + COST[d];
            if (ng < gScore[ni]) {
                gScore[ni] = ng;
                cameFrom[ni] = ci;
                open.push({ng + heuristic(nx, nz), nx, nz});
            }
        }
    }

    // Reconstruct — to the goal when reached, else to the closest reachable
    // cell (partial path, Godot-style clamping).
    result.partial = !goalReached || goalClamped;
    if (result.partial && requireFullPath) return result;  // empty, partial=true
    const int gi = goalReached ? idx(gx, gz) : bestIdx;

    std::vector<bromath::Vec2> raw;
    for (int i = gi; i != -1; i = cameFrom[i]) {
        int pz = i / width_, px = i % width_;
        raw.push_back({toWorldX(px), toWorldZ(pz)});
    }
    std::reverse(raw.begin(), raw.end());

    // Replace last point with the exact target position — only when it was
    // actually reached; a clamped path ends at the closest cell's center.
    if (!raw.empty() && goalReached && !goalClamped) raw.back() = to;

    result.points = smoothPath(raw);
    return result;
}

std::vector<bromath::Vec2> NavGrid::smoothPath(const std::vector<bromath::Vec2>& raw) const {
    if (raw.size() <= 2) return raw;

    std::vector<bromath::Vec2> smooth;
    smooth.push_back(raw.front());

    size_t current = 0;
    while (current < raw.size() - 1) {
        // Look as far ahead as possible with clear LOS
        size_t farthest = current + 1;
        for (size_t i = current + 2; i < raw.size(); i++) {
            if (hasGridLOS(raw[current], raw[i])) {
                farthest = i;
            }
        }
        smooth.push_back(raw[farthest]);
        current = farthest;
    }

    return smooth;
}

int NavGrid::toGridX(float worldX) const {
    return static_cast<int>((worldX - minX_) / cellSize_);
}

int NavGrid::toGridZ(float worldZ) const {
    return static_cast<int>((worldZ - minZ_) / cellSize_);
}

float NavGrid::toWorldX(int gx) const {
    return minX_ + (static_cast<float>(gx) + 0.5f) * cellSize_;
}

float NavGrid::toWorldZ(int gz) const {
    return minZ_ + (static_cast<float>(gz) + 0.5f) * cellSize_;
}

bool NavGrid::inBounds(int gx, int gz) const {
    return gx >= 0 && gx < width_ && gz >= 0 && gz < height_;
}

} // namespace brogameagent
