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

bool NavGrid::hasGridLOS(Vec2 from, Vec2 to) const {
    int x0 = toGridX(from.x), z0 = toGridZ(from.z);
    int x1 = toGridX(to.x),   z1 = toGridZ(to.z);

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
std::vector<Vec2> NavGrid::findPath(Vec2 from, Vec2 to) const {
    int sx = toGridX(from.x), sz = toGridZ(from.z);
    int gx = toGridX(to.x),   gz = toGridZ(to.z);

    if (!inBounds(sx, sz) || !inBounds(gx, gz)) return {};
    if (grid_[sz * width_ + sx] != 0) return {};
    if (grid_[gz * width_ + gx] != 0) return {};
    if (sx == gx && sz == gz) return {to};

    const int N = width_ * height_;
    std::vector<float> gScore(N, 1e18f);
    std::vector<int> cameFrom(N, -1);

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

    // 8 directions: dx, dz, cost
    static constexpr int DX[] = {1, -1, 0, 0, 1, -1, 1, -1};
    static constexpr int DZ[] = {0, 0, 1, -1, 1, 1, -1, -1};
    static constexpr float COST[] = {1, 1, 1, 1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

    while (!open.empty()) {
        auto [f, cx, cz] = open.top();
        open.pop();

        if (cx == gx && cz == gz) break;

        int ci = idx(cx, cz);
        if (f > gScore[ci] + heuristic(cx, cz) + 0.01f) continue; // stale entry

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

    // Reconstruct
    int gi = idx(gx, gz);
    if (cameFrom[gi] == -1 && !(sx == gx && sz == gz)) return {};

    std::vector<Vec2> raw;
    for (int i = gi; i != -1; i = cameFrom[i]) {
        int pz = i / width_, px = i % width_;
        raw.push_back({toWorldX(px), toWorldZ(pz)});
    }
    std::reverse(raw.begin(), raw.end());

    // Replace last point with exact target position
    if (!raw.empty()) raw.back() = to;

    return smoothPath(raw);
}

std::vector<Vec2> NavGrid::smoothPath(const std::vector<Vec2>& raw) const {
    if (raw.size() <= 2) return raw;

    std::vector<Vec2> smooth;
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
