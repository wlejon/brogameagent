// NavGrid::flowField — the integration / flow field for a crowd heading to
// one goal: one search for N units, where A* would be N searches.
//
// A fast-marching (eikonal) solve rather than 8-way Dijkstra: each cell's
// cost-to-goal comes from its best horizontal and vertical neighbours
// together, which approximates true straight-line distance. Octile Dijkstra
// distances have ridges along the rows and diagonals through the goal, and a
// swarm descending them funnels into a few thin lanes; the eikonal field
// descends in straight rays. 4-connected, so it never cuts a wall corner.

#include "brogameagent/nav_grid.h"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

namespace brogameagent {

NavGridField NavGrid::flowField(bromath::Vec2 goal, const float* extraCost,
                                const float* costs) const {
    NavGridField out;
    const int W = width_, H = height_, N = W * H;
    out.width = W;
    out.height = H;
    const float INF = std::numeric_limits<float>::infinity();
    out.dist.assign(N, INF);
    out.dirX.assign(N, 0.0f);
    out.dirZ.assign(N, 0.0f);
    if (N <= 0) return out;

    // Speed cost per cell in world units per cell step; <0 = blocked.
    std::vector<float> f(N);
    for (int i = 0; i < N; ++i) {
        float c;
        if (costs) {
            c = costs[i];
            if (!(c > 0.0f && c < 1e6f) || grid_[i] != 0) { f[i] = -1.0f; continue; }
        } else {
            if (grid_[i] != 0) { f[i] = -1.0f; continue; }
            c = cost_[i];
        }
        if (extraCost) {
            const float e = extraCost[i];
            if (std::isfinite(e)) c += e;
            if (!(c > 0.0f)) c = 1e-3f;   // never a free or negative cell
        }
        f[i] = c * cellSize_;
    }

    const int gx = toGridX(goal.x), gz = toGridZ(goal.y);
    if (!inBounds(gx, gz) || goal.x < minX_ || goal.y < minZ_) return out;
    const int g = gz * W + gx;
    if (f[g] < 0.0f) return out;

    // Finite stand-in for "unknown" so the eikonal update never computes
    // inf - inf; converted to +infinity on the way out.
    const float BIG = 1e30f;
    std::vector<float> d(N, BIG);
    std::vector<uint8_t> frozen(N, 0);
    using Entry = std::pair<float, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    d[g] = 0.0f;
    heap.push({0.0f, g});

    auto relax = [&](int n) {
        const int x = n % W, z = n / W;
        const float fc = f[n];
        const float a = std::min(x > 0 ? d[n - 1] : BIG, x < W - 1 ? d[n + 1] : BIG);
        const float b = std::min(z > 0 ? d[n - W] : BIG, z < H - 1 ? d[n + W] : BIG);
        const float lo = a < b ? a : b, gap = a - b;
        const float nd = (gap > -fc && gap < fc)
            ? (a + b + std::sqrt(2.0f * fc * fc - gap * gap)) * 0.5f
            : lo + fc;
        if (nd < d[n]) {
            d[n] = nd;
            heap.push({nd, n});
        }
    };

    int reached = 0;
    while (!heap.empty()) {
        const auto [k, c] = heap.top();
        heap.pop();
        if (frozen[c] || k > d[c]) continue;   // stale entry
        frozen[c] = 1;                          // accepted: never updated again
        ++reached;
        const int x = c % W, z = c / W;
        if (x < W - 1 && f[c + 1] >= 0.0f && !frozen[c + 1]) relax(c + 1);
        if (x > 0     && f[c - 1] >= 0.0f && !frozen[c - 1]) relax(c - 1);
        if (z < H - 1 && f[c + W] >= 0.0f && !frozen[c + W]) relax(c + W);
        if (z > 0     && f[c - W] >= 0.0f && !frozen[c - W]) relax(c - W);
    }
    out.reached = reached;

    // Direction: minus the upwind gradient, per axis (blocked and unreached
    // cells hold BIG and are never upwind).
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            const int c = z * W + x;
            const float dc = d[c];
            if (dc >= BIG) continue;
            out.dist[c] = dc;
            const float l = x > 0 ? d[c - 1] : BIG, r = x < W - 1 ? d[c + 1] : BIG;
            const float u = z > 0 ? d[c - W] : BIG, dn = z < H - 1 ? d[c + W] : BIG;
            const float fx = l < r ? (l < dc ? l - dc : 0.0f) : (r < dc ? dc - r : 0.0f);
            const float fz = u < dn ? (u < dc ? u - dc : 0.0f) : (dn < dc ? dc - dn : 0.0f);
            const float fl = std::sqrt(fx * fx + fz * fz);
            if (fl > 1e-6f) { out.dirX[c] = fx / fl; out.dirZ[c] = fz / fl; }
        }
    }
    return out;
}

} // namespace brogameagent
