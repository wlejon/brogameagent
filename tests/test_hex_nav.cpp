// HexNav tests — the weighted hex-grid navigator against a naive reference.
// Core-only: builds with BROGAMEAGENT_WITH_NN=OFF.
//
// The reference A* keeps its open list as a plain array and takes the FIRST
// minimum of (f, h) on every pop — the earliest-inserted among equals. That
// is the tie rule an embedder's linear-scan search has, and the one HexNav's
// sequence-numbered heap must reproduce: the test demands the same path cell
// for cell, not merely the same cost.

#include <brogameagent/hex_nav.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

using namespace brogameagent;

struct TestEntry {
    const char* name;
    void (*fn)();
};

static std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { registry().push_back({#name, test_##name}); } } reg_##name; \
    static void test_##name()

static void check(bool cond, const char* msg, int line) {
    if (!cond) {
        printf("    assertion failed at line %d: %s\n", line, msg);
        throw 0;
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)

static const double INF = std::numeric_limits<double>::infinity();

// ─── A seeded LCG, so every grid is reproducible ─────────────────────────────

struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    double unit() { return (next() & 0xffffff) / double(0x1000000); }
};

// ─── Grid authoring ──────────────────────────────────────────────────────────

struct Grid {
    int size;
    std::vector<double> table;   // size*size*6, entry cost per cell per direction
    std::vector<uint8_t> clr;    // clearance, 1/2/3
    int idx(int x, int y) const { return y * size + x; }
};

static bool inB(int size, int x, int y) { return x >= 0 && y >= 0 && x < size && y < size; }

// A grid where every cell has one terrain cost (1, 1.5, 2 or impassable) and
// entering it from any side costs that; optionally some random walls
// (single directed edges made impassable) and an elevation climb (+1).
static Grid makeGrid(int size, uint32_t seed, double wallShare, bool climbs) {
    Grid g;
    g.size = size;
    Lcg r(seed);
    std::vector<double> terrain(size * size);
    std::vector<int> elev(size * size);
    for (int i = 0; i < size * size; i++) {
        const double u = r.unit();
        terrain[i] = u < 0.08 ? INF : u < 0.4 ? 1.0 : u < 0.7 ? 1.5 : 2.0;
        elev[i] = climbs ? int(r.unit() * 3) : 0;
    }
    g.table.assign(size * size * 6, INF);
    for (int cy = 0; cy < size; cy++) for (int cx = 0; cx < size; cx++) {
        const int c = g.idx(cx, cy);
        for (int d = 0; d < 6; d++) {
            const int nx = cx + HexNav::STEP[cy & 1][d][0], ny = cy + HexNav::STEP[cy & 1][d][1];
            if (!inB(size, nx, ny)) continue;
            // n → c enters c from direction d.
            double cost = terrain[c];
            if (cost == INF) continue;
            const int dz = elev[c] - elev[g.idx(nx, ny)];
            if (dz >= 2) continue;
            if (dz == 1) cost += 1;
            if (r.unit() < wallShare) continue;
            g.table[c * 6 + d] = cost;
        }
    }
    g.clr.assign(size * size, 1);
    for (int i = 0; i < size * size; i++) {
        const double u = r.unit();
        g.clr[i] = u < 0.15 ? 3 : u < 0.3 ? 2 : 1;
    }
    return g;
}

// ─── The reference search ────────────────────────────────────────────────────

static double hexH(int x, int y, int gx, int gy) {
    const int q = x - ((y - (y & 1)) >> 1), gq = gx - ((gy - (gy & 1)) >> 1);
    const int dq = gq - q, dr = gy - y;
    return (std::abs(dq) + std::abs(dq + dr) + std::abs(dr)) / 2.0;
}

struct RefResult {
    bool reached = false;
    std::vector<int32_t> path;
    std::vector<float> cost;
    std::vector<int32_t> parent;
};

// Linear-scan A* (goal ≥ 0) / Dijkstra (goal = −1) with a Float32 cost field,
// exactly the reference an embedder's JS search is.
static RefResult refSearch(const Grid& g, const uint8_t* clr, int x0, int y0, int goal, double maxCost) {
    const int size = g.size;
    RefResult r;
    r.cost.assign(size * size, std::numeric_limits<float>::infinity());
    r.parent.assign(size * size, -1);
    struct Open { double k, h; int v; };
    std::vector<Open> open;
    const bool astar = goal >= 0;
    const int gx = astar ? goal % size : 0, gy = astar ? goal / size : 0;
    const int start = g.idx(x0, y0);
    r.cost[start] = 0;
    const double h0 = astar ? hexH(x0, y0, gx, gy) : 0;
    open.push_back({h0, h0, start});
    while (!open.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < open.size(); i++) {
            if (open[i].k < open[best].k || (open[i].k == open[best].k && open[i].h < open[best].h)) best = i;
        }
        const Open o = open[best];
        open.erase(open.begin() + (long)best);
        const int i = o.v;
        const double gcost = o.k - o.h;
        if (gcost > (double)r.cost[i]) continue;
        if (i == goal) { r.reached = true; break; }
        const int cx = i % size, cy = i / size;
        for (int d = 0; d < 6; d++) {
            const int nx = cx + HexNav::STEP[cy & 1][d][0], ny = cy + HexNav::STEP[cy & 1][d][1];
            if (!inB(size, nx, ny)) continue;
            const int ni = g.idx(nx, ny);
            double sc = g.table[ni * 6 + ((d + 3) % 6)];
            if (sc == INF) continue;
            if (clr) {
                if (clr[ni] != 1 && clr[ni] != 2) continue;
                sc *= clr[ni];
            }
            const double nc = gcost + sc;
            if (nc > maxCost) continue;
            if (nc < (double)r.cost[ni]) {
                r.cost[ni] = (float)nc;
                r.parent[ni] = i;
                const double h = astar ? hexH(nx, ny, gx, gy) : 0;
                open.push_back({(double)r.cost[ni] + h, h, ni});
            }
        }
    }
    if (r.reached) {
        for (int i = goal; i != -1; i = r.parent[i]) r.path.push_back(i);
        std::vector<int32_t> rev(r.path.rbegin(), r.path.rend());
        r.path = rev;
    }
    return r;
}

static HexNav makeNav(const Grid& g, const char* clrId = nullptr) {
    HexNav nav(g.size);
    CHECK(nav.setStepCosts("t", g.table.data(), g.table.size()));
    if (clrId) CHECK(nav.setClearance(clrId, g.clr.data(), g.clr.size()));
    return nav;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST(flat_grid_tie_order_matches_reference) {
    // Every step costs 1: the equal-cost plateau where tie-breaking decides
    // the path. The heap's (f, h, sequence) order must give the reference's.
    Grid g;
    g.size = 24;
    g.table.assign(24 * 24 * 6, INF);
    for (int cy = 0; cy < 24; cy++) for (int cx = 0; cx < 24; cx++) for (int d = 0; d < 6; d++) {
        const int nx = cx + HexNav::STEP[cy & 1][d][0], ny = cy + HexNav::STEP[cy & 1][d][1];
        if (inB(24, nx, ny)) g.table[g.idx(cx, cy) * 6 + d] = 1.0;
    }
    HexNav nav = makeNav(g);
    for (int k = 0; k < 40; k++) {
        Lcg r(100 + k);
        const int x0 = r.next() % 24, y0 = r.next() % 24, x1 = r.next() % 24, y1 = r.next() % 24;
        std::vector<int32_t> path;
        const bool ok = nav.findPath("t", x0, y0, x1, y1, INF, path);
        RefResult ref = refSearch(g, nullptr, x0, y0, g.idx(x1, y1), INF);
        CHECK(ok == ref.reached);
        CHECK(path == ref.path);
    }
}

TEST(weighted_grid_paths_identical) {
    for (uint32_t seed = 1; seed <= 12; seed++) {
        Grid g = makeGrid(32, seed, 0.05, true);
        HexNav nav = makeNav(g);
        Lcg r(seed * 7919);
        for (int k = 0; k < 30; k++) {
            const int x0 = r.next() % 32, y0 = r.next() % 32, x1 = r.next() % 32, y1 = r.next() % 32;
            const double cap = (k % 3 == 0) ? 20.0 : INF;
            std::vector<int32_t> path;
            const bool ok = nav.findPath("t", x0, y0, x1, y1, cap, path);
            RefResult ref = refSearch(g, nullptr, x0, y0, g.idx(x1, y1), cap);
            CHECK(ok == ref.reached);
            if (ok) CHECK(path == ref.path);
        }
    }
}

TEST(clearance_paths_identical) {
    for (uint32_t seed = 21; seed <= 28; seed++) {
        Grid g = makeGrid(32, seed, 0.0, false);
        HexNav nav = makeNav(g, "c");
        Lcg r(seed * 31);
        for (int k = 0; k < 30; k++) {
            const int x0 = r.next() % 32, y0 = r.next() % 32, x1 = r.next() % 32, y1 = r.next() % 32;
            std::vector<int32_t> path;
            const bool ok = nav.findPathRadius("t", "c", x0, y0, x1, y1, INF, path);
            const uint8_t gv = g.clr[g.idx(x1, y1)];
            RefResult ref;
            if (gv == 1 || gv == 2) ref = refSearch(g, g.clr.data(), x0, y0, g.idx(x1, y1), INF);
            CHECK(ok == ref.reached);
            if (ok) CHECK(path == ref.path);
        }
    }
}

TEST(movement_field_identical) {
    Grid g = makeGrid(40, 77, 0.03, true);
    HexNav nav = makeNav(g);
    std::vector<float> cost;
    std::vector<int32_t> parent;
    CHECK(nav.movementField("t", 20, 20, 30.0, cost, parent));
    RefResult ref = refSearch(g, nullptr, 20, 20, -1, 30.0);
    CHECK(cost == ref.cost);
    CHECK(parent == ref.parent);
    // The scratch is clean afterwards: a second query answers the same.
    std::vector<float> cost2;
    std::vector<int32_t> parent2;
    CHECK(nav.movementField("t", 20, 20, 30.0, cost2, parent2));
    CHECK(cost2 == cost);
    CHECK(parent2 == parent);
}

TEST(components_only_shortcut_unreachable) {
    // Two halves split by an impassable column: cross-half queries answer
    // false without a search, same-half queries answer as the reference.
    Grid g = makeGrid(30, 5, 0.0, false);
    // Column 15 is a wall in both directions: nothing enters it and nothing
    // leaves it (an impassable cell with out-edges would still weakly connect
    // the halves — that is the directed case the search, not the labelling,
    // answers).
    auto sealColumn = [&](double v) {
        for (int y = 0; y < 30; y++) for (int d = 0; d < 6; d++) {
            g.table[g.idx(15, y) * 6 + d] = v;
            const int nx = 15 + HexNav::STEP[y & 1][d][0], ny = y + HexNav::STEP[y & 1][d][1];
            if (inB(30, nx, ny)) g.table[g.idx(nx, ny) * 6 + ((d + 3) % 6)] = v;
        }
    };
    sealColumn(INF);
    HexNav nav = makeNav(g);
    const std::vector<int32_t>& comp = nav.components("t");
    CHECK(comp.size() == 900);
    std::vector<int32_t> path;
    CHECK(!nav.findPath("t", 2, 3, 27, 3, INF, path));
    CHECK(comp[g.idx(2, 3)] != comp[g.idx(27, 3)]);
    Lcg r(99);
    for (int k = 0; k < 40; k++) {
        const int x0 = r.next() % 30, y0 = r.next() % 30, x1 = r.next() % 30, y1 = r.next() % 30;
        const bool ok = nav.findPath("t", x0, y0, x1, y1, INF, path);
        RefResult ref = refSearch(g, nullptr, x0, y0, g.idx(x1, y1), INF);
        CHECK(ok == ref.reached);
        if (ok) CHECK(path == ref.path);
    }
    // Open a gap both ways: the components refresh and the route goes
    // through it. The update rewrites the gap cell and its six neighbours.
    const int gy = 10;
    std::vector<int32_t> cellsTouched = {g.idx(15, gy)};
    for (int d = 0; d < 6; d++) {
        const int nx = 15 + HexNav::STEP[gy & 1][d][0], ny = gy + HexNav::STEP[gy & 1][d][1];
        g.table[g.idx(15, gy) * 6 + d] = 1.0;
        g.table[g.idx(nx, ny) * 6 + ((d + 3) % 6)] = 1.0;
        cellsTouched.push_back(g.idx(nx, ny));
    }
    std::vector<double> vals(cellsTouched.size() * 6);
    for (size_t k = 0; k < cellsTouched.size(); k++)
        for (int d = 0; d < 6; d++) vals[k * 6 + d] = g.table[cellsTouched[k] * 6 + d];
    CHECK(nav.updateStepCosts("t", cellsTouched.data(), cellsTouched.size(), vals.data()));
    const bool ok = nav.findPath("t", 2, 3, 27, 3, INF, path);
    RefResult ref = refSearch(g, nullptr, 2, 3, g.idx(27, 3), INF);
    CHECK(ok && ref.reached);
    CHECK(path == ref.path);
}

TEST(target_field_matches_dijkstra_and_counts_pops) {
    Grid g = makeGrid(36, 123, 0.02, true);
    HexNav nav = makeNav(g);
    std::vector<int32_t> seeds = {g.idx(18, 18), g.idx(19, 18), g.idx(18, 19), g.idx(18, 18)};
    std::vector<uint8_t> aura(36 * 36, 0), blocked(36 * 36, 0);
    for (int y = 10; y < 16; y++) for (int x = 10; x < 16; x++) aura[g.idx(x, y)] = 1;
    for (int y = 22; y < 26; y++) for (int x = 20; x < 24; x++) blocked[g.idx(x, y)] = 1;
    std::vector<double> dist;
    std::vector<int32_t> parent;
    const size_t pops = nav.targetField("t", seeds.data(), seeds.size(), blocked.data(), aura.data(),
                                        0.5, 0.25, 64, dist, parent);
    CHECK(pops > 0);
    // Naive reference: repeated relaxation to a fixed point.
    std::vector<double> ref(36 * 36, INF);
    for (int32_t s : seeds) ref[s] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int cy = 0; cy < 36; cy++) for (int cx = 0; cx < 36; cx++) {
            const int c = g.idx(cx, cy);
            if (ref[c] == INF) continue;
            const double mult = aura[c] ? 0.5 : 1.0;
            for (int d = 0; d < 6; d++) {
                const int nx = cx + HexNav::STEP[cy & 1][d][0], ny = cy + HexNav::STEP[cy & 1][d][1];
                if (!inB(36, nx, ny)) continue;
                const int n = g.idx(nx, ny);
                if (blocked[n]) continue;
                const double sc = g.table[c * 6 + d];
                if (sc == INF) continue;
                const double cand = ref[c] + sc * mult;
                if (cand < ref[n] - 1e-9) { ref[n] = cand; changed = true; }
            }
        }
    }
    CHECK(dist == ref);
    for (int i = 0; i < 36 * 36; i++) {
        if (dist[i] == INF || dist[i] == 0) { CHECK(parent[i] == -1); continue; }
        CHECK(parent[i] >= 0 && dist[parent[i]] < dist[i]);
    }
    // Seeds are FIFO-ordered and deduplicated: the fourth seed was the first
    // again and the count of settled cells is the number reached.
    size_t reached = 0;
    for (double v : dist) if (v != INF) reached++;
    CHECK(pops >= reached);
    // A cost the ring cannot order takes the heap and lands the same field.
    std::vector<double> dist2;
    std::vector<int32_t> parent2;
    nav.targetField("t", seeds.data(), seeds.size(), blocked.data(), aura.data(), 0.5, 0.3, 64, dist2, parent2);
    CHECK(dist2 == ref);
}

TEST(build_clearance_matches_walk) {
    const int size = 40, radius = 3, crush = 2;
    Lcg r(4242);
    std::vector<uint8_t> passable(size * size);
    std::vector<int8_t> elev(size * size);
    std::vector<int16_t> floors(size * size, -1);
    for (int i = 0; i < size * size; i++) {
        passable[i] = r.unit() < 0.1 ? 0 : 1;
        elev[i] = (int8_t)(r.unit() * 3);
        const double u = r.unit();
        floors[i] = u < 0.1 ? (int16_t)(1 + int(r.unit() * 4)) : (int16_t)-1;
    }
    HexNav nav(size);
    const std::vector<uint8_t>& out = nav.buildClearance("c", radius, passable.data(), elev.data(), floors.data(), crush);
    CHECK(out.size() == (size_t)(size * size));
    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++) {
        const int e0 = elev[y * size + x];
        const int q0 = x - ((y - (y & 1)) >> 1);
        uint8_t v = 0;
        bool crushing = false;
        for (int dq = -radius; dq <= radius && !v; dq++) {
            const int lo = std::max(-radius, -dq - radius), hi = std::min(radius, -dq + radius);
            for (int dr = lo; dr <= hi; dr++) {
                const int cy = y + dr, cx = q0 + dq + ((cy - (cy & 1)) >> 1);
                if (!inB(size, cx, cy)) { v = 3; break; }
                const int i = cy * size + cx;
                if (!passable[i]) { v = 3; break; }
                if (floors[i] >= 0) { if (floors[i] > crush) { v = 3; break; } crushing = true; }
                const int de = elev[i] - e0;
                if (de > 1 || de < -1) { v = 3; break; }
            }
        }
        if (!v) v = crushing ? 2 : 1;
        CHECK(out[y * size + x] == v);
    }
    CHECK(nav.hasClearance("c"));
    // Radius 0 with crushing forbidden: a structure cell cannot be stood on.
    const std::vector<uint8_t>& out0 = nav.buildClearance("c0", 0, passable.data(), elev.data(), floors.data(), -1);
    for (int i = 0; i < size * size; i++) {
        const uint8_t want = (!passable[i] || floors[i] >= 0) ? 3 : 1;
        CHECK(out0[i] == want);
    }
}

TEST(rejects_bad_sizes_and_unknown_tables) {
    HexNav nav(8);
    std::vector<double> t(8 * 8 * 6, 1.0);
    CHECK(!nav.setStepCosts("t", t.data(), t.size() - 1));
    CHECK(!nav.hasStepCosts("t"));
    std::vector<int32_t> path;
    CHECK(!nav.findPath("t", 0, 0, 7, 7, INF, path));
    CHECK(nav.setStepCosts("t", t.data(), t.size()));
    CHECK(!nav.findPath("t", 0, 0, 8, 7, INF, path));   // out of bounds
    CHECK(nav.findPath("t", 0, 0, 7, 7, INF, path));
    CHECK(path.front() == 0 && path.back() == 63);
    CHECK(nav.findPath("t", 3, 3, 3, 3, INF, path));
    CHECK(path.size() == 1 && path[0] == 27);
    CHECK(!nav.findPathRadius("t", "nope", 0, 0, 7, 7, INF, path));
}

int main() {
    printf("brogameagent hex nav tests\n");
    printf("==========================\n");

    int passed = 0;
    for (const auto& t : registry()) {
        try {
            t.fn();
            passed++;
            printf("  PASS  %s\n", t.name);
        } catch (...) {
            printf("  FAIL  %s\n", t.name);
        }
    }

    int total = static_cast<int>(registry().size());
    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
