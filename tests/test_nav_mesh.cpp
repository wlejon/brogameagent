// NavMesh tests — Recast/Detour-backed polygon navmesh: bake from triangle
// soup, obstacle routing, slope limits, multi-level (bridge over floor —
// the case NavGrid cannot represent), snapping, raycast, serialization,
// determinism. Core-only: builds with BROGAMEAGENT_WITH_NN=OFF.

#define _USE_MATH_DEFINES
#include <cmath>

#include <brogameagent/brogameagent.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace brogameagent;
using bromath::Vec3;

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
#define CHECK_NEAR(a, b, eps) check(std::abs((a) - (b)) < (eps), #a " ~= " #b, __LINE__)

// ─── Triangle-soup helpers ──────────────────────────────────────────────────

struct Soup {
    std::vector<float> verts;    // xyz triples
    std::vector<uint32_t> idx;

    uint32_t addVert(Vec3 p) {
        verts.insert(verts.end(), {p.x, p.y, p.z});
        return static_cast<uint32_t>(verts.size() / 3 - 1);
    }
    void addTri(Vec3 a, Vec3 b, Vec3 c) {
        idx.push_back(addVert(a));
        idx.push_back(addVert(b));
        idx.push_back(addVert(c));
    }
    // Quad from 4 corners given counter-clockwise when viewed along the
    // desired normal. For an upward-facing (+Y) quad pass corners CCW seen
    // from above.
    void addQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
        addTri(a, b, c);
        addTri(a, c, d);
    }
    // Horizontal rectangle spanning [x0,x1]x[z0,z1]; y may differ per corner
    // to make ramps (y at (x0,*) = y0, y at (x1,*) = y1). Faces +Y.
    void addFloor(float x0, float z0, float x1, float z1, float y0, float y1) {
        addQuad({x0, y0, z0}, {x0, y0, z1}, {x1, y1, z1}, {x1, y1, z0});
    }
    // Closed vertical-sided box (4 walls + top) — an obstacle sitting on the
    // ground. Outward-facing winding (not that it matters for blocking:
    // solid spans block regardless of walkability).
    void addBox(float cx, float cz, float hw, float hd, float y0, float y1) {
        const float x0 = cx - hw, x1 = cx + hw, z0 = cz - hd, z1 = cz + hd;
        addQuad({x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}); // top
        addQuad({x0, y0, z0}, {x0, y1, z0}, {x0, y1, z1}, {x0, y0, z1}); // -X
        addQuad({x1, y0, z1}, {x1, y1, z1}, {x1, y1, z0}, {x1, y0, z0}); // +X
        addQuad({x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}, {x0, y0, z0}); // -Z
        addQuad({x0, y0, z1}, {x0, y1, z1}, {x1, y1, z1}, {x1, y0, z1}); // +Z
    }
};

static NavMeshBakeConfig testConfig() {
    NavMeshBakeConfig cfg; // library defaults: 0.5 m radius, 2 m tall agent
    return cfg;
}

// 20x20 m floor with a 2x2 m, 3 m tall box at the center.
static Soup floorWithBox() {
    Soup s;
    s.addFloor(-10, -10, 10, 10, 0, 0);
    s.addBox(0, 0, 1, 1, 0, 3);
    return s;
}

static bool bakeSoup(NavMesh& nm, const Soup& s,
                     const NavMeshBakeConfig& cfg = testConfig()) {
    bool ok = nm.bake(s.verts.data(), s.verts.size() / 3,
                      s.idx.data(), s.idx.size(), cfg);
    if (!ok) printf("    bake failed: %s\n", nm.lastError().c_str());
    return ok;
}

static float pathLength(const std::vector<Vec3>& path) {
    float len = 0;
    for (size_t i = 1; i < path.size(); i++)
        len += bromath::vdist(path[i - 1], path[i]);
    return len;
}

// ─── Bake + obstacle routing ────────────────────────────────────────────────

TEST(bake_floor_with_box_and_route_around) {
    NavMesh nm;
    CHECK(!nm.valid());
    CHECK(bakeSoup(nm, floorWithBox()));
    CHECK(nm.valid());

    const Vec3 start{-8, 0, 0}, end{8, 0, 0};
    auto path = nm.findPath(start, end);
    CHECK(path.size() >= 3); // must bend around the box

    // Endpoints snap onto the floor near the requested points.
    CHECK_NEAR(path.front().x, start.x, 0.5f);
    CHECK_NEAR(path.front().z, start.z, 0.5f);
    CHECK_NEAR(path.back().x, end.x, 0.5f);
    CHECK_NEAR(path.back().z, end.z, 0.5f);

    // Detour around the box is longer than the straight line...
    const float straight = bromath::vdist(path.front(), path.back());
    CHECK(pathLength(path) > straight + 0.2f);

    // ...and no waypoint lands inside the box footprint inflated by (half)
    // the agent radius — contour simplification may shave a little off the
    // full-radius clearance, but never cut into the box itself.
    const float inflate = 1.0f + 0.5f * 0.5f;
    for (const Vec3& p : path)
        CHECK(!(std::abs(p.x) < inflate && std::abs(p.z) < inflate));
}

TEST(bake_reports_errors) {
    NavMesh nm;
    // Degenerate input: no triangles.
    CHECK(!nm.bake(nullptr, 0, nullptr, 0, testConfig()));
    CHECK(!nm.lastError().empty());

    // All triangles wound clockwise (downward normals) → nothing walkable.
    Soup s;
    s.addQuad({10, 0, -10}, {10, 0, 10}, {-10, 0, 10}, {-10, 0, -10}); // CW from above
    CHECK(!nm.bake(s.verts.data(), s.verts.size() / 3, s.idx.data(), s.idx.size(),
                   testConfig()));
    CHECK(!nm.lastError().empty());
    CHECK(!nm.valid());
}

// ─── Slopes ─────────────────────────────────────────────────────────────────

// Two plateaus joined by a 4 m-long ramp of the given rise.
static Soup rampSoup(float rise) {
    Soup s;
    s.addFloor(-10, -4, -2, 4, 0, 0);        // low plateau
    s.addFloor(-2, -4, 2, 4, 0, rise);        // ramp
    s.addFloor(2, -4, 10, 4, rise, rise);     // high plateau
    return s;
}

TEST(slope_within_limit_is_walkable) {
    const float rise = 4.0f * std::tan(20.0f * 3.14159265f / 180.0f); // 20 deg
    NavMesh nm;
    CHECK(bakeSoup(nm, rampSoup(rise)));

    auto path = nm.findPath({-8, 0, 0}, {8, rise, 0});
    CHECK(!path.empty());
    CHECK_NEAR(path.back().y, rise, 0.5f); // actually got up there
    // The path traverses intermediate heights (crosses the ramp).
    bool sawMid = false;
    for (const Vec3& p : path)
        if (p.y > rise * 0.2f && p.y < rise * 0.95f) sawMid = true;
    CHECK(sawMid || path.size() == 2); // straight funnel may skip mid points
}

TEST(slope_beyond_limit_is_not_walkable) {
    const float rise = 4.0f * std::tan(60.0f * 3.14159265f / 180.0f); // 60 deg
    NavMesh nm;
    CHECK(bakeSoup(nm, rampSoup(rise))); // default max slope: 45 deg

    // The high plateau exists as walkable mesh but is disconnected — a
    // partial path toward it must be reported as failure.
    auto path = nm.findPath({-8, 0, 0}, {8, rise, 0});
    CHECK(path.empty());

    // Both endpoints individually are on the mesh (so the failure above is
    // about connectivity, not snapping).
    Vec3 snapped;
    CHECK(nm.nearestPoint({-8, 0, 0}, snapped));
    CHECK(nm.nearestPoint({8, rise, 0}, snapped));
}

// ─── Multi-level: bridge over a floor (the NavGrid-impossible case) ─────────

TEST(bridge_over_floor_resolves_levels) {
    Soup s;
    s.addFloor(-10, -10, 10, 10, 0, 0);   // ground floor
    s.addFloor(-10, -2, 10, 2, 4, 4);     // bridge deck 4 m up, crossing above
    NavMesh nm;
    CHECK(bakeSoup(nm, s));

    // Path on the deck stays on the deck…
    auto deckPath = nm.findPath({-8, 4, 0}, {8, 4, 0});
    CHECK(!deckPath.empty());
    for (const Vec3& p : deckPath) CHECK(p.y > 3.5f);

    // …and the path on the ground floor passes UNDER the deck at y≈0.
    auto floorPath = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!floorPath.empty());
    for (const Vec3& p : floorPath) CHECK(p.y < 0.5f);

    // The floor path is straight (nothing blocks it), the deck is 4 m above.
    CHECK(pathLength(floorPath) <
          bromath::vdist(floorPath.front(), floorPath.back()) + 0.5f);

    // nearestPoint disambiguates by y: a mid-air point just under the deck
    // snaps to the deck, a point near the ground snaps to the floor.
    Vec3 snapped;
    CHECK(nm.nearestPoint({0, 3.5f, 0}, snapped, {0.5f, 1.0f, 0.5f}));
    CHECK(snapped.y > 3.5f);
    CHECK(nm.nearestPoint({0, 0.5f, 0}, snapped, {0.5f, 1.0f, 0.5f}));
    CHECK(snapped.y < 0.5f);
}

// ─── nearestPoint / raycast ─────────────────────────────────────────────────

TEST(nearest_point_snaps_onto_mesh) {
    NavMesh nm;
    CHECK(bakeSoup(nm, floorWithBox()));

    // Point floating above open floor snaps straight down.
    Vec3 snapped;
    CHECK(nm.nearestPoint({5, 0.8f, 5}, snapped));
    CHECK_NEAR(snapped.x, 5.0f, 0.3f);
    CHECK_NEAR(snapped.y, 0.0f, 0.3f);
    CHECK_NEAR(snapped.z, 5.0f, 0.3f);

    // Point outside the floor snaps to the (eroded) boundary.
    CHECK(nm.nearestPoint({12, 0, 0}, snapped, {4, 2, 4}));
    CHECK(snapped.x < 10.0f);
    CHECK(snapped.x > 8.0f);

    // Nothing within tiny extents far off the mesh.
    CHECK(!nm.nearestPoint({50, 0, 50}, snapped, {1, 1, 1}));
}

TEST(raycast_stops_at_obstacle_and_edge) {
    NavMesh nm;
    CHECK(bakeSoup(nm, floorWithBox()));

    // Toward the box: blocked in front of its eroded boundary.
    auto hit = nm.raycast({-5, 0, 0}, {5, 0, 0});
    CHECK(hit.hit);
    CHECK(hit.t < 1.0f);
    CHECK(hit.point.x > -2.5f);
    CHECK(hit.point.x < -0.9f);

    // Across open floor: unobstructed.
    auto open = nm.raycast({-5, 0, 0}, {-5, 0, 5});
    CHECK(!open.hit);
    CHECK_NEAR(open.t, 1.0f, 1e-6f);
    CHECK_NEAR(open.point.z, 5.0f, 1e-4f);

    // Off the edge of the world: stops at the mesh boundary.
    auto edge = nm.raycast({-5, 0, 0}, {-5, 0, 20});
    CHECK(edge.hit);
    CHECK(edge.point.z < 10.0f);
    CHECK(edge.point.z > 8.0f);
}

TEST(random_point_lands_on_mesh_and_is_seed_deterministic) {
    NavMesh nm;
    CHECK(bakeSoup(nm, floorWithBox()));

    Vec3 a, b, c;
    CHECK(nm.randomPoint(42, a));
    CHECK(nm.randomPoint(42, b));
    CHECK(nm.randomPoint(7, c));
    CHECK(a.x == b.x && a.y == b.y && a.z == b.z); // same seed, same point
    // On the mesh: snapping it moves it (essentially) nowhere.
    Vec3 snapped;
    CHECK(nm.nearestPoint(a, snapped, {0.5f, 0.5f, 0.5f}));
    CHECK(bromath::vdist(a, snapped) < 0.2f);
    // In bounds.
    CHECK(std::abs(a.x) <= 10.0f && std::abs(a.z) <= 10.0f);
}

// ─── Serialization + determinism ────────────────────────────────────────────

static bool pathsIdentical(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::memcmp(&a[i], &b[i], sizeof(Vec3)) != 0) return false;
    return true;
}

TEST(save_load_round_trip_preserves_paths) {
    NavMesh nm;
    CHECK(bakeSoup(nm, floorWithBox()));
    auto original = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!original.empty());

    std::vector<uint8_t> blob;
    CHECK(nm.saveTo(blob));
    CHECK(!blob.empty());

    NavMesh loaded;
    CHECK(loaded.loadFrom(blob.data(), blob.size()));
    CHECK(loaded.valid());
    auto reloaded = loaded.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(pathsIdentical(original, reloaded));

    // Garbage data is rejected with a diagnosis, not accepted.
    std::vector<uint8_t> garbage(64, 0xAB);
    NavMesh bad;
    CHECK(!bad.loadFrom(garbage.data(), garbage.size()));
    CHECK(!bad.lastError().empty());
    CHECK(!bad.valid());
}

TEST(bake_is_deterministic) {
    Soup s = floorWithBox();
    NavMesh a, b;
    CHECK(bakeSoup(a, s));
    CHECK(bakeSoup(b, s));

    std::vector<uint8_t> blobA, blobB;
    CHECK(a.saveTo(blobA));
    CHECK(b.saveTo(blobB));
    CHECK(blobA == blobB); // bit-identical bake

    auto pathA = a.findPath({-8, 0, 0}, {6, 0, 7});
    auto pathB = b.findPath({-8, 0, 0}, {6, 0, 7});
    CHECK(!pathA.empty());
    CHECK(pathsIdentical(pathA, pathB));
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("brogameagent navmesh tests\n");
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
