// NavMesh tests — Recast/Detour-backed polygon navmesh: bake from triangle
// soup, obstacle routing, slope limits, multi-level (bridge over floor —
// the case NavGrid cannot represent), snapping, raycast, serialization,
// determinism, and dynamic obstacles (tiled dtTileCache bake: add/remove
// cylinder/box/oriented-box obstacles with incremental tile rebuilds).
// Core-only: builds with BROGAMEAGENT_WITH_NN=OFF.

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

static bool pathsEqual(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::memcmp(&a[i], &b[i], sizeof(Vec3)) != 0) return false;
    return true;
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

    // The high plateau exists as walkable mesh but is disconnected — the
    // path clamps to the closest reachable point on the low plateau and
    // reports partial.
    auto res = nm.findPathEx({-8, 0, 0}, {8, rise, 0});
    CHECK(res.partial);
    CHECK(!res.points.empty());
    CHECK(res.points.back().y < 0.5f);        // never climbed the cliff
    CHECK(res.points.back().x > -4.0f);       // walked toward the ramp base

    // findPath() mirrors the clamped points.
    auto path = nm.findPath({-8, 0, 0}, {8, rise, 0});
    CHECK(pathsEqual(path, res.points));

    // requireFullPath restores hard-fail semantics: empty but flagged
    // partial, distinguishing "unreachable" from "unsnappable".
    auto strict = nm.findPathEx({-8, 0, 0}, {8, rise, 0}, NavMesh::kDefaultExtents, true);
    CHECK(strict.points.empty());
    CHECK(strict.partial);
    auto unsnappable = nm.findPathEx({-8, 0, 0}, {500, 0, 500});
    CHECK(unsnappable.points.empty());
    CHECK(!unsnappable.partial);

    // Both endpoints individually are on the mesh (so the clamping above is
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
    CHECK(pathsEqual(original, reloaded));

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
    CHECK(pathsEqual(pathA, pathB));
}

// ─── Dynamic obstacles (tiled tile-cache bake) ──────────────────────────────

static NavMeshBakeConfig obstacleConfig() {
    NavMeshBakeConfig cfg;
    cfg.dynamicObstacles = true;
    cfg.tileSize = 8.0f;
    return cfg;
}

// Pump update() until the cache reports up-to-date. Returns the number of
// calls it took, or -1 if it failed to settle within maxIters.
static int pumpUntilSettled(NavMesh& nm, int maxIters = 64) {
    for (int i = 0; i < maxIters; i++)
        if (nm.update()) return i + 1;
    return -1;
}

TEST(tiled_bake_routes_like_static_and_gates_the_api) {
    NavMesh nm;
    CHECK(bakeSoup(nm, floorWithBox(), obstacleConfig()));
    CHECK(nm.valid());
    CHECK(nm.supportsObstacles());
    CHECK(nm.obstacleCount() == 0);
    CHECK(!nm.obstaclesPending());
    CHECK(nm.update());  // nothing pending: immediately up to date

    // Same routing contract as the static bake: bends around the baked box,
    // never through it.
    auto path = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(path.size() >= 3);
    const float straight = bromath::vdist(path.front(), path.back());
    CHECK(pathLength(path) > straight + 0.2f);
    for (const Vec3& p : path)
        CHECK(!(std::abs(p.x) < 1.0f && std::abs(p.z) < 1.0f));

    // Tiled meshes do not serialize (state lives in the tile cache).
    std::vector<uint8_t> blob;
    CHECK(!nm.saveTo(blob));

    // Static meshes reject the obstacle API cleanly.
    NavMesh flat;
    CHECK(bakeSoup(flat, floorWithBox()));
    CHECK(!flat.supportsObstacles());
    CHECK(flat.addObstacle({0, 0, 0}, 1.0f, 2.0f) == 0);
    CHECK(!flat.lastError().empty());
    CHECK(flat.addBoxObstacle({-1, 0, -1}, {1, 2, 1}) == 0);
    CHECK(!flat.removeObstacle(1));
    CHECK(flat.update());  // no-op, reports up to date
    CHECK(flat.generation() >= 1);  // bake counts as a surface change
}

TEST(cylinder_obstacle_blocks_then_restores) {
    Soup s;
    s.addFloor(-10, -10, 10, 10, 0, 0);
    NavMesh nm;
    CHECK(bakeSoup(nm, s, obstacleConfig()));
    const uint32_t gen0 = nm.generation();

    auto before = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!before.empty());
    const float straightLen = pathLength(before);

    NavMesh::ObstacleId ob = nm.addObstacle({0, -0.5f, 0}, 2.0f, 3.0f);
    CHECK(ob != 0);
    CHECK(nm.obstacleCount() == 1);
    CHECK(nm.obstaclesPending());
    CHECK(nm.generation() == gen0);  // nothing applied until update() pumps

    // Path is unchanged until the touched tiles rebuild.
    auto stillOpen = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!stillOpen.empty());
    CHECK(std::abs(pathLength(stillOpen) - straightLen) < 0.01f);

    const int pumps = pumpUntilSettled(nm);
    CHECK(pumps > 0);  // converges incrementally (one tile rebuild per call)
    CHECK(!nm.obstaclesPending());
    CHECK(nm.generation() == gen0 + 1);

    auto blocked = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!blocked.empty());
    CHECK(blocked.size() >= 3);                       // bends — no straight run
    CHECK(pathLength(blocked) > straightLen + 0.2f);  // detours around it
    for (const Vec3& p : blocked)
        CHECK(std::sqrt(p.x * p.x + p.z * p.z) > 1.9f);  // outside the cylinder

    CHECK(nm.removeObstacle(ob));
    CHECK(nm.obstacleCount() == 0);
    CHECK(nm.obstaclesPending());
    CHECK(pumpUntilSettled(nm) > 0);
    CHECK(nm.generation() == gen0 + 2);

    auto restored = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!restored.empty());
    CHECK(pathLength(restored) < straightLen + 0.5f);  // straight again

    // Stale handle: a second remove is a clean no-op.
    CHECK(!nm.removeObstacle(ob));
    CHECK(!nm.removeObstacle(0));
}

TEST(box_obstacle_severs_corridor_then_restores) {
    Soup s;
    s.addFloor(-10, -2, 10, 2, 0, 0);  // 4 m wide corridor
    NavMesh nm;
    CHECK(bakeSoup(nm, s, obstacleConfig()));
    CHECK(!nm.findPath({-8, 0, 0}, {8, 0, 0}).empty());

    // AABB spanning the corridor's full width: the goal becomes unreachable —
    // the path clamps to the closest reachable point before the box (partial).
    NavMesh::ObstacleId ob = nm.addBoxObstacle({-1, -1, -3}, {1, 3, 3});
    CHECK(ob != 0);
    CHECK(pumpUntilSettled(nm) > 0);
    auto severed = nm.findPathEx({-8, 0, 0}, {8, 0, 0});
    CHECK(severed.partial);
    CHECK(!severed.points.empty());
    CHECK(severed.points.back().x < -0.9f);  // stops at the box's near face
    // requireFullPath restores hard-fail semantics.
    auto strict = nm.findPathEx({-8, 0, 0}, {8, 0, 0}, NavMesh::kDefaultExtents, true);
    CHECK(strict.points.empty());
    CHECK(strict.partial);

    // Both endpoints still snap — the clamping is connectivity, not snapping.
    Vec3 snapped;
    CHECK(nm.nearestPoint({-8, 0, 0}, snapped));
    CHECK(nm.nearestPoint({8, 0, 0}, snapped));

    CHECK(nm.removeObstacle(ob));
    CHECK(pumpUntilSettled(nm) > 0);
    auto reopened = nm.findPathEx({-8, 0, 0}, {8, 0, 0});
    CHECK(!reopened.points.empty());
    CHECK(!reopened.partial);
}

TEST(oriented_box_obstacle_carves_rotated_footprint) {
    Soup s;
    s.addFloor(-10, -10, 10, 10, 0, 0);
    NavMesh nm;
    CHECK(bakeSoup(nm, s, obstacleConfig()));

    const float straightLen = pathLength(nm.findPath({-8, 0, 0}, {8, 0, 0}));

    // Long thin box rotated 45°: blocks the straight run through the middle.
    NavMesh::ObstacleId ob =
        nm.addBoxObstacle({0, 0.5f, 0}, {3.0f, 1.5f, 0.75f},
                          0.25f * 3.14159265f);
    CHECK(ob != 0);
    CHECK(pumpUntilSettled(nm) > 0);

    auto blocked = nm.findPath({-8, 0, 0}, {8, 0, 0});
    CHECK(!blocked.empty());
    CHECK(pathLength(blocked) > straightLen + 0.5f);

    CHECK(nm.removeObstacle(ob));
    CHECK(pumpUntilSettled(nm) > 0);
    CHECK(pathLength(nm.findPath({-8, 0, 0}, {8, 0, 0})) < straightLen + 0.5f);
}

TEST(many_obstacles_track_counts_and_generations) {
    Soup s;
    s.addFloor(-10, -10, 10, 10, 0, 0);
    NavMesh nm;
    CHECK(bakeSoup(nm, s, obstacleConfig()));
    const uint32_t gen0 = nm.generation();

    // A batch of adds applied by one pump loop bumps generation ONCE.
    std::vector<NavMesh::ObstacleId> ids;
    for (int i = 0; i < 4; i++) {
        NavMesh::ObstacleId ob =
            nm.addObstacle({-6.0f + 4.0f * static_cast<float>(i), -0.5f, -5.0f},
                           1.0f, 2.0f);
        CHECK(ob != 0);
        ids.push_back(ob);
    }
    CHECK(nm.obstacleCount() == 4);
    CHECK(pumpUntilSettled(nm) > 0);
    CHECK(nm.generation() == gen0 + 1);

    // Remove them all: surface restores, one more generation.
    for (NavMesh::ObstacleId ob : ids) CHECK(nm.removeObstacle(ob));
    CHECK(nm.obstacleCount() == 0);
    CHECK(pumpUntilSettled(nm) > 0);
    CHECK(nm.generation() == gen0 + 2);
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
