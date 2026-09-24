// NavGrid::flowField (the square-grid integration / flow field) and the cell
// costs NavGrid::setCellCost now stores and findPath charges. Core-only.

#include <brogameagent/nav_grid.h>

#include <cmath>
#include <cstdio>
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

// A 40x20 grid of unit cells over [0,40) x [0,20).
static NavGrid makeGrid() { return NavGrid(0, 0, 40, 20, 1); }
static int at(const NavGridField& f, int x, int z) { return z * f.width + x; }

TEST(open_field_is_euclidean_and_points_at_the_goal) {
    NavGrid g = makeGrid();
    const NavGridField f = g.flowField({30.5f, 10.5f});
    CHECK(f.width == 40 && f.height == 20);
    CHECK(f.reached == 40 * 20);
    CHECK(f.dist[at(f, 30, 10)] == 0.0f);
    // Eikonal: close to the straight-line distance, well under octile/Manhattan.
    const float d = f.dist[at(f, 10, 0)];
    const float euclid = std::hypot(20.0f, 10.0f);
    CHECK(d >= euclid * 0.97f && d <= euclid * 1.08f);
    // West of the goal the flow runs east; north of it, south (+z).
    CHECK(f.dirX[at(f, 10, 10)] > 0.99f);
    CHECK(f.dirZ[at(f, 30, 2)] > 0.99f);
    const int c = at(f, 20, 5);
    CHECK(std::fabs(std::hypot(f.dirX[c], f.dirZ[c]) - 1.0f) < 1e-4f);
    CHECK(f.dirX[at(f, 30, 10)] == 0.0f && f.dirZ[at(f, 30, 10)] == 0.0f);
}

TEST(walls_detour_and_block) {
    NavGrid g = makeGrid();
    for (int z = 0; z < 20; ++z) if (z != 15) g.setWalkable(20.5f, z + 0.5f, false);   // wall, gap at z=15
    const NavGridField f = g.flowField({35.5f, 5.5f});
    CHECK(std::isinf(f.dist[at(f, 20, 5)]));
    CHECK(f.reached == 40 * 20 - 19);
    // From (5,5) the route goes through the gap, so it costs much more than
    // the straight line and the flow near the wall turns toward the gap (+z).
    // Via the gap: about 2 * hypot(15, 10) = 36 against 30 straight.
    CHECK(f.dist[at(f, 5, 5)] > 34.0f && f.dist[at(f, 5, 5)] < 39.0f);
    CHECK(f.dirZ[at(f, 18, 5)] > 0.5f);
    // Fully walled off: the far side is unreached.
    g.setWalkable(20.5f, 15.5f, false);
    const NavGridField cut = g.flowField({35.5f, 5.5f});
    CHECK(std::isinf(cut.dist[at(cut, 5, 5)]));
    CHECK(cut.dirX[at(cut, 5, 5)] == 0.0f);
    CHECK(cut.reached == 19 * 20);
}

TEST(costs_and_extra_cost_reprice_the_wave) {
    NavGrid g = makeGrid();
    const float open = g.flowField({35.5f, 10.5f}).dist[10 * 40 + 5];
    // A mud band across the whole map (no way round) costs its price.
    for (int z = 0; z < 20; ++z) for (int x = 18; x < 22; ++x) g.setCellCost(x + 0.5f, z + 0.5f, 5.0f);
    CHECK(g.cellCost(19.5f, 3.5f) == 5.0f);
    CHECK(g.cellCost(1.5f, 3.5f) == 1.0f);
    const float muddy = g.flowField({35.5f, 10.5f}).dist[10 * 40 + 5];
    CHECK(muddy > open + 4 * 4 - 0.5f && muddy < open + 4 * 4 + 0.5f);
    // extraCost adds per cell (a danger map); costs overrides the stored ones.
    std::vector<float> extra(800, 0.0f);
    extra[10 * 40 + 10] = 100.0f;
    const NavGridField e = g.flowField({35.5f, 10.5f}, extra.data());
    CHECK(e.dist[10 * 40 + 5] > muddy);                 // the danger cell is avoided...
    CHECK(e.dist[10 * 40 + 5] < muddy + 5.0f);         // ...by going round, not through
    std::vector<float> flat(800, 1.0f);
    const NavGridField o = g.flowField({35.5f, 10.5f}, nullptr, flat.data());
    CHECK(std::fabs(o.dist[10 * 40 + 5] - open) < 1e-3f);
    // A blocked or off-grid goal gives an empty field.
    g.setWalkable(35.5f, 10.5f, false);
    CHECK(g.flowField({35.5f, 10.5f}).reached == 0);
    CHECK(g.flowField({-3.0f, 10.5f}).reached == 0);
    CHECK(std::isinf(g.cellCost(35.5f, 10.5f)));
}

TEST(find_path_charges_cell_costs) {
    NavGrid g = makeGrid();
    // A mud block straight between start and goal, with open ground round it.
    for (int z = 5; z < 15; ++z) for (int x = 15; x < 25; ++x) g.setCellCost(x + 0.5f, z + 0.5f, 20.0f);
    const auto path = g.findPath({5.5f, 10.5f}, {34.5f, 10.5f});
    CHECK(path.size() >= 3);                            // bends: not one straight segment
    bool crossesMud = false;
    for (const auto& p : path) if (p.x > 15 && p.x < 25 && p.y > 5 && p.y < 15) crossesMud = true;
    CHECK(!crossesMud);
    // setCellCost with a valid cost re-opens a blocked cell; 0 blocks it.
    g.setCellCost(2.5f, 2.5f, 0.0f);
    CHECK(!g.isWalkable(2.5f, 2.5f));
    g.setCellCost(2.5f, 2.5f, 3.0f);
    CHECK(g.isWalkable(2.5f, 2.5f) && g.cellCost(2.5f, 2.5f) == 3.0f);
}

int main() {
    printf("brogameagent nav grid field tests\n");
    printf("=================================\n");
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
    const int total = static_cast<int>(registry().size());
    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
