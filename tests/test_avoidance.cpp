// ORCA local-avoidance tests — AvoidanceSim solver correctness plus the
// World::tick() integration (avoidance pass over path-following agents).
// Core-only: builds with BROGAMEAGENT_WITH_NN=OFF.

#define _USE_MATH_DEFINES
#include <cmath>

#include <brogameagent/brogameagent.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace brogameagent;
using bromath::Vec2;

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

// ─── Helpers ────────────────────────────────────────────────────────────────

// Arrive-style preferred velocity: full speed toward the goal, decelerating
// inside the last unit of distance.
static Vec2 prefTowards(Vec2 pos, Vec2 goal, float maxSpeed) {
    Vec2 d = goal - pos;
    float len = bromath::vlen(d);
    if (len < 1e-4f) return {0, 0};
    if (len > 1.0f) return d / len * maxSpeed;
    return d * maxSpeed;
}

// Min distance between any agent pair in the sim.
static float minPairDist(const AvoidanceSim& sim) {
    float best = 1e30f;
    for (int i = 0; i < sim.agentCount(); i++) {
        for (int j = i + 1; j < sim.agentCount(); j++) {
            float d = bromath::vdist(sim.position(i), sim.position(j));
            if (d < best) best = d;
        }
    }
    return best;
}

// ─── AvoidanceSim: agent-agent ──────────────────────────────────────────────

TEST(avoid_head_on_agents_pass_without_overlap) {
    AvoidanceSim sim;
    AvoidanceAgentParams p;
    p.radius = 0.5f;
    p.maxSpeed = 4.0f;
    int a = sim.addAgent({-6, 0}, p);
    int b = sim.addAgent({6, 0}, p);
    Vec2 goalA{6, 0}, goalB{-6, 0};

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    bool arrivedA = false, arrivedB = false;
    for (int step = 0; step < 15 * 60; step++) {
        sim.setPrefVelocity(a, prefTowards(sim.position(a), goalA, p.maxSpeed));
        sim.setPrefVelocity(b, prefTowards(sim.position(b), goalB, p.maxSpeed));
        sim.step(dt);
        minDist = std::min(minDist, minPairDist(sim));
        arrivedA = bromath::vdist(sim.position(a), goalA) < 0.3f;
        arrivedB = bromath::vdist(sim.position(b), goalB) < 0.3f;
        if (arrivedA && arrivedB) break;
    }
    CHECK(arrivedA && arrivedB);
    // ORCA guarantee: discs never interpenetrate (small tolerance for the
    // LP3 fallback / float accumulation).
    CHECK(minDist >= (p.radius * 2.0f) * 0.9f);
}

TEST(avoid_circle_crossing_crowd_no_interpenetration) {
    AvoidanceSim sim;
    AvoidanceAgentParams p;
    p.radius = 0.4f;
    p.maxSpeed = 4.0f;
    const int N = 10;
    const float R = 8.0f;
    std::vector<Vec2> goals((size_t)N);
    for (int i = 0; i < N; i++) {
        // Slight deterministic angular skew breaks the perfect symmetry that
        // otherwise deadlocks any reciprocal scheme.
        float ang = (float)i / N * 2.0f * (float)M_PI + 0.013f * (float)i;
        Vec2 start{R * std::cos(ang), R * std::sin(ang)};
        goals[(size_t)i] = -1.0f * start;  // antipode: everyone crosses the center
        sim.addAgent(start, p);
    }

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    int arrived = 0;
    for (int step = 0; step < 30 * 60; step++) {
        for (int i = 0; i < N; i++)
            sim.setPrefVelocity(i, prefTowards(sim.position(i), goals[(size_t)i], p.maxSpeed));
        sim.step(dt);
        minDist = std::min(minDist, minPairDist(sim));
        arrived = 0;
        for (int i = 0; i < N; i++)
            if (bromath::vdist(sim.position(i), goals[(size_t)i]) < 0.5f) arrived++;
        if (arrived == N) break;
    }
    CHECK(arrived == N);
    CHECK(minDist >= (p.radius * 2.0f) * 0.9f);
}

TEST(avoid_nonresponsive_agent_is_steered_around) {
    AvoidanceSim sim;
    AvoidanceAgentParams p;
    p.radius = 0.5f;
    p.maxSpeed = 4.0f;
    int blocker = sim.addAgent({0, 0}, p);
    sim.setResponsive(blocker, false);
    int mover = sim.addAgent({-6, 0.01f}, p);
    Vec2 goal{6, 0};

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    bool arrived = false;
    for (int step = 0; step < 15 * 60; step++) {
        sim.setPrefVelocity(mover, prefTowards(sim.position(mover), goal, p.maxSpeed));
        sim.step(dt);
        minDist = std::min(minDist, minPairDist(sim));
        if (bromath::vdist(sim.position(mover), goal) < 0.3f) { arrived = true; break; }
    }
    CHECK(arrived);
    CHECK(minDist >= (p.radius * 2.0f) * 0.9f);
    // The blocker never had its velocity solved, so it never moved.
    CHECK_NEAR(sim.position(blocker).x, 0.0f, 1e-6f);
    CHECK_NEAR(sim.position(blocker).y, 0.0f, 1e-6f);
}

TEST(avoid_elevation_filter_separates_levels) {
    // Two agents crossing head-on but on different vertical levels (bridge
    // over tunnel). With disjoint vertical spans they must NOT see each
    // other: both walk essentially straight (only the tiny symmetry dither
    // bends the path) instead of swerving. Same setup with overlapping
    // spans still avoids — proving the filter, not a broken solve.
    const float dt = 1.0f / 60.0f;

    // Different levels: |dy| = 6 > (2 + 2) / 2 with the default height 2.
    {
        AvoidanceSim sim;
        AvoidanceAgentParams p;
        p.radius = 0.5f;
        p.maxSpeed = 4.0f;
        int a = sim.addAgent({-5, 0}, p);
        int b = sim.addAgent({5, 0}, p);
        sim.setElevation(a, 0.0f);
        sim.setElevation(b, 6.0f);
        CHECK_NEAR(sim.elevation(b), 6.0f, 1e-6f);

        float maxLateral = 0.0f;
        for (int step = 0; step < 5 * 60; step++) {
            sim.setPrefVelocity(a, prefTowards(sim.position(a), {5, 0}, p.maxSpeed));
            sim.setPrefVelocity(b, prefTowards(sim.position(b), {-5, 0}, p.maxSpeed));
            sim.step(dt);
            maxLateral = std::max(maxLateral, std::abs(sim.position(a).y));
            maxLateral = std::max(maxLateral, std::abs(sim.position(b).y));
        }
        CHECK(bromath::vdist(sim.position(a), {5, 0}) < 0.3f);
        CHECK(bromath::vdist(sim.position(b), {-5, 0}) < 0.3f);
        // Straight-line pass-through: no avoidance swerve (dither alone
        // deflects < ~0.06 over the 10-unit run).
        CHECK(maxLateral < 0.15f);
    }

    // Overlapping spans (|dy| = 1 < 2): normal reciprocal avoidance.
    {
        AvoidanceSim sim;
        AvoidanceAgentParams p;
        p.radius = 0.5f;
        p.maxSpeed = 4.0f;
        int a = sim.addAgent({-5, 0}, p);
        int b = sim.addAgent({5, 0}, p);
        sim.setElevation(a, 0.0f);
        sim.setElevation(b, 1.0f);

        float minDist = 1e30f;
        bool arrivedA = false, arrivedB = false;
        for (int step = 0; step < 15 * 60; step++) {
            sim.setPrefVelocity(a, prefTowards(sim.position(a), {5, 0}, p.maxSpeed));
            sim.setPrefVelocity(b, prefTowards(sim.position(b), {-5, 0}, p.maxSpeed));
            sim.step(dt);
            minDist = std::min(minDist, minPairDist(sim));
            arrivedA = bromath::vdist(sim.position(a), {5, 0}) < 0.3f;
            arrivedB = bromath::vdist(sim.position(b), {-5, 0}) < 0.3f;
            if (arrivedA && arrivedB) break;
        }
        CHECK(arrivedA && arrivedB);
        CHECK(minDist >= (p.radius * 2.0f) * 0.9f);
    }
}

// ─── AvoidanceSim: static obstacles ─────────────────────────────────────────

// Distance from p to segment [a,b].
static float distToSegment(Vec2 a, Vec2 b, Vec2 p) {
    Vec2 ab = b - a;
    float t = bromath::vdot(p - a, ab) / bromath::vlen2(ab);
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return bromath::vdist(p, a + t * ab);
}

TEST(avoid_wall_segment_is_respected) {
    AvoidanceSim sim;
    CHECK(sim.addObstacleSegment({-2, 0}, {2, 0}));
    AvoidanceAgentParams p;
    p.radius = 0.4f;
    p.maxSpeed = 4.0f;
    // Diagonal route whose straight line crosses the wall interior — the
    // agent must slide along it and round the end at x=2.
    int a = sim.addAgent({-3, -2}, p);
    Vec2 goal{3, 2};

    const float dt = 1.0f / 60.0f;
    float minWallDist = 1e30f;
    bool arrived = false;
    for (int step = 0; step < 20 * 60; step++) {
        sim.setPrefVelocity(a, prefTowards(sim.position(a), goal, p.maxSpeed));
        sim.step(dt);
        minWallDist = std::min(minWallDist, distToSegment({-2, 0}, {2, 0}, sim.position(a)));
        if (bromath::vdist(sim.position(a), goal) < 0.3f) { arrived = true; break; }
    }
    CHECK(arrived);
    CHECK(minWallDist >= p.radius * 0.9f);
}

TEST(avoid_box_obstacle_is_respected) {
    AvoidanceSim sim;
    AABB box{0, 0, 1.5f, 0.5f};
    CHECK(sim.addObstacleBox(box));
    AvoidanceAgentParams p;
    p.radius = 0.4f;
    p.maxSpeed = 4.0f;
    int a = sim.addAgent({-4, -2}, p);
    Vec2 goal{4, 2};

    const float dt = 1.0f / 60.0f;
    float minClearance = 1e30f;
    bool arrived = false;
    for (int step = 0; step < 20 * 60; step++) {
        sim.setPrefVelocity(a, prefTowards(sim.position(a), goal, p.maxSpeed));
        sim.step(dt);
        // Clearance to the box (0 inside).
        Vec2 pos = sim.position(a);
        float dx = std::max(std::abs(pos.x - box.cx) - box.hw, 0.0f);
        float dz = std::max(std::abs(pos.y - box.cz) - box.hd, 0.0f);
        minClearance = std::min(minClearance, std::sqrt(dx * dx + dz * dz));
        if (bromath::vdist(pos, goal) < 0.3f) { arrived = true; break; }
    }
    CHECK(arrived);
    CHECK(minClearance >= p.radius * 0.9f);
}

TEST(avoid_degenerate_obstacles_rejected) {
    AvoidanceSim sim;
    CHECK(!sim.addObstacleSegment({1, 1}, {1, 1}));           // zero length
    CHECK(!sim.addObstacle({}));                              // empty
    CHECK(!sim.addObstacle({Vec2{0, 0}}));                    // single vertex
    CHECK(sim.obstacleVertexCount() == 0);
}

// ─── Determinism ────────────────────────────────────────────────────────────

TEST(avoid_deterministic_bit_identical) {
    auto run = [](std::vector<Vec2>& out) {
        AvoidanceSim sim;
        sim.addObstacleSegment({-1, 1}, {1, 1});
        AvoidanceAgentParams p;
        p.radius = 0.4f;
        p.maxSpeed = 4.0f;
        const int N = 8;
        std::vector<Vec2> goals((size_t)N);
        for (int i = 0; i < N; i++) {
            float ang = (float)i / N * 2.0f * (float)M_PI + 0.017f * (float)i;
            Vec2 start{6.0f * std::cos(ang), 6.0f * std::sin(ang)};
            goals[(size_t)i] = -1.0f * start;
            sim.addAgent(start, p);
        }
        const float dt = 1.0f / 60.0f;
        for (int step = 0; step < 600; step++) {
            for (int i = 0; i < N; i++)
                sim.setPrefVelocity(i, prefTowards(sim.position(i), goals[(size_t)i], p.maxSpeed));
            sim.step(dt);
        }
        out.clear();
        for (int i = 0; i < N; i++) out.push_back(sim.position(i));
    };

    std::vector<Vec2> run1, run2;
    run(run1);
    run(run2);
    CHECK(run1.size() == run2.size());
    CHECK(std::memcmp(run1.data(), run2.data(), run1.size() * sizeof(Vec2)) == 0);
}

// ─── World integration ──────────────────────────────────────────────────────

TEST(world_avoidance_swapped_goals_pass_and_arrive) {
    World world;
    world.setAvoidanceEnabled(true);

    Agent a, b;
    a.unit().id = 1;
    b.unit().id = 2;
    a.setPosition(-5, 0);
    b.setPosition(5, 0);
    a.setSpeed(4);
    b.setSpeed(4);
    a.setRadius(0.5f);
    b.setRadius(0.5f);
    world.addAgent(&a);
    world.addAgent(&b);
    a.setTarget(5, 0);
    b.setTarget(-5, 0);

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    for (int step = 0; step < 15 * 60; step++) {
        world.tick(dt);
        float d = std::hypot(a.x() - b.x(), a.z() - b.z());
        minDist = std::min(minDist, d);
        if (a.atTarget() && b.atTarget()) break;
    }
    CHECK(a.atTarget());
    CHECK(b.atTarget());
    CHECK(minDist >= 1.0f * 0.9f);  // sum of radii, small tolerance
}

TEST(world_avoidance_off_agents_pass_through) {
    // Regression guard: default (avoidance off) keeps the legacy behavior
    // where agents walk straight through each other.
    World world;
    Agent a, b;
    a.unit().id = 1;
    b.unit().id = 2;
    a.setPosition(-5, 0);
    b.setPosition(5, 0);
    a.setSpeed(4);
    b.setSpeed(4);
    a.setRadius(0.5f);
    b.setRadius(0.5f);
    world.addAgent(&a);
    world.addAgent(&b);
    a.setTarget(5, 0);
    b.setTarget(-5, 0);

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    for (int step = 0; step < 15 * 60; step++) {
        world.tick(dt);
        minDist = std::min(minDist, std::hypot(a.x() - b.x(), a.z() - b.z()));
        if (a.atTarget() && b.atTarget()) break;
    }
    CHECK(a.atTarget());
    CHECK(b.atTarget());
    CHECK(minDist < 0.5f);  // they overlapped mid-way
}

TEST(world_avoidance_respects_world_obstacle) {
    World world;
    world.setAvoidanceEnabled(true);
    AABB wall{0, 0, 2.0f, 0.25f};
    world.addObstacle(wall);

    Agent a;
    a.unit().id = 1;
    a.setPosition(-4, -2);
    a.setSpeed(4);
    a.setRadius(0.4f);
    world.addAgent(&a);
    a.setTarget(4, 2);  // straight line crosses the wall

    const float dt = 1.0f / 60.0f;
    float minClearance = 1e30f;
    for (int step = 0; step < 20 * 60; step++) {
        world.tick(dt);
        float dx = std::max(std::abs(a.x() - wall.cx) - wall.hw, 0.0f);
        float dz = std::max(std::abs(a.z() - wall.cz) - wall.hd, 0.0f);
        minClearance = std::min(minClearance, std::sqrt(dx * dx + dz * dz));
        if (a.atTarget()) break;
    }
    CHECK(a.atTarget());
    CHECK(minClearance >= a.radius() * 0.9f);
}

TEST(world_avoidance_crowd_corridor_makes_progress) {
    // 8 agents crossing through a corridor formed by two avoidance-only
    // walls; everyone must arrive with no hard interpenetration.
    World world;
    world.setAvoidanceEnabled(true);
    world.addAvoidanceObstacle({0, 3.0f, 6.0f, 0.5f});   // top wall
    world.addAvoidanceObstacle({0, -3.0f, 6.0f, 0.5f});  // bottom wall

    const int N = 8;
    std::vector<Agent> agents((size_t)N);
    for (int i = 0; i < N; i++) {
        Agent& ag = agents[(size_t)i];
        ag.unit().id = i + 1;
        ag.setSpeed(4);
        ag.setRadius(0.4f);
        // Half start on the left, half on the right; goals swapped, with a
        // small deterministic z-stagger so lanes aren't perfectly mirrored.
        float z = -1.5f + 1.0f * (float)(i % 4) + 0.05f * (float)i;
        if (i < 4) ag.setPosition(-9, z);
        else       ag.setPosition(9, z);
        world.addAgent(&ag);
    }
    for (int i = 0; i < N; i++) {
        float z = agents[(size_t)i].z();
        agents[(size_t)i].setTarget(i < 4 ? 9.0f : -9.0f, z);
    }

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    int arrived = 0;
    for (int step = 0; step < 40 * 60; step++) {
        world.tick(dt);
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                minDist = std::min(minDist,
                    std::hypot(agents[(size_t)i].x() - agents[(size_t)j].x(),
                               agents[(size_t)i].z() - agents[(size_t)j].z()));
        arrived = 0;
        for (int i = 0; i < N; i++)
            if (agents[(size_t)i].atTarget()) arrived++;
        if (arrived == N) break;
    }
    CHECK(arrived == N);
    CHECK(minDist >= (0.4f * 2.0f) * 0.9f);
}

TEST(world_avoidance_per_agent_opt_out) {
    // Opt-out agent keeps legacy straight-line movement; the opted-in one
    // still steers around it.
    World world;
    world.setAvoidanceEnabled(true);

    Agent blocker, mover;
    blocker.unit().id = 1;
    mover.unit().id = 2;
    blocker.setPosition(0, 0.01f);
    blocker.setRadius(0.5f);
    AgentAvoidance off;
    off.enabled = false;
    blocker.setAvoidance(off);
    mover.setPosition(-6, 0);
    mover.setSpeed(4);
    mover.setRadius(0.5f);
    world.addAgent(&blocker);
    world.addAgent(&mover);
    mover.setTarget(6, 0);

    const float dt = 1.0f / 60.0f;
    float minDist = 1e30f;
    for (int step = 0; step < 15 * 60; step++) {
        world.tick(dt);
        minDist = std::min(minDist,
            std::hypot(mover.x() - blocker.x(), mover.z() - blocker.z()));
        if (mover.atTarget()) break;
    }
    CHECK(mover.atTarget());
    CHECK(minDist >= 1.0f * 0.9f);
    CHECK_NEAR(blocker.x(), 0.0f, 1e-4f);   // never displaced
    CHECK_NEAR(blocker.z(), 0.01f, 1e-4f);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("brogameagent avoidance tests\n");
    printf("============================\n");

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
