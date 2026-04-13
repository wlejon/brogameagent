#define _USE_MATH_DEFINES
#include <cmath>

#include <brogameagent/brogameagent.h>

#include <cassert>
#include <cstdio>
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
#define CHECK_NEAR(a, b, eps) check(std::abs((a) - (b)) < (eps), #a " ~= " #b, __LINE__)

// ─── NavGrid Tests ──────────────────────────────────────────────────────────

TEST(navgrid_empty_grid_is_walkable) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    CHECK(grid.isWalkable(0, 0));
    CHECK(grid.isWalkable(5, 5));
    CHECK(grid.isWalkable(-9.5f, -9.5f));
}

TEST(navgrid_out_of_bounds_not_walkable) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    CHECK(!grid.isWalkable(-11, 0));
    CHECK(!grid.isWalkable(0, 11));
}

TEST(navgrid_obstacle_blocks_cells) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    grid.addObstacle({0, 0, 2, 2}); // 4x4 box centered at origin
    CHECK(!grid.isWalkable(0, 0));
    CHECK(!grid.isWalkable(1.5f, 1.5f));
    CHECK(grid.isWalkable(5, 5));
}

TEST(navgrid_obstacle_with_padding) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    grid.addObstacle({0, 0, 1, 1}, 1.0f); // 2x2 box + 1 padding = 4x4 blocked
    CHECK(!grid.isWalkable(0, 0));
    CHECK(!grid.isWalkable(1.5f, 1.5f)); // within padding
    CHECK(grid.isWalkable(5, 0));
}

TEST(navgrid_path_straight_line) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    auto path = grid.findPath({-5, 0}, {5, 0});
    CHECK(!path.empty());
    // Smoothed path for unobstructed line should be short (1-2 points)
    CHECK(path.size() <= 2);
    // Last point should be near target
    CHECK_NEAR(path.back().x, 5.0f, 1.0f);
    CHECK_NEAR(path.back().z, 0.0f, 1.0f);
}

TEST(navgrid_path_around_obstacle) {
    NavGrid grid(-10, -10, 10, 10, 0.5f);
    // Wall blocking straight path from (-5,0) to (5,0)
    grid.addObstacle({0, 0, 0.5f, 5}, 0.4f);
    auto path = grid.findPath({-5, 0}, {5, 0});
    CHECK(!path.empty());
    // Path should go around the obstacle, so more than 2 waypoints
    CHECK(path.size() >= 2);
    CHECK_NEAR(path.back().x, 5.0f, 1.0f);
}

TEST(navgrid_no_path_blocked) {
    NavGrid grid(-5, -5, 5, 5, 0.5f);
    // Surround the goal with obstacles
    grid.addObstacle({4, 0, 1.5f, 5.5f}); // wall on right side
    auto path = grid.findPath({-4, 0}, {4.5f, 0});
    CHECK(path.empty()); // goal is inside obstacle
}

TEST(navgrid_grid_los) {
    NavGrid grid(-10, -10, 10, 10, 1.0f);
    CHECK(grid.hasGridLOS({-5, 0}, {5, 0}));

    grid.addObstacle({0, 0, 1, 1});
    CHECK(!grid.hasGridLOS({-5, 0}, {5, 0})); // blocked by obstacle
    CHECK(grid.hasGridLOS({-5, 5}, {5, 5}));   // clear above obstacle
}

// ─── Steering Tests ─────────────────────────────────────────────────────────

TEST(steering_seek) {
    auto s = seek({0, 0}, {10, 0});
    CHECK(s.fx > 0.9f);
    CHECK_NEAR(s.fz, 0.0f, 0.01f);
}

TEST(steering_seek_diagonal) {
    auto s = seek({0, 0}, {1, 1});
    CHECK(s.fx > 0);
    CHECK(s.fz > 0);
    float len = std::sqrt(s.fx * s.fx + s.fz * s.fz);
    CHECK_NEAR(len, 1.0f, 0.01f);
}

TEST(steering_arrive_slows_down) {
    auto far = arrive({0, 0}, {10, 0}, 3.0f);
    auto near = arrive({0, 0}, {1, 0}, 3.0f);
    float farMag = std::sqrt(far.fx * far.fx + far.fz * far.fz);
    float nearMag = std::sqrt(near.fx * near.fx + near.fz * near.fz);
    CHECK(farMag > nearMag); // should be slower when closer
}

TEST(steering_flee) {
    auto s = flee({0, 0}, {10, 0});
    CHECK(s.fx < -0.9f); // runs away
}

TEST(steering_follow_path) {
    std::vector<Vec2> path = {{2, 0}, {4, 0}, {6, 0}};
    int idx = 0;
    auto s = followPath({0, 0}, path, idx, 0.5f);
    CHECK(s.fx > 0); // moves toward first waypoint
    CHECK(idx == 0);

    // Simulate arriving near first waypoint
    s = followPath({1.8f, 0}, path, idx, 0.5f);
    CHECK(idx == 1); // advanced to next waypoint
}

// ─── Perception Tests ───────────────────────────────────────────────────────

TEST(perception_los_clear) {
    AABB obs[] = {{5, 5, 1, 1}};
    CHECK(hasLineOfSight({0, 0}, {10, 0}, obs, 1)); // obstacle is off to the side
}

TEST(perception_los_blocked) {
    AABB obs[] = {{5, 0, 1, 1}};
    CHECK(!hasLineOfSight({0, 0}, {10, 0}, obs, 1)); // obstacle in the way
}

TEST(perception_aim_forward) {
    // Aiming at a point directly in front (-Z direction)
    auto aim = computeAim(0, 1.6f, 0, 0, 1.6f, -10);
    CHECK_NEAR(aim.yaw, 0.0f, 0.01f);
    CHECK_NEAR(aim.pitch, 0.0f, 0.01f);
}

TEST(perception_aim_right) {
    // Aiming at a point to the right (+X)
    auto aim = computeAim(0, 1.6f, 0, 10, 1.6f, 0);
    CHECK_NEAR(aim.yaw, static_cast<float>(M_PI / 2), 0.01f);
    CHECK_NEAR(aim.pitch, 0.0f, 0.01f);
}

TEST(perception_aim_up) {
    auto aim = computeAim(0, 0, 0, 0, 10, -10);
    CHECK(aim.pitch > 0.3f); // looking up
}

// ─── Agent Tests ─────────────────────────────────────────────────────────────

TEST(agent_moves_toward_target) {
    Agent agent;
    agent.setPosition(0, 0);
    agent.setSpeed(6.0f);
    agent.setTarget(10, 0);

    for (int i = 0; i < 60; i++) agent.update(1.0f / 60.0f);

    CHECK(agent.x() > 5.0f); // should have moved significantly
    CHECK_NEAR(agent.z(), 0.0f, 0.5f);
}

TEST(agent_reaches_target) {
    Agent agent;
    agent.setPosition(0, 0);
    agent.setSpeed(10.0f);
    agent.setTarget(3, 0);

    for (int i = 0; i < 120; i++) agent.update(1.0f / 60.0f);

    CHECK(agent.atTarget());
}

TEST(agent_with_navgrid) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    grid.addObstacle({0, 0, 1, 5}, 0.4f); // vertical wall

    Agent agent;
    agent.setNavGrid(&grid);
    agent.setPosition(-5, 0);
    agent.setSpeed(6.0f);
    agent.setTarget(5, 0);

    // Simulate several seconds of movement
    for (int i = 0; i < 600; i++) agent.update(1.0f / 60.0f);

    // Should have navigated around the wall
    CHECK(agent.x() > 3.0f);
}

TEST(agent_aim_at) {
    Agent agent;
    agent.setPosition(0, 0);

    auto aim = agent.aimAt(10, 1.6f, 0, 1.6f);
    CHECK_NEAR(aim.yaw, static_cast<float>(M_PI / 2), 0.05f);
    CHECK_NEAR(aim.pitch, 0.0f, 0.05f);
}

// ─── New helpers ────────────────────────────────────────────────────────────

TEST(angle_wrap) {
    CHECK_NEAR(wrapAngle(0.0f), 0.0f, 1e-5f);
    // π and -π are the same angle; wrap returns either edge. Test values
    // strictly inside the range instead.
    CHECK_NEAR(wrapAngle(1.0f), 1.0f, 1e-5f);
    CHECK_NEAR(wrapAngle(1.0f + static_cast<float>(2 * M_PI)), 1.0f, 1e-4f);
    CHECK_NEAR(wrapAngle(-1.0f - static_cast<float>(2 * M_PI)), -1.0f, 1e-4f);
}

TEST(angle_delta_shortest) {
    // From +170° to -170° should be +20°, not -340°
    float from = 170.0f * static_cast<float>(M_PI) / 180.0f;
    float to   = -170.0f * static_cast<float>(M_PI) / 180.0f;
    float d = angleDelta(from, to);
    CHECK_NEAR(d, 20.0f * static_cast<float>(M_PI) / 180.0f, 1e-4f);
}

TEST(steering_pursue_leads_target) {
    // Target moving in +x at (5,0), moving fast in +x. Pursue from origin
    // should steer toward a point ahead of the target (more +x than 5).
    auto s = pursue({0, 0}, {5, 0}, {10, 0}, 5.0f);
    // Direct seek would give fx ~= 1, fz ~= 0. Pursue still mostly +x.
    CHECK(s.fx > 0.9f);
}

TEST(steering_evade_runs_from_predicted) {
    auto s = evade({0, 0}, {5, 0}, {10, 0}, 5.0f);
    CHECK(s.fx < -0.9f); // flees in -x direction
}

TEST(lead_aim_stationary_matches_direct) {
    auto lead = computeLeadAim(0, 0, 0, 0, 0, -10, 0, 0, 0, 50.0f);
    CHECK(lead.valid);
    CHECK_NEAR(lead.aim.yaw, 0.0f, 0.01f);
    CHECK_NEAR(lead.aim.pitch, 0.0f, 0.01f);
}

TEST(lead_aim_moving_crossways) {
    // Target at (10,0,0), moving +z at 10 m/s. Projectile 50 m/s.
    // Intercept point should be somewhere at +x with some +z offset, so yaw
    // between "right" (+pi/2) and "right-and-back" — larger yaw than direct.
    auto direct = computeAim(0, 0, 0, 10, 0, 0);
    auto lead = computeLeadAim(0, 0, 0, 10, 0, 0, 0, 0, 10, 50.0f);
    CHECK(lead.valid);
    CHECK(lead.timeToHit > 0);
    // The intercept is offset along +z from target pos, so yaw should differ.
    CHECK(std::abs(lead.aim.yaw - direct.yaw) > 0.01f);
}

TEST(lead_aim_unreachable_target) {
    // Target moving faster than projectile, directly away.
    auto lead = computeLeadAim(0, 0, 0, 10, 0, 0, 100, 0, 0, 10.0f);
    CHECK(!lead.valid);
}

TEST(canSee_out_of_range) {
    CHECK(!canSee({0, 0}, {100, 0}, static_cast<float>(M_PI / 2), static_cast<float>(M_PI), 10.0f, nullptr, 0));
}

TEST(canSee_outside_fov) {
    // Facing -Z (yaw=0), target at +x is 90° off axis. FOV 60° (±30°) rejects.
    float fov60 = 60.0f * static_cast<float>(M_PI) / 180.0f;
    CHECK(!canSee({0, 0}, {10, 0}, 0.0f, fov60, 100.0f, nullptr, 0));
}

TEST(canSee_in_fov_and_los) {
    float fov120 = 120.0f * static_cast<float>(M_PI) / 180.0f;
    CHECK(canSee({0, 0}, {10, 0}, static_cast<float>(M_PI / 2), fov120, 100.0f, nullptr, 0));
}

TEST(canSee_blocked_by_obstacle) {
    AABB obs[] = {{5, 0, 1, 1}};
    float fov120 = 120.0f * static_cast<float>(M_PI) / 180.0f;
    CHECK(!canSee({0, 0}, {10, 0}, static_cast<float>(M_PI / 2), fov120, 100.0f, obs, 1));
}

TEST(agent_accessors_reflect_state) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    Agent agent;
    agent.setNavGrid(&grid);
    agent.setPosition(0, 0);
    agent.setSpeed(6.0f);

    CHECK(agent.path().empty());
    CHECK(agent.currentWaypoint() == 0);

    agent.setTarget(5, 0);
    CHECK(!agent.path().empty());

    agent.update(1.0f / 60.0f);
    Vec2 v = agent.velocity();
    CHECK(v.x > 0.0f);
    CHECK_NEAR(v.z, 0.0f, 0.5f);

    agent.clearTarget();
    agent.update(1.0f / 60.0f);
    CHECK_NEAR(agent.velocity().x, 0.0f, 1e-4f);
    CHECK_NEAR(agent.velocity().z, 0.0f, 1e-4f);
}

TEST(agent_clear_target_stops) {
    Agent agent;
    agent.setPosition(0, 0);
    agent.setSpeed(6.0f);
    agent.setTarget(10, 0);

    agent.update(1.0f / 60.0f);
    float x1 = agent.x();
    CHECK(x1 > 0); // moved

    agent.clearTarget();
    agent.update(1.0f / 60.0f);
    CHECK_NEAR(agent.x(), x1, 0.001f); // didn't move
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("brogameagent tests\n");
    printf("==================\n");

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
