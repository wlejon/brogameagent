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

// ─── Unit / World / Action / Observation ────────────────────────────────────

TEST(unit_cooldowns_tick_down) {
    Unit u;
    u.attackCooldown = 1.0f;
    u.abilityCooldowns[0] = 0.5f;
    u.tickCooldowns(0.3f);
    CHECK_NEAR(u.attackCooldown, 0.7f, 1e-4f);
    CHECK_NEAR(u.abilityCooldowns[0], 0.2f, 1e-4f);
    u.tickCooldowns(10.0f);
    CHECK_NEAR(u.attackCooldown, 0.0f, 1e-4f);
    CHECK_NEAR(u.abilityCooldowns[0], 0.0f, 1e-4f);
}

TEST(agent_applyAction_moves_in_local_frame) {
    // yaw=0 means facing -Z. Local +Z = back, local -Z = forward.
    // Pushing moveZ=-1 (forward) should move agent in -Z world direction.
    Agent agent;
    agent.setPosition(0, 0);
    agent.unit().moveSpeed = 10.0f;

    AgentAction act;
    act.moveZ = -1.0f;
    for (int i = 0; i < 60; i++) agent.applyAction(act, 1.0f / 60.0f);

    CHECK(agent.z() < -5.0f); // moved forward (-Z)
    CHECK_NEAR(agent.x(), 0.0f, 0.5f);
}

TEST(agent_maxAccel_clamps_velocity_change) {
    Agent agent;
    agent.setPosition(0, 0);
    agent.unit().moveSpeed = 10.0f;
    agent.setMaxAccel(2.0f); // 2 u/s^2

    AgentAction act;
    act.moveZ = -1.0f;
    agent.applyAction(act, 1.0f); // 1 second, accel cap gives max |dv|=2
    Vec2 v = agent.velocity();
    float speed = std::sqrt(v.x * v.x + v.z * v.z);
    CHECK(speed <= 2.01f); // clamped to maxAccel * dt
}

TEST(agent_maxTurnRate_clamps_yaw) {
    Agent agent;
    agent.setPosition(0, 0);
    agent.unit().moveSpeed = 10.0f;
    agent.setMaxTurnRate(static_cast<float>(M_PI / 4)); // 45 deg/sec

    // Push to the right (+X local) — needs yaw to rotate by pi/2.
    AgentAction act;
    act.moveX = 1.0f;
    agent.applyAction(act, 0.5f); // half a second, can only turn 22.5 deg

    float turned = std::abs(agent.yaw());
    CHECK(turned <= static_cast<float>(M_PI / 8) + 0.01f);
}

TEST(world_enemies_and_allies) {
    World world;
    Agent a1, a2, a3, a4;
    a1.unit().id = 1; a1.unit().teamId = 0; a1.setPosition(0, 0);
    a2.unit().id = 2; a2.unit().teamId = 0; a2.setPosition(1, 0);
    a3.unit().id = 3; a3.unit().teamId = 1; a3.setPosition(5, 0);
    a4.unit().id = 4; a4.unit().teamId = 1; a4.setPosition(2, 0);

    world.addAgent(&a1);
    world.addAgent(&a2);
    world.addAgent(&a3);
    world.addAgent(&a4);

    auto enemies = world.enemiesOf(a1);
    CHECK(enemies.size() == 2);

    auto allies = world.alliesOf(a1);
    CHECK(allies.size() == 1);
    CHECK(allies[0]->unit().id == 2);

    Agent* nearest = world.nearestEnemy(a1);
    CHECK(nearest != nullptr);
    CHECK(nearest->unit().id == 4); // a4 at dist 2 beats a3 at dist 5

    auto inRange = world.enemiesInRange(a1, 3.0f);
    CHECK(inRange.size() == 1);
    CHECK(inRange[0]->unit().id == 4);
}

TEST(world_skips_dead_agents) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0; a.setPosition(0, 0);
    b.unit().id = 2; b.unit().teamId = 1; b.setPosition(1, 0);
    world.addAgent(&a);
    world.addAgent(&b);
    b.unit().hp = 0;
    CHECK(world.nearestEnemy(a) == nullptr);
    CHECK(world.enemiesOf(a).empty());
}

TEST(world_tick_advances_cooldowns) {
    World world;
    Agent a;
    a.unit().attackCooldown = 2.0f;
    world.addAgent(&a);
    world.tick(0.5f);
    CHECK_NEAR(a.unit().attackCooldown, 1.5f, 1e-4f);
}

TEST(world_findById) {
    World world;
    Agent a, b;
    a.unit().id = 42;
    b.unit().id = 99;
    world.addAgent(&a);
    world.addAgent(&b);
    CHECK(world.findById(42) == &a);
    CHECK(world.findById(99) == &b);
    CHECK(world.findById(7) == nullptr);
}

TEST(observation_total_size_nonzero) {
    CHECK(observation::TOTAL > 0);
    // SELF(8) + 5*6 + 4*5 = 58
    CHECK(observation::TOTAL == 58);
}

TEST(observation_self_block_populated) {
    World world;
    Agent self;
    self.unit().teamId = 0;
    self.unit().hp = 50; self.unit().maxHp = 100;
    self.unit().mana = 20; self.unit().maxMana = 100;
    self.unit().attackCooldown = 5.0f;
    self.setPosition(0, 0);
    world.addAgent(&self);

    float obs[observation::TOTAL];
    observation::build(self, world, obs);

    CHECK_NEAR(obs[0], 0.5f, 1e-4f);  // hp ratio
    CHECK_NEAR(obs[1], 0.2f, 1e-4f);  // mana ratio
    CHECK_NEAR(obs[2], 0.5f, 1e-4f);  // attack cd / 10
}

TEST(observation_enemies_sorted_and_local_frame) {
    World world;
    Agent self;
    self.unit().id = 1; self.unit().teamId = 0;
    self.unit().attackRange = 3.0f;
    self.setPosition(0, 0);

    Agent far_enemy, near_enemy;
    far_enemy.unit().id = 2; far_enemy.unit().teamId = 1;
    far_enemy.unit().hp = 100; far_enemy.unit().maxHp = 100;
    far_enemy.setPosition(10, 0);

    near_enemy.unit().id = 3; near_enemy.unit().teamId = 1;
    near_enemy.unit().hp = 50; near_enemy.unit().maxHp = 100;
    near_enemy.setPosition(2, 0);

    world.addAgent(&self);
    world.addAgent(&far_enemy);
    world.addAgent(&near_enemy);

    float obs[observation::TOTAL];
    observation::build(self, world, obs);

    int base0 = observation::SELF_FEATURES; // first enemy slot
    int base1 = observation::SELF_FEATURES + observation::ENEMY_FEATURES;

    // Nearest-first: slot 0 is the near_enemy
    CHECK_NEAR(obs[base0 + 0], 1.0f, 1e-4f);        // valid
    CHECK_NEAR(obs[base0 + 3], 2.0f / 50.0f, 1e-4f); // distance / OBS_RANGE
    CHECK_NEAR(obs[base0 + 4], 0.5f, 1e-4f);         // hp ratio
    CHECK_NEAR(obs[base0 + 5], 1.0f, 1e-4f);         // in attack range (2 <= 3)

    // Slot 1 is the far_enemy; out of attack range.
    CHECK_NEAR(obs[base1 + 0], 1.0f, 1e-4f);
    CHECK_NEAR(obs[base1 + 5], 0.0f, 1e-4f);
}

TEST(observation_empty_slots_are_zero) {
    World world;
    Agent self;
    self.unit().teamId = 0;
    world.addAgent(&self);

    float obs[observation::TOTAL];
    observation::build(self, world, obs);

    // Every enemy valid flag should be 0.
    for (int k = 0; k < observation::K_ENEMIES; k++) {
        int base = observation::SELF_FEATURES + k * observation::ENEMY_FEATURES;
        CHECK_NEAR(obs[base + 0], 0.0f, 1e-6f);
    }
}

TEST(observation_local_frame_rotates_with_yaw) {
    // Place enemy at world +X (east). With yaw=0 (facing -Z/north), enemy is
    // to the right, so local x > 0, local z ~ 0. After turning to face +X,
    // the enemy should be straight ahead, meaning local x ~ 0, local z < 0.
    World world;
    Agent self;
    self.unit().id = 1; self.unit().teamId = 0;
    self.setPosition(0, 0);
    Agent enemy;
    enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.setPosition(10, 0);
    world.addAgent(&self);
    world.addAgent(&enemy);

    float obs[observation::TOTAL];

    // yaw = 0: enemy to the right (+X local)
    self.unit().moveSpeed = 10.0f;
    // We need to set yaw; the only public route is applyAction or scripted
    // movement. Use applyAction with moveX=1 briefly then zero.
    // Faster: run a quick applyAction that rotates facing to +X.
    self.setMaxTurnRate(0.0f); // unlimited
    AgentAction act;
    act.moveX = 1.0f;
    self.applyAction(act, 0.001f); // sets yaw toward +X
    self.setPosition(0, 0);        // reset position drift
    // velocity may be nonzero but doesn't matter for observation geometry

    observation::build(self, world, obs);
    int base = observation::SELF_FEATURES;
    // Enemy is now "straight ahead" in local frame → -Z_local big, X_local ~ 0
    CHECK(obs[base + 2] < -0.1f);              // lz/OBS_RANGE negative
    CHECK(std::abs(obs[base + 1]) < 0.05f);    // lx ~ 0
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
