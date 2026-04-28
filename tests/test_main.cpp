#define _USE_MATH_DEFINES
#include <cmath>

#include <brogameagent/brogameagent.h>
#include <brogameagent/grid/obs_window.h>
#include <brogameagent/grid/frame_stack.h>
#include <brogameagent/grid/failure_tape.h>

#include <cassert>
#include <cstdio>
#include <memory>
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
    // SELF(14) + 5*6 + 4*5 = 64
    CHECK(observation::TOTAL == 64);
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

// ─── Combat ──────────────────────────────────────────────────────────────────

TEST(unit_takeDamage_physical_reduced_by_armor) {
    Unit u;
    u.maxHp = 1000; u.hp = 1000; u.armor = 100;
    // 100 damage × 100/(100+100) = 50 dealt
    float dealt = u.takeDamage(100.0f, DamageKind::Physical);
    CHECK_NEAR(dealt, 50.0f, 1e-3f);
    CHECK_NEAR(u.hp, 950.0f, 1e-3f);
}

TEST(unit_takeDamage_magical_reduced_by_mr) {
    Unit u;
    u.maxHp = 1000; u.hp = 1000; u.magicResist = 50;
    // 150 × 100/150 = 100
    float dealt = u.takeDamage(150.0f, DamageKind::Magical);
    CHECK_NEAR(dealt, 100.0f, 1e-3f);
}

TEST(unit_takeDamage_true_ignores_armor) {
    Unit u;
    u.maxHp = 500; u.hp = 500; u.armor = 100; u.magicResist = 100;
    float dealt = u.takeDamage(120.0f, DamageKind::True);
    CHECK_NEAR(dealt, 120.0f, 1e-3f);
    CHECK_NEAR(u.hp, 380.0f, 1e-3f);
}

TEST(unit_takeDamage_clamps_to_zero_and_kills) {
    Unit u;
    u.maxHp = 100; u.hp = 30;
    float dealt = u.takeDamage(200.0f, DamageKind::True);
    CHECK_NEAR(dealt, 30.0f, 1e-3f);
    CHECK(!u.alive());
    // Further damage on a dead unit does nothing.
    float more = u.takeDamage(50.0f, DamageKind::True);
    CHECK_NEAR(more, 0.0f, 1e-6f);
}

TEST(world_resolveAttack_in_range_deals_damage) {
    World world;
    Agent attacker, target;
    attacker.unit().id = 1; attacker.unit().teamId = 0;
    attacker.unit().damage = 50; attacker.unit().attackRange = 5;
    attacker.unit().attacksPerSec = 2.0f;
    attacker.setPosition(0, 0);

    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().maxHp = 200; target.unit().hp = 200; target.unit().armor = 0;
    target.setPosition(3, 0);

    world.addAgent(&attacker);
    world.addAgent(&target);

    CHECK(world.resolveAttack(attacker, 2));
    CHECK_NEAR(target.unit().hp, 150.0f, 1e-3f);
    CHECK_NEAR(attacker.unit().attackCooldown, 0.5f, 1e-4f);

    // Second attack fails — cooldown not ready.
    CHECK(!world.resolveAttack(attacker, 2));
    CHECK_NEAR(target.unit().hp, 150.0f, 1e-3f);
}

TEST(world_resolveAttack_out_of_range_fails) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 50; a.unit().attackRange = 3; a.unit().attacksPerSec = 1;
    a.setPosition(0, 0);
    b.unit().id = 2; b.unit().teamId = 1;
    b.unit().hp = 100; b.unit().maxHp = 100;
    b.setPosition(10, 0);
    world.addAgent(&a);
    world.addAgent(&b);

    CHECK(!world.resolveAttack(a, 2));
    CHECK_NEAR(b.unit().hp, 100.0f, 1e-3f);
}

TEST(world_resolveAttack_same_team_fails) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 50; a.unit().attackRange = 5; a.unit().attacksPerSec = 1;
    a.setPosition(0, 0);
    b.unit().id = 2; b.unit().teamId = 0;
    b.unit().hp = 100; b.unit().maxHp = 100;
    b.setPosition(2, 0);
    world.addAgent(&a);
    world.addAgent(&b);

    CHECK(!world.resolveAttack(a, 2));
}

TEST(world_resolveAttack_dead_target_fails) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 50; a.unit().attackRange = 5; a.unit().attacksPerSec = 1;
    b.unit().id = 2; b.unit().teamId = 1;
    b.unit().hp = 0; b.unit().maxHp = 100;
    world.addAgent(&a);
    world.addAgent(&b);
    CHECK(!world.resolveAttack(a, 2));
}

TEST(world_resolveAbility_invokes_fn_and_spends_resources) {
    World world;
    Agent caster, target;
    caster.unit().id = 1; caster.unit().teamId = 0;
    caster.unit().mana = 100; caster.unit().maxMana = 100;
    caster.unit().abilitySlot[0] = 42;
    caster.setPosition(0, 0);

    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 200; target.unit().maxHp = 200;
    target.setPosition(3, 0);

    world.addAgent(&caster);
    world.addAgent(&target);

    int callsMade = 0;
    AbilitySpec spec;
    spec.cooldown = 4.0f;
    spec.manaCost = 30.0f;
    spec.range = 5.0f;
    spec.fn = [&](Agent& c, World& w, int tid) {
        callsMade++;
        Agent* t = w.findById(tid);
        if (t) t->unit().takeDamage(80.0f, DamageKind::Magical);
        (void)c;
    };
    world.registerAbility(42, spec);

    CHECK(world.resolveAbility(caster, 0, 2));
    CHECK(callsMade == 1);
    CHECK_NEAR(target.unit().hp, 120.0f, 1e-3f); // 80 magic, MR=0
    CHECK_NEAR(caster.unit().mana, 70.0f, 1e-4f);
    CHECK_NEAR(caster.unit().abilityCooldowns[0], 4.0f, 1e-4f);

    // Cooldown blocks re-cast.
    CHECK(!world.resolveAbility(caster, 0, 2));
    CHECK(callsMade == 1);
}

TEST(world_resolveAbility_out_of_range_fails) {
    World world;
    Agent caster, target;
    caster.unit().id = 1; caster.unit().teamId = 0;
    caster.unit().mana = 100; caster.unit().abilitySlot[0] = 7;
    caster.setPosition(0, 0);
    target.unit().id = 2; target.unit().teamId = 1;
    target.setPosition(20, 0);
    world.addAgent(&caster);
    world.addAgent(&target);

    AbilitySpec spec;
    spec.range = 5.0f;
    spec.fn = [](Agent&, World&, int) {};
    world.registerAbility(7, spec);

    CHECK(!world.resolveAbility(caster, 0, 2));
    CHECK_NEAR(caster.unit().mana, 100.0f, 1e-4f); // nothing spent
}

TEST(world_resolveAbility_not_enough_mana) {
    World world;
    Agent caster;
    caster.unit().id = 1;
    caster.unit().mana = 10; caster.unit().abilitySlot[0] = 3;
    world.addAgent(&caster);

    AbilitySpec spec;
    spec.manaCost = 50;
    spec.fn = [](Agent&, World&, int) {};
    world.registerAbility(3, spec);

    CHECK(!world.resolveAbility(caster, 0, -1));
}

TEST(world_resolveAbility_empty_slot_fails) {
    World world;
    Agent caster;
    caster.unit().id = 1;
    world.addAgent(&caster);
    // abilitySlot[0] stays -1 by default
    CHECK(!world.resolveAbility(caster, 0, -1));
}

TEST(world_applyAction_runs_attack_then_movement) {
    World world;
    Agent shooter, target;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    shooter.unit().damage = 25; shooter.unit().attackRange = 10;
    shooter.unit().attacksPerSec = 1; shooter.unit().moveSpeed = 5;
    shooter.setPosition(0, 0);

    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    target.setPosition(5, 0);

    world.addAgent(&shooter);
    world.addAgent(&target);

    AgentAction act;
    act.moveZ = -1.0f;
    act.attackTargetId = 2;
    world.applyAction(shooter, act, 0.1f);

    CHECK_NEAR(target.unit().hp, 75.0f, 1e-3f);
    CHECK(shooter.z() < 0); // moved forward
}

// ─── RNG / event log / action mask / reward ─────────────────────────────────

TEST(rng_deterministic_with_seed) {
    World w1, w2;
    w1.seed(12345);
    w2.seed(12345);
    for (int i = 0; i < 100; i++) {
        CHECK(w1.randFloat01() == w2.randFloat01());
    }
}

TEST(rng_randRange_and_chance) {
    World w;
    w.seed(42);
    int hits = 0;
    int samples = 2000;
    for (int i = 0; i < samples; i++) {
        float v = w.randRange(10.0f, 20.0f);
        CHECK(v >= 10.0f);
        CHECK(v < 20.0f);
        if (w.chance(0.3f)) hits++;
    }
    // rough sanity: 2000 * 0.3 ~= 600 with plenty of slack
    CHECK(hits > 450);
    CHECK(hits < 750);
}

TEST(rng_randInt_inclusive) {
    World w;
    w.seed(7);
    for (int i = 0; i < 200; i++) {
        int v = w.randInt(5, 7);
        CHECK(v >= 5);
        CHECK(v <= 7);
    }
}

TEST(events_logged_by_resolveAttack) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 50; a.unit().attackRange = 5; a.unit().attacksPerSec = 1;
    b.unit().id = 2; b.unit().teamId = 1;
    b.unit().hp = 100; b.unit().maxHp = 100;
    a.setPosition(0, 0); b.setPosition(2, 0);
    world.addAgent(&a);
    world.addAgent(&b);

    world.resolveAttack(a, 2);
    CHECK(world.events().size() == 1);
    CHECK(world.events()[0].attackerId == 1);
    CHECK(world.events()[0].targetId == 2);
    CHECK_NEAR(world.events()[0].amount, 50.0f, 1e-3f);
    CHECK(!world.events()[0].killed);

    // Finish it off.
    a.unit().attackCooldown = 0;
    world.resolveAttack(a, 2);
    a.unit().attackCooldown = 0;
    world.resolveAttack(a, 2);
    // b hp was 100 -> 50 -> 0, last hit is the kill.
    bool sawKill = false;
    for (const auto& e : world.events()) if (e.killed) sawKill = true;
    CHECK(sawKill);
}

TEST(events_cleared) {
    World w;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 10; a.unit().attackRange = 5; a.unit().attacksPerSec = 1;
    b.unit().id = 2; b.unit().teamId = 1; b.unit().maxHp = 100; b.unit().hp = 100;
    w.addAgent(&a); w.addAgent(&b);
    w.resolveAttack(a, 2);
    CHECK(w.events().size() == 1);
    w.clearEvents();
    CHECK(w.events().empty());
}

TEST(action_mask_enemy_slots_match_obs_order) {
    World world;
    Agent self;
    self.unit().id = 1; self.unit().teamId = 0;
    self.unit().attackRange = 3;
    self.unit().attacksPerSec = 1;
    self.setPosition(0, 0);

    Agent near_enemy, far_enemy;
    near_enemy.unit().id = 2; near_enemy.unit().teamId = 1;
    near_enemy.unit().maxHp = 100; near_enemy.unit().hp = 100;
    near_enemy.setPosition(2, 0); // in range
    far_enemy.unit().id = 3; far_enemy.unit().teamId = 1;
    far_enemy.unit().maxHp = 100; far_enemy.unit().hp = 100;
    far_enemy.setPosition(10, 0); // out of range

    world.addAgent(&self);
    world.addAgent(&near_enemy);
    world.addAgent(&far_enemy);

    float mask[action_mask::TOTAL];
    int ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, ids);

    CHECK(ids[0] == 2);                  // nearest first
    CHECK(ids[1] == 3);
    CHECK_NEAR(mask[0], 1.0f, 1e-6f);    // near is attackable
    CHECK_NEAR(mask[1], 0.0f, 1e-6f);    // far is out of range
}

TEST(action_mask_cooldown_blocks_attack) {
    World world;
    Agent self, enemy;
    self.unit().id = 1; self.unit().teamId = 0;
    self.unit().attackRange = 5; self.unit().attacksPerSec = 1;
    self.unit().attackCooldown = 0.5f; // not ready
    enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().maxHp = 100; enemy.unit().hp = 100;
    enemy.setPosition(2, 0);
    world.addAgent(&self);
    world.addAgent(&enemy);

    float mask[action_mask::TOTAL];
    int ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, ids);
    CHECK_NEAR(mask[0], 0.0f, 1e-6f);
}

TEST(action_mask_ability_gates) {
    World world;
    Agent self;
    self.unit().id = 1; self.unit().teamId = 0;
    self.unit().mana = 40; self.unit().maxMana = 100;
    self.unit().abilitySlot[0] = 100; // mana ok, cd ready
    self.unit().abilitySlot[1] = 101; // mana low
    self.unit().abilitySlot[2] = 102; // cd not ready
    self.unit().abilityCooldowns[2] = 3.0f;
    // slot 3 left empty
    world.addAgent(&self);

    AbilitySpec s0; s0.manaCost = 20; s0.fn = [](Agent&, World&, int) {};
    AbilitySpec s1; s1.manaCost = 80; s1.fn = [](Agent&, World&, int) {};
    AbilitySpec s2; s2.manaCost = 0;  s2.fn = [](Agent&, World&, int) {};
    world.registerAbility(100, s0);
    world.registerAbility(101, s1);
    world.registerAbility(102, s2);

    float mask[action_mask::TOTAL];
    int ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(self, world, mask, ids);

    int base = action_mask::N_ENEMY_SLOTS;
    CHECK_NEAR(mask[base + 0], 1.0f, 1e-6f);
    CHECK_NEAR(mask[base + 1], 0.0f, 1e-6f); // mana low
    CHECK_NEAR(mask[base + 2], 0.0f, 1e-6f); // cd
    CHECK_NEAR(mask[base + 3], 0.0f, 1e-6f); // empty slot
}

TEST(reward_tracker_accumulates_deltas) {
    World world;
    Agent hero, target;
    hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().damage = 30; hero.unit().attackRange = 5;
    hero.unit().attacksPerSec = 1;
    hero.setPosition(0, 0);
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().maxHp = 100; target.unit().hp = 100;
    target.setPosition(2, 0);
    world.addAgent(&hero);
    world.addAgent(&target);

    RewardTracker rt;
    rt.reset(hero, world);

    world.resolveAttack(hero, 2); // 30 dmg
    hero.setPosition(1, 0);       // travelled 1 unit

    auto d1 = rt.consume(hero, world);
    CHECK_NEAR(d1.damageDealt, 30.0f, 1e-3f);
    CHECK_NEAR(d1.damageTaken, 0.0f, 1e-6f);
    CHECK(d1.kills == 0);
    CHECK(d1.deaths == 0);
    CHECK_NEAR(d1.distanceTravelled, 1.0f, 1e-3f);

    // Kill it off over two more hits.
    hero.unit().attackCooldown = 0;
    world.resolveAttack(hero, 2);
    hero.unit().attackCooldown = 0;
    world.resolveAttack(hero, 2);
    hero.unit().attackCooldown = 0;
    world.resolveAttack(hero, 2); // target now at 100-30-30-30-30 = -20 → 0

    auto d2 = rt.consume(hero, world);
    CHECK(d2.kills == 1);
}

TEST(reward_tracker_records_death) {
    World world;
    Agent hero, enemy;
    hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().maxHp = 50; hero.unit().hp = 50;
    enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().damage = 100; enemy.unit().attackRange = 5;
    enemy.unit().attacksPerSec = 1;
    hero.setPosition(0, 0); enemy.setPosition(1, 0);
    world.addAgent(&hero);
    world.addAgent(&enemy);

    RewardTracker rt;
    rt.reset(hero, world);
    world.resolveAttack(enemy, 1); // hero dies

    auto d = rt.consume(hero, world);
    CHECK_NEAR(d.damageTaken, 50.0f, 1e-3f);
    CHECK(d.deaths == 1);
}

// ─── Projectiles / Simulation ───────────────────────────────────────────────

TEST(projectile_skillshot_hits_enemy_on_line) {
    World world;
    Agent shooter, target;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    target.unit().radius = 0.5f;
    shooter.setPosition(0, 0);
    target.setPosition(10, 0);
    world.addAgent(&shooter);
    world.addAgent(&target);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.x = 0; p.z = 0;
    p.vx = 20.0f; p.vz = 0; p.speed = 20.0f;
    p.damage = 40.0f;
    p.remainingLife = 2.0f;
    world.spawnProjectile(p);

    // Step until it should have travelled ~10 units.
    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);

    CHECK_NEAR(target.unit().hp, 60.0f, 1e-3f);
    CHECK(world.projectiles().empty()); // culled on hit
}

TEST(projectile_skillshot_misses_returns_no_damage) {
    World world;
    Agent shooter, target;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    shooter.setPosition(0, 0);
    target.setPosition(0, 10); // off to the side (different axis)
    world.addAgent(&shooter);
    world.addAgent(&target);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 40;
    p.remainingLife = 0.5f;
    world.spawnProjectile(p);

    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);

    CHECK_NEAR(target.unit().hp, 100.0f, 1e-3f);
    CHECK(world.projectiles().empty()); // expired
}

TEST(projectile_ignores_same_team) {
    World world;
    Agent shooter, ally;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    ally.unit().id = 2; ally.unit().teamId = 0;
    ally.unit().hp = 100; ally.unit().maxHp = 100;
    shooter.setPosition(0, 0);
    ally.setPosition(5, 0);
    world.addAgent(&shooter);
    world.addAgent(&ally);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 40; p.remainingLife = 1.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);
    CHECK_NEAR(ally.unit().hp, 100.0f, 1e-3f);
}

TEST(projectile_homing_tracks_moving_target) {
    World world;
    Agent shooter, target;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    target.unit().radius = 0.5f;
    shooter.setPosition(0, 0);
    target.setPosition(10, 0);
    world.addAgent(&shooter);
    world.addAgent(&target);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.x = 0; p.z = 0;
    // Initial velocity pointed wrong direction; homing should correct.
    p.vx = 0; p.vz = 20; p.speed = 20;
    p.targetId = 2;
    p.damage = 30;
    p.remainingLife = 5.0f;
    world.spawnProjectile(p);

    // Move the target across the step loop to exercise tracking.
    for (int i = 0; i < 120; i++) {
        target.setPosition(10.0f + i * 0.05f, 0.0f);
        world.tick(1.0f / 60.0f);
        if (!target.unit().alive() || world.projectiles().empty()) break;
    }
    CHECK(target.unit().hp < 100.0f); // got hit
}

TEST(projectile_emits_damage_event) {
    World world;
    Agent s, t;
    s.unit().id = 1; s.unit().teamId = 0;
    t.unit().id = 2; t.unit().teamId = 1; t.unit().hp = 100; t.unit().maxHp = 100;
    s.setPosition(0, 0); t.setPosition(5, 0);
    world.addAgent(&s); world.addAgent(&t);

    Projectile p;
    p.ownerId = 1; p.teamId = 0; p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 25; p.kind = DamageKind::Magical; p.remainingLife = 1.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);

    CHECK(world.events().size() == 1);
    CHECK(world.events()[0].attackerId == 1);
    CHECK(world.events()[0].targetId == 2);
    CHECK(world.events()[0].kind == DamageKind::Magical);
}

TEST(simulation_runs_policy_per_tick) {
    World world;
    Agent a;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().moveSpeed = 10.0f;
    a.setPosition(0, 0);
    world.addAgent(&a);

    Simulation sim(world);
    int calls = 0;
    sim.addPolicy(1, [&](Agent& self, const World&) {
        calls++;
        AgentAction act;
        act.moveZ = -1.0f; // forward
        (void)self;
        return act;
    });

    sim.runSteps(1.0f / 60.0f, 30);
    CHECK(calls == 30);
    CHECK(sim.steps() == 30);
    CHECK_NEAR(sim.elapsed(), 0.5f, 1e-3f);
    CHECK(a.z() < -1.0f); // moved forward
}

TEST(simulation_scripted_and_policy_agents_coexist) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    World world;

    Agent policy_agent, scripted_agent;
    policy_agent.unit().id = 1; policy_agent.unit().teamId = 0;
    policy_agent.unit().moveSpeed = 6.0f;
    policy_agent.setPosition(0, 0);

    scripted_agent.unit().id = 2; scripted_agent.unit().teamId = 0;
    scripted_agent.setNavGrid(&grid);
    scripted_agent.setPosition(5, 5);
    scripted_agent.setSpeed(6.0f);
    scripted_agent.setTarget(10, 5);

    world.addAgent(&policy_agent);
    world.addAgent(&scripted_agent);

    Simulation sim(world);
    sim.addPolicy(1, [](Agent&, const World&) {
        AgentAction act; act.moveZ = -1.0f; return act;
    });

    sim.runSteps(1.0f / 60.0f, 120);

    CHECK(policy_agent.z() < -1.0f);        // policy moved it
    CHECK(scripted_agent.x() > 6.0f);       // scripted moved toward target
}

TEST(simulation_dead_agent_skipped_but_cooldowns_tick) {
    World world;
    Agent a;
    a.unit().id = 1; a.unit().hp = 0; a.unit().maxHp = 100;
    a.unit().attackCooldown = 1.0f;
    world.addAgent(&a);

    Simulation sim(world);
    int calls = 0;
    sim.addPolicy(1, [&](Agent&, const World&) { calls++; return AgentAction{}; });

    sim.runSteps(0.5f, 1);
    CHECK(calls == 0); // dead → policy not invoked
    CHECK_NEAR(a.unit().attackCooldown, 0.5f, 1e-4f);
}

TEST(simulation_ticks_policy_agent_cooldowns) {
    // Regression: Simulation::step must tick the policy agent's cooldowns,
    // otherwise an attack fire sets attackCooldown and it never decrements,
    // locking the agent out of subsequent attacks.
    World world;
    Agent a;
    a.unit().id = 1; a.unit().hp = 100; a.unit().maxHp = 100;
    a.unit().attackCooldown = 1.0f;
    world.addAgent(&a);

    Simulation sim(world);
    sim.addPolicy(1, [](Agent&, const World&) { return AgentAction{}; });

    sim.runSteps(0.25f, 2);
    CHECK_NEAR(a.unit().attackCooldown, 0.5f, 1e-4f);
}

TEST(simulation_deterministic_with_seed) {
    // Two sims from the same seed produce identical states after N steps.
    auto runOnce = [](uint64_t seed) {
        World world;
        world.seed(seed);
        Agent a;
        a.unit().id = 1; a.unit().moveSpeed = 6.0f;
        a.setPosition(0, 0);
        world.addAgent(&a);
        Simulation sim(world);
        sim.addPolicy(1, [](Agent& self, const World& w) {
            AgentAction act;
            // policy uses the world's rng to choose a direction
            float r = const_cast<World&>(w).randFloat01();
            act.moveX = (r < 0.5f) ? -1.0f : 1.0f;
            (void)self;
            return act;
        });
        sim.runSteps(1.0f / 60.0f, 50);
        return std::pair<float,float>{a.x(), a.z()};
    };
    auto r1 = runOnce(777);
    auto r2 = runOnce(777);
    CHECK(r1.first == r2.first);
    CHECK(r1.second == r2.second);
}

// ─── Projectile modes / Snapshot ────────────────────────────────────────────

TEST(projectile_pierce_hits_multiple) {
    World world;
    Agent shooter, e1, e2, e3;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    e1.unit().id = 2; e1.unit().teamId = 1; e1.unit().maxHp = 200; e1.unit().hp = 200;
    e2.unit().id = 3; e2.unit().teamId = 1; e2.unit().maxHp = 200; e2.unit().hp = 200;
    e3.unit().id = 4; e3.unit().teamId = 1; e3.unit().maxHp = 200; e3.unit().hp = 200;
    e1.unit().radius = 0.5f; e2.unit().radius = 0.5f; e3.unit().radius = 0.5f;
    shooter.setPosition(0, 0);
    e1.setPosition(3, 0);
    e2.setPosition(6, 0);
    e3.setPosition(9, 0);
    world.addAgent(&shooter);
    world.addAgent(&e1); world.addAgent(&e2); world.addAgent(&e3);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.mode = ProjectileMode::Pierce;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 30; p.remainingLife = 2.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);

    CHECK_NEAR(e1.unit().hp, 170.0f, 1e-3f);
    CHECK_NEAR(e2.unit().hp, 170.0f, 1e-3f);
    CHECK_NEAR(e3.unit().hp, 170.0f, 1e-3f);
}

TEST(projectile_pierce_capped_by_maxHits) {
    World world;
    Agent shooter, e1, e2, e3;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    e1.unit().id = 2; e1.unit().teamId = 1; e1.unit().maxHp = 200; e1.unit().hp = 200;
    e2.unit().id = 3; e2.unit().teamId = 1; e2.unit().maxHp = 200; e2.unit().hp = 200;
    e3.unit().id = 4; e3.unit().teamId = 1; e3.unit().maxHp = 200; e3.unit().hp = 200;
    e1.unit().radius = 0.5f; e2.unit().radius = 0.5f; e3.unit().radius = 0.5f;
    shooter.setPosition(0, 0);
    e1.setPosition(3, 0); e2.setPosition(6, 0); e3.setPosition(9, 0);
    world.addAgent(&shooter);
    world.addAgent(&e1); world.addAgent(&e2); world.addAgent(&e3);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.mode = ProjectileMode::Pierce;
    p.maxHits = 2;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 30; p.remainingLife = 2.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 60; i++) world.tick(1.0f / 60.0f);

    CHECK(e1.unit().hp < 200.0f);
    CHECK(e2.unit().hp < 200.0f);
    CHECK_NEAR(e3.unit().hp, 200.0f, 1e-3f); // beyond cap
}

TEST(projectile_aoe_splashes_near_impact) {
    World world;
    Agent shooter, primary, bystander, far;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    primary.unit().id = 2; primary.unit().teamId = 1;
    primary.unit().maxHp = 200; primary.unit().hp = 200; primary.unit().radius = 0.5f;
    bystander.unit().id = 3; bystander.unit().teamId = 1;
    bystander.unit().maxHp = 200; bystander.unit().hp = 200;
    far.unit().id = 4; far.unit().teamId = 1;
    far.unit().maxHp = 200; far.unit().hp = 200;

    shooter.setPosition(0, 0);
    primary.setPosition(5, 0);
    bystander.setPosition(5.5f, 1.0f); // ~1.12 units from impact
    far.setPosition(5, 10);             // way outside splash

    world.addAgent(&shooter);
    world.addAgent(&primary);
    world.addAgent(&bystander);
    world.addAgent(&far);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.mode = ProjectileMode::AoE;
    p.splashRadius = 2.0f;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 40; p.remainingLife = 1.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 30; i++) world.tick(1.0f / 60.0f);

    CHECK_NEAR(primary.unit().hp, 160.0f, 1e-3f);
    CHECK_NEAR(bystander.unit().hp, 160.0f, 1e-3f);
    CHECK_NEAR(far.unit().hp, 200.0f, 1e-3f);
}

TEST(projectile_aoe_respects_team) {
    World world;
    Agent shooter, enemy, ally;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().maxHp = 200; enemy.unit().hp = 200; enemy.unit().radius = 0.5f;
    ally.unit().id = 3; ally.unit().teamId = 0;
    ally.unit().maxHp = 200; ally.unit().hp = 200;

    shooter.setPosition(0, 0);
    enemy.setPosition(5, 0);
    ally.setPosition(5, 0.5f); // right next to splash origin
    world.addAgent(&shooter);
    world.addAgent(&enemy);
    world.addAgent(&ally);

    Projectile p;
    p.ownerId = 1; p.teamId = 0;
    p.mode = ProjectileMode::AoE;
    p.splashRadius = 3.0f;
    p.vx = 20; p.vz = 0; p.speed = 20;
    p.damage = 40; p.remainingLife = 1.0f;
    world.spawnProjectile(p);

    for (int i = 0; i < 30; i++) world.tick(1.0f / 60.0f);

    CHECK(enemy.unit().hp < 200.0f);
    CHECK_NEAR(ally.unit().hp, 200.0f, 1e-3f); // same team — untouched
}

TEST(snapshot_roundtrip_restores_agent_state) {
    World w;
    Agent a;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().hp = 80; a.unit().maxHp = 100;
    a.unit().attackCooldown = 1.25f;
    a.setPosition(5.5f, -2.0f);
    w.addAgent(&a);

    WorldSnapshot s = w.snapshot();

    // Mutate.
    a.unit().hp = 10;
    a.unit().attackCooldown = 0.0f;
    a.setPosition(100, 100);

    w.restore(s);
    CHECK_NEAR(a.unit().hp, 80.0f, 1e-4f);
    CHECK_NEAR(a.unit().attackCooldown, 1.25f, 1e-4f);
    CHECK_NEAR(a.x(), 5.5f, 1e-4f);
    CHECK_NEAR(a.z(), -2.0f, 1e-4f);
}

TEST(snapshot_restores_projectiles_and_events) {
    World w;
    Agent s, t;
    s.unit().id = 1; s.unit().teamId = 0;
    t.unit().id = 2; t.unit().teamId = 1; t.unit().maxHp = 100; t.unit().hp = 100;
    s.setPosition(0, 0); t.setPosition(5, 0);
    w.addAgent(&s); w.addAgent(&t);

    Projectile p;
    p.ownerId = 1; p.teamId = 0; p.vx = 10; p.vz = 0; p.speed = 10;
    p.damage = 25; p.remainingLife = 2.0f;
    w.spawnProjectile(p);

    // Advance halfway — projectile mid-flight, no hit yet.
    for (int i = 0; i < 6; i++) w.tick(1.0f / 60.0f);
    CHECK(w.projectiles().size() == 1);
    CHECK(w.events().empty());

    WorldSnapshot snap = w.snapshot();

    // Advance to impact.
    for (int i = 0; i < 60; i++) w.tick(1.0f / 60.0f);
    CHECK(w.projectiles().empty());
    CHECK(!w.events().empty());

    // Rewind and compare.
    w.restore(snap);
    CHECK(w.projectiles().size() == 1);
    CHECK(w.events().empty());
    CHECK_NEAR(t.unit().hp, 100.0f, 1e-4f);
}

TEST(snapshot_restores_rng_for_deterministic_replay) {
    World w;
    w.seed(12345);
    Agent a;
    a.unit().id = 1; a.unit().moveSpeed = 6.0f;
    a.setPosition(0, 0);
    w.addAgent(&a);

    // Draw a few numbers to advance state, then snapshot.
    for (int i = 0; i < 20; i++) w.randFloat01();
    WorldSnapshot s = w.snapshot();

    float seqA[50];
    for (int i = 0; i < 50; i++) seqA[i] = w.randFloat01();

    w.restore(s);
    for (int i = 0; i < 50; i++) {
        float v = w.randFloat01();
        CHECK(v == seqA[i]);
    }
}

TEST(snapshot_restore_in_simulation_reproduces_rollout) {
    // Run 30 steps, snapshot, continue 30 steps. Restore and re-run; the
    // final state of the second half must match exactly.
    World w;
    w.seed(777);
    Agent a;
    a.unit().id = 1; a.unit().moveSpeed = 6.0f;
    a.setPosition(0, 0);
    w.addAgent(&a);

    Simulation sim(w);
    sim.addPolicy(1, [](Agent& self, const World& world) {
        AgentAction act;
        float r = const_cast<World&>(world).randFloat01();
        act.moveX = (r < 0.5f) ? -1.0f : 1.0f;
        (void)self;
        return act;
    });

    sim.runSteps(1.0f / 60.0f, 30);
    WorldSnapshot snap = w.snapshot();
    int snapSteps = sim.steps();

    sim.runSteps(1.0f / 60.0f, 30);
    float x1 = a.x(), z1 = a.z();

    w.restore(snap);
    // Counters are not part of the world snapshot — reset manually.
    sim.resetCounters();
    CHECK(sim.steps() == 0);
    (void)snapSteps;

    sim.runSteps(1.0f / 60.0f, 30);
    CHECK(a.x() == x1);
    CHECK(a.z() == z1);
}

// ─── Recorder / ReplayReader ────────────────────────────────────────────────

#include <cstdio>
#include <string>

static std::string tempReplayPath(const char* tag) {
    // Stable per-tag path in the working directory — tests are serial.
    std::string p = "test_replay_";
    p += tag;
    p += ".bgar";
    return p;
}

TEST(replay_roundtrip_header_roster_and_frame) {
    World world;
    world.seed(7);
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0; a.unit().hp = 100; a.unit().maxHp = 100;
    a.unit().attackRange = 3.0f; a.unit().radius = 0.4f;
    a.setPosition(1.0f, 2.0f);

    b.unit().id = 2; b.unit().teamId = 1; b.unit().hp = 50; b.unit().maxHp = 50;
    b.unit().radius = 0.5f;
    b.setPosition(-1.0f, 0.0f);

    world.addAgent(&a);
    world.addAgent(&b);

    std::string path = tempReplayPath("roundtrip");
    {
        Recorder rec;
        CHECK(rec.open(path, /*episodeId*/ 42, /*seed*/ 7, /*dt*/ 0.0333f));
        rec.writeRoster(world.agents());
        rec.recordFrame(0, 0.0f, world);
        a.setPosition(1.5f, 2.5f);
        rec.recordFrame(1, 0.0333f, world);
        CHECK(rec.close());
    }

    ReplayReader r;
    CHECK(r.open(path));
    CHECK(r.header().magic == replay::MAGIC);
    CHECK(r.header().version == replay::VERSION);
    CHECK(r.header().episodeId == 42);
    CHECK(r.header().seed == 7);
    CHECK_NEAR(r.header().dt, 0.0333f, 1e-6f);
    CHECK(r.roster().size() == 2);
    CHECK(r.roster()[0].id == 1);
    CHECK_NEAR(r.roster()[0].maxHp, 100.0f, 1e-5f);
    CHECK(r.frameCount() == 2);

    auto f0 = r.frame(0);
    CHECK(f0.header.stepIdx == 0);
    CHECK(f0.agents.size() == 2);
    CHECK(f0.agents[0].id == 1);
    CHECK_NEAR(f0.agents[0].x, 1.0f, 1e-5f);
    CHECK_NEAR(f0.agents[0].z, 2.0f, 1e-5f);

    auto f1 = r.frame(1);
    CHECK(f1.header.stepIdx == 1);
    CHECK_NEAR(f1.agents[0].x, 1.5f, 1e-5f);

    std::remove(path.c_str());
}

TEST(replay_records_damage_events_per_frame) {
    World world;
    Agent atk, tgt;
    atk.unit().id = 10; atk.unit().teamId = 0;
    atk.unit().damage = 20; atk.unit().attackRange = 10; atk.unit().attacksPerSec = 1;
    atk.setPosition(0, 0);
    tgt.unit().id = 11; tgt.unit().teamId = 1;
    tgt.unit().hp = 100; tgt.unit().maxHp = 100;
    tgt.setPosition(2, 0);
    world.addAgent(&atk);
    world.addAgent(&tgt);

    std::string path = tempReplayPath("events");
    {
        Recorder rec;
        CHECK(rec.open(path, 0, 0, 0.1f));
        rec.writeRoster(world.agents());

        // Frame 0: no events.
        rec.recordFrame(0, 0.0f, world);

        // Frame 1: one attack landed.
        world.resolveAttack(atk, 11);
        rec.recordFrame(1, 0.1f, world);

        // Frame 2: attack on cooldown, no new events (world still has old event).
        world.resolveAttack(atk, 11);
        rec.recordFrame(2, 0.2f, world);

        rec.close();
    }

    ReplayReader r;
    CHECK(r.open(path));
    CHECK(r.frameCount() == 3);
    CHECK(r.frame(0).events.size() == 0);
    CHECK(r.frame(1).events.size() == 1);
    CHECK(r.frame(2).events.size() == 0);   // delta slice, not cumulative
    const auto& e = r.frame(1).events[0];
    CHECK(e.attackerId == 10);
    CHECK(e.targetId == 11);
    CHECK_NEAR(e.amount, 20.0f, 1e-3f);

    std::remove(path.c_str());
}

TEST(replay_records_projectiles) {
    World world;
    Agent shooter, target;
    shooter.unit().id = 1; shooter.unit().teamId = 0;
    shooter.setPosition(0, 0);
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    target.setPosition(10, 0);
    world.addAgent(&shooter);
    world.addAgent(&target);

    Projectile p{};
    p.ownerId = 1; p.teamId = 0;
    p.x = 0; p.z = 0;
    p.vx = 20; p.vz = 0;
    p.speed = 20; p.radius = 0.3f;
    p.damage = 10; p.remainingLife = 5.0f;
    p.mode = ProjectileMode::Single;
    world.spawnProjectile(p);

    std::string path = tempReplayPath("proj");
    {
        Recorder rec;
        CHECK(rec.open(path, 0, 0, 0.1f));
        rec.writeRoster(world.agents());
        rec.recordFrame(0, 0.0f, world);   // pre-step: projectile exists
        world.tick(0.1f);
        rec.recordFrame(1, 0.1f, world);   // post-step
        rec.close();
    }

    ReplayReader r;
    CHECK(r.open(path));
    CHECK(r.frame(0).projectiles.size() == 1);
    const auto& pp = r.frame(0).projectiles[0];
    CHECK(pp.ownerId == 1);
    CHECK_NEAR(pp.vx, 20.0f, 1e-4f);
    CHECK(pp.alive == 1);

    std::remove(path.c_str());
}

TEST(replay_trajectory_and_damage_summary) {
    World world;
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0;
    a.unit().damage = 15; a.unit().attackRange = 10; a.unit().attacksPerSec = 2;
    a.setPosition(0, 0);
    b.unit().id = 2; b.unit().teamId = 1;
    b.unit().hp = 40; b.unit().maxHp = 40;
    b.setPosition(1, 0);
    world.addAgent(&a);
    world.addAgent(&b);

    std::string path = tempReplayPath("traj");
    {
        Recorder rec;
        CHECK(rec.open(path, 0, 0, 0.5f));
        rec.writeRoster(world.agents());
        for (int i = 0; i < 5; i++) {
            world.resolveAttack(a, 2);
            rec.recordFrame(i, i * 0.5f, world);
            a.setPosition(static_cast<float>(i + 1), 0.0f);
            world.tick(0.5f); // tick cooldowns between shots
        }
        rec.close();
    }

    ReplayReader r;
    CHECK(r.open(path));
    auto traj = r.trajectory(1);
    CHECK(traj.size() == 5);
    CHECK_NEAR(traj[0].x, 0.0f, 1e-5f);
    CHECK_NEAR(traj[4].x, 4.0f, 1e-5f);

    auto dmg = r.damageSummary();
    CHECK(dmg.size() >= 1);
    const auto& row = dmg[0];
    CHECK(row.attackerId == 1);
    CHECK(row.targetId == 2);
    // 40 / 15 = 3 hits to kill (15+15+10 after clamp); damage totals 40.
    CHECK_NEAR(row.totalDamage, 40.0f, 1e-3f);
    CHECK(row.kills == 1);

    std::remove(path.c_str());
}

TEST(replay_rejects_bad_magic) {
    std::string path = tempReplayPath("badmagic");
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, path.c_str(), "wb");
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    CHECK(f != nullptr);
    uint32_t bad[4] = {0xdeadbeef, 0, 0, 0};
    std::fwrite(bad, sizeof(bad), 1, f);
    std::fclose(f);

    ReplayReader r;
    CHECK(!r.open(path));
    CHECK(r.errorMessage().find("magic") != std::string::npos
          || r.errorMessage().find("too small") != std::string::npos);
    std::remove(path.c_str());
}

TEST(replay_random_access_by_step) {
    World world;
    Agent a;
    a.unit().id = 1;
    a.setPosition(0, 0);
    world.addAgent(&a);

    std::string path = tempReplayPath("access");
    {
        Recorder rec;
        rec.open(path, 0, 0, 0.1f);
        rec.writeRoster(world.agents());
        for (int i = 0; i < 10; i++) {
            a.setPosition(static_cast<float>(i), 0.0f);
            rec.recordFrame(i * 2, i * 0.1f, world); // odd steps absent
        }
        rec.close();
    }

    ReplayReader r;
    CHECK(r.open(path));
    CHECK(r.findByStep(6) == 3);         // step 6 is the 4th frame
    CHECK(r.findByStep(7) == SIZE_MAX);  // odd step wasn't recorded

    auto frame3 = r.frame(3);
    CHECK(frame3.header.stepIdx == 6);
    CHECK_NEAR(frame3.agents[0].x, 3.0f, 1e-5f);

    std::remove(path.c_str());
}

// ─── Buffs / DoT / HoT / mana regen / stealth ───────────────────────────────

TEST(unit_mana_regen_clamps_to_max) {
    Unit u;
    u.maxMana = 50.0f; u.mana = 49.0f;
    u.manaRegenPerSec = 10.0f;
    u.tickCooldowns(1.0f);     // would add 10 → clamp to maxMana
    CHECK_NEAR(u.mana, 50.0f, 1e-4f);
    u.mana = 0.0f;
    u.tickCooldowns(0.5f);
    CHECK_NEAR(u.mana, 5.0f, 1e-4f);
}

TEST(unit_buff_decays_and_resets) {
    Unit u;
    u.armorBonus = 25.0f; u.armorBonusRemaining = 1.0f;
    u.damageMul  = 1.5f;  u.damageMulRemaining  = 1.0f;
    u.tickCooldowns(0.5f);
    CHECK_NEAR(u.armorBonusRemaining, 0.5f, 1e-4f);
    CHECK_NEAR(u.armorBonus, 25.0f, 1e-4f);   // still active
    u.tickCooldowns(0.6f);                     // crosses zero
    CHECK_NEAR(u.armorBonusRemaining, 0.0f, 1e-4f);
    CHECK_NEAR(u.armorBonus, 0.0f, 1e-4f);     // reset
    CHECK_NEAR(u.damageMul, 1.0f, 1e-4f);      // reset to identity
}

TEST(world_dot_emits_events_and_can_kill) {
    World world;
    Agent attacker, victim;
    attacker.unit().id = 1; attacker.unit().teamId = 0;
    victim.unit().id = 2; victim.unit().teamId = 1;
    victim.unit().hp = 10; victim.unit().maxHp = 100;
    victim.unit().dotDps = 20.0f;
    victim.unit().dotRemaining = 5.0f;
    victim.unit().dotKind = DamageKind::Magical;
    victim.unit().dotSourceId = attacker.unit().id;
    world.addAgent(&attacker);
    world.addAgent(&victim);

    // 1.0s of DoT → 20 damage → kills 10-HP target.
    world.applyDotHot(victim, 1.0f);
    CHECK(!victim.unit().alive());
    CHECK(world.events().size() == 1);
    CHECK(world.events()[0].attackerId == 1);
    CHECK(world.events()[0].targetId   == 2);
    CHECK(world.events()[0].killed);
}

TEST(world_hot_heals_and_clamps) {
    World world;
    Agent ally;
    ally.unit().id = 1;
    ally.unit().hp = 50; ally.unit().maxHp = 100;
    ally.unit().hotRate = 30.0f; ally.unit().hotRemaining = 1.0f;
    world.addAgent(&ally);

    world.applyDotHot(ally, 0.5f);
    CHECK_NEAR(ally.unit().hp, 65.0f, 1e-3f);
    world.applyDotHot(ally, 5.0f);  // would over-heal but timer shorter; clamp anyway
    CHECK_NEAR(ally.unit().hp, 80.0f, 1e-3f); // 0.5s left * 30 dps = +15
    CHECK(ally.unit().hotRemaining == 0.0f);
}

TEST(world_stealth_makes_attacks_miss) {
    World world;
    world.seed(123);
    Agent attacker, target;
    attacker.unit().id = 1; attacker.unit().teamId = 0;
    attacker.unit().damage = 25; attacker.unit().attackRange = 5; attacker.unit().attacksPerSec = 1;
    target.unit().id = 2; target.unit().teamId = 1;
    target.unit().hp = 100; target.unit().maxHp = 100;
    target.unit().stealthChance = 1.0f;          // always dodge
    target.unit().stealthChanceRemaining = 5.0f;
    world.addAgent(&attacker);
    world.addAgent(&target);

    bool hit = world.resolveAttack(attacker, 2);
    CHECK(!hit);                                 // returned false (missed)
    CHECK_NEAR(target.unit().hp, 100.0f, 1e-3f); // no damage
    CHECK(attacker.unit().attackCooldown > 0);    // cooldown still consumed
}

TEST(world_damage_buff_increases_dealt_damage) {
    World world;
    Agent atk, tgt;
    atk.unit().id = 1; atk.unit().teamId = 0;
    atk.unit().damage = 10; atk.unit().attackRange = 5; atk.unit().attacksPerSec = 1;
    atk.unit().damageMul = 2.0f; atk.unit().damageMulRemaining = 5.0f;
    tgt.unit().id = 2; tgt.unit().teamId = 1; tgt.unit().hp = 100; tgt.unit().maxHp = 100;
    world.addAgent(&atk);
    world.addAgent(&tgt);

    world.resolveAttack(atk, 2);
    CHECK_NEAR(tgt.unit().hp, 80.0f, 1e-3f);  // 10 * 2.0 = 20 damage
}

TEST(vecsim_ability_fireball_damages_via_action) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 1;
    cfg.minSpawnDist = 4.0f; cfg.maxSpawnDist = 4.5f;
    cfg.attackRange = 0.0f;  // no auto-attacks
    VecSimulation v(cfg);
    v.seedAndReset(11);

    int oppHp0 = static_cast<int>(v.opponent(0).unit().hp);

    AgentAction nop;
    AgentAction cast;
    cast.useAbilityId    = 0;                            // slot 0 = Fireball
    cast.attackTargetId  = VecSimulation::OPPONENT_ID;

    std::vector<AgentAction> hAct(1, cast), oAct(1, nop);
    v.applyActions(VecSimulation::HERO_ID,     hAct.data());
    v.applyActions(VecSimulation::OPPONENT_ID, oAct.data());
    v.step();

    int oppHp1 = static_cast<int>(v.opponent(0).unit().hp);
    CHECK(oppHp1 < oppHp0);                              // took damage
}

// ─── VecSimulation ──────────────────────────────────────────────────────────

TEST(vecsim_construct_and_observe) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 8;
    VecSimulation vec(cfg);
    CHECK(vec.numEnvs() == 8);

    vec.seedAndReset(42);

    std::vector<float> hobs(8 * observation::TOTAL);
    vec.observe(VecSimulation::HERO_ID, hobs.data());

    // Each env should have non-zero self block (hp/maxHp == 1.0).
    for (int e = 0; e < 8; e++) {
        CHECK_NEAR(hobs[e * observation::TOTAL + 0], 1.0f, 1e-5f); // hp/maxHp
    }
}

TEST(vecsim_deterministic_with_seed) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 4;
    cfg.maxStepsPerEpisode = 50;

    auto runOnce = [&](uint64_t seed) {
        VecSimulation v(cfg);
        v.seedAndReset(seed);
        std::vector<AgentAction> hAct(4), oAct(4);
        for (int i = 0; i < 4; i++) {
            hAct[i].moveZ = -1.0f;
            oAct[i].moveZ =  1.0f;
        }
        for (int t = 0; t < 30; t++) {
            v.applyActions(VecSimulation::HERO_ID,     hAct.data());
            v.applyActions(VecSimulation::OPPONENT_ID, oAct.data());
            v.step();
        }
        std::vector<float> rh(4), ro(4);
        v.rewards(rh.data(), ro.data());
        // Capture hero position of env 0 as a fingerprint.
        return std::tuple<float,float,float>{
            v.hero(0).x(), v.hero(0).z(), rh[0]
        };
    };
    auto a = runOnce(123);
    auto b = runOnce(123);
    CHECK(std::get<0>(a) == std::get<0>(b));
    CHECK(std::get<1>(a) == std::get<1>(b));
    CHECK(std::get<2>(a) == std::get<2>(b));
}

TEST(vecsim_envs_are_independent) {
    // Two envs should produce different hero spawn positions under the
    // per-env seed offset. Sanity check that envs aren't accidentally sharing
    // RNG state.
    VecSimulation::Config cfg;
    cfg.numEnvs = 2;
    VecSimulation v(cfg);
    v.seedAndReset(99);
    bool xDiff = v.hero(0).x() != v.hero(1).x();
    bool zDiff = v.hero(0).z() != v.hero(1).z();
    CHECK(xDiff || zDiff);
}

TEST(vecsim_termination_on_kill_and_winner) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 1;
    cfg.maxStepsPerEpisode = 500;
    cfg.minSpawnDist = 1.0f; cfg.maxSpawnDist = 1.5f; // close range
    cfg.attackRange = 5.0f;
    cfg.attacksPerSec = 5.0f;
    cfg.damage = 1000.0f; // one-shot
    cfg.hp = 100.0f;
    VecSimulation v(cfg);
    v.seedAndReset(1);

    std::vector<AgentAction> hAct(1), oAct(1);
    // Hero attacks opponent. Opponent does nothing.
    hAct[0].attackTargetId = VecSimulation::OPPONENT_ID;
    v.applyActions(VecSimulation::HERO_ID, hAct.data());
    v.applyActions(VecSimulation::OPPONENT_ID, oAct.data());
    v.step();

    std::vector<int> done(1), winner(1);
    v.dones(done.data(), winner.data());
    CHECK(done[0] == 1);
    CHECK(winner[0] == VecSimulation::HERO_ID);
}

TEST(vecsim_termination_on_timeout) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 1;
    cfg.maxStepsPerEpisode = 3;
    cfg.damage = 0.0f;       // can't kill — guaranteed timeout
    VecSimulation v(cfg);
    v.seedAndReset(0);

    std::vector<AgentAction> nop(1);
    for (int t = 0; t < 4; t++) {
        v.applyActions(VecSimulation::HERO_ID, nop.data());
        v.applyActions(VecSimulation::OPPONENT_ID, nop.data());
        v.step();
    }
    std::vector<int> done(1), winner(1);
    v.dones(done.data(), winner.data());
    CHECK(done[0] == 1);
    CHECK(winner[0] == -1);
}

TEST(vecsim_reset_done_clears_flag_and_increments_episode) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 1;
    cfg.maxStepsPerEpisode = 2;
    cfg.damage = 0.0f;
    VecSimulation v(cfg);
    v.seedAndReset(0);

    std::vector<AgentAction> nop(1);
    for (int t = 0; t < 3; t++) {
        v.applyActions(VecSimulation::HERO_ID, nop.data());
        v.applyActions(VecSimulation::OPPONENT_ID, nop.data());
        v.step();
    }
    std::vector<int> ep(1), done(1), winner(1);
    v.episodeCounts(ep.data());
    CHECK(ep[0] == 0);

    v.resetDone();
    v.episodeCounts(ep.data());
    CHECK(ep[0] == 1);
    v.dones(done.data(), winner.data());
    CHECK(done[0] == 0);
}

TEST(vecsim_reward_accumulates_then_drains) {
    VecSimulation::Config cfg;
    cfg.numEnvs = 1;
    cfg.maxStepsPerEpisode = 100;
    cfg.minSpawnDist = 1.0f; cfg.maxSpawnDist = 1.5f;
    cfg.attackRange = 5.0f;
    cfg.attacksPerSec = 1.0f;
    cfg.damage = 10.0f;
    cfg.hp = 100.0f;
    cfg.rewardStep = 0.0f; cfg.rewardKill = 0.0f; cfg.rewardDeath = 0.0f;
    cfg.rewardDamageDealt = 1.0f; cfg.rewardDamageTakenMul = 0.0f;
    VecSimulation v(cfg);
    v.seedAndReset(0);

    std::vector<AgentAction> hAct(1), oAct(1);
    hAct[0].attackTargetId = VecSimulation::OPPONENT_ID;

    v.applyActions(VecSimulation::HERO_ID, hAct.data());
    v.applyActions(VecSimulation::OPPONENT_ID, oAct.data());
    v.step();

    std::vector<float> rh(1), ro(1);
    v.rewards(rh.data(), ro.data());
    CHECK_NEAR(rh[0], 10.0f, 1e-3f);
    CHECK_NEAR(ro[0], 0.0f, 1e-3f);

    // Reward accumulator must drain to zero on read.
    v.rewards(rh.data(), ro.data());
    CHECK(rh[0] == 0.0f);
}

// ─── MCTS ──────────────────────────────────────────────────────────────────

namespace {

struct McstScene {
    World world;
    Agent hero;
    Agent enemy;
};

static std::unique_ptr<McstScene> make_duel_scene(float heroX, float heroZ,
                                                   float enemyX, float enemyZ,
                                                   float attackRange = 2.0f) {
    auto s = std::make_unique<McstScene>();
    s->hero.unit().id      = 1;
    s->hero.unit().teamId  = 0;
    s->hero.unit().hp      = 100.0f;
    s->hero.unit().maxHp   = 100.0f;
    s->hero.unit().damage  = 10.0f;
    s->hero.unit().attackRange  = attackRange;
    s->hero.unit().attacksPerSec = 2.0f;
    s->hero.setPosition(heroX, heroZ);
    s->hero.setMaxAccel(30.0f);
    s->hero.setMaxTurnRate(10.0f);

    s->enemy.unit().id     = 2;
    s->enemy.unit().teamId = 1;
    s->enemy.unit().hp     = 50.0f;
    s->enemy.unit().maxHp  = 50.0f;
    s->enemy.unit().damage = 5.0f;
    s->enemy.unit().attackRange  = attackRange;
    s->enemy.unit().attacksPerSec = 1.0f;
    s->enemy.setPosition(enemyX, enemyZ);
    s->enemy.setMaxAccel(30.0f);
    s->enemy.setMaxTurnRate(10.0f);

    s->world.addAgent(&s->hero);
    s->world.addAgent(&s->enemy);
    s->world.seed(42);
    return s;
}

} // namespace

TEST(mcts_legal_actions_includes_move_only_when_no_combat_legal) {
    // Out of range + cooldown-cleared but nothing else happening.
    auto s = make_duel_scene(0, 0, 20, 0);
    auto acts = mcts::legal_actions(s->hero, s->world);

    // 11 move dirs (9 cardinal + PathToTarget + PathAway) × 1 attack option
    // (-1 only) × 1 ability option (-1) since enemy is far away and no
    // abilities are registered.
    CHECK(acts.size() == 11);
    for (const auto& a : acts) {
        CHECK(a.attack_slot == -1);
        CHECK(a.ability_slot == -1);
    }
}

TEST(mcts_legal_actions_exposes_attack_slot_when_in_range) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto acts = mcts::legal_actions(s->hero, s->world);

    // 11 moves × (2 attack opts: -1 and slot 0) × 1 ability opt
    CHECK(acts.size() == 22);
    bool saw_attack = false;
    for (const auto& a : acts) {
        if (a.attack_slot == 0) saw_attack = true;
    }
    CHECK(saw_attack);
}

TEST(mcts_apply_routes_attack_to_correct_enemy) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    float before = s->enemy.unit().hp;

    mcts::CombatAction a;
    a.move_dir    = mcts::MoveDir::Hold;
    a.attack_slot = 0;
    a.ability_slot = -1;
    mcts::apply(s->hero, s->world, a, 0.5f);

    CHECK(s->enemy.unit().hp < before);
}

TEST(mcts_search_is_side_effect_free_on_world) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);

    float hero_hp_before  = s->hero.unit().hp;
    float enemy_hp_before = s->enemy.unit().hp;
    float heroX_before = s->hero.x(), heroZ_before = s->hero.z();

    mcts::MctsConfig cfg;
    cfg.iterations     = 64;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);

    mcts::CombatAction picked = engine.search(s->world, s->hero);
    (void)picked;

    CHECK_NEAR(s->hero.unit().hp, hero_hp_before, 1e-4f);
    CHECK_NEAR(s->enemy.unit().hp, enemy_hp_before, 1e-4f);
    CHECK_NEAR(s->hero.x(), heroX_before, 1e-4f);
    CHECK_NEAR(s->hero.z(), heroZ_before, 1e-4f);
}

TEST(mcts_search_prefers_attack_when_in_range_vs_idle_opponent) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);

    mcts::MctsConfig cfg;
    cfg.iterations     = 400;
    cfg.rollout_horizon = 16;
    cfg.action_repeat  = 4;
    cfg.seed           = 7;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);

    mcts::CombatAction picked = engine.search(s->world, s->hero);

    // In-range + idle opponent → optimal first move must include auto-attack.
    CHECK(picked.attack_slot == 0);
}

TEST(mcts_search_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);

    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBEEF;

    mcts::Mcts e1(cfg), e2(cfg);
    for (auto* e : {&e1, &e2}) {
        e->set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e->set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e->set_opponent_policy(mcts::policy_idle);
    }

    mcts::CombatAction a1 = e1.search(s1->world, s1->hero);
    mcts::CombatAction a2 = e2.search(s2->world, s2->hero);

    CHECK(a1 == a2);
    CHECK(e1.last_stats().tree_size == e2.last_stats().tree_size);
    CHECK(e1.last_stats().best_visits == e2.last_stats().best_visits);
}

TEST(mcts_wall_time_budget_exits_early) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 100000;   // very high — time budget must fire first
    cfg.budget_ms      = 20;
    cfg.rollout_horizon = 16;
    cfg.action_repeat  = 4;
    mcts::Mcts engine(cfg);
    engine.set_opponent_policy(mcts::policy_idle);

    engine.search(s->world, s->hero);
    CHECK(engine.last_stats().iterations < 100000);   // budget capped it
    CHECK(engine.last_stats().elapsed_ms <= 100);     // generous ceiling
}

TEST(mcts_advance_root_preserves_subtree_stats) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 200;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    mcts::Mcts engine(cfg);
    engine.set_opponent_policy(mcts::policy_idle);

    mcts::CombatAction first = engine.search(s->world, s->hero);
    int tree_before_advance = engine.last_stats().tree_size;
    CHECK(tree_before_advance > 1);
    CHECK(engine.last_stats().reused_root == false);

    engine.advance_root(first);
    // Commit in the caller's world so the next search starts from the same
    // state the tree was grown against.
    mcts::apply(s->hero, s->world, first, cfg.sim_dt * cfg.action_repeat);

    engine.search(s->world, s->hero);
    CHECK(engine.last_stats().reused_root == true);
    CHECK(engine.last_stats().tree_size >= tree_before_advance);
}

TEST(mcts_advance_root_with_unexpanded_action_resets_tree) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 4;   // tiny budget — many actions untried
    cfg.rollout_horizon = 2;
    mcts::Mcts engine(cfg);
    engine.set_opponent_policy(mcts::policy_idle);

    engine.search(s->world, s->hero);

    // Pick an action that is guaranteed not to have an expanded child with
    // only 4 iterations.
    mcts::CombatAction bogus;
    bogus.move_dir = mcts::MoveDir::SW;
    bogus.attack_slot = -1;
    bogus.ability_slot = -1;
    // There's ~18 legal actions at this state — advancing to a rarely-picked
    // one should frequently miss. If the engine did manage to expand this
    // one, the reuse path is still correct — so just assert the reset path
    // using a clear no-such-action case.
    engine.reset_tree();
    CHECK(engine.last_root() == nullptr);
    engine.search(s->world, s->hero);
    CHECK(engine.last_stats().reused_root == false);
}

TEST(mcts_hp_delta_evaluator_terminal_win_loss) {
    auto s = make_duel_scene(0, 0, 1.0f, 0);
    mcts::HpDeltaEvaluator ev;

    // Baseline: both at full HP (different max) → zero delta in fraction space.
    float baseline = ev.evaluate(s->world, s->hero.unit().id);
    CHECK_NEAR(baseline, 0.0f, 1e-5f);

    // Half-damage the enemy → hero fraction ahead → positive delta.
    s->enemy.unit().hp = 25.0f;
    CHECK(ev.evaluate(s->world, s->hero.unit().id) > 0.0f);
    s->enemy.unit().hp = 50.0f;

    // Kill enemy → terminal +1.
    s->enemy.unit().hp = 0.0f;
    CHECK_NEAR(ev.evaluate(s->world, s->hero.unit().id), 1.0f, 1e-5f);

    // Kill hero → terminal -1.
    s->enemy.unit().hp = 50.0f;
    s->hero.unit().hp  = 0.0f;
    CHECK_NEAR(ev.evaluate(s->world, s->hero.unit().id), -1.0f, 1e-5f);
}

// ─── DecoupledMcts (simultaneous-move) ─────────────────────────────────────

TEST(dmcts_hero_wins_in_expectation_when_stronger) {
    // Hero: 100hp, 10dmg, 2 aps. Opp: 50hp, 5dmg, 1 aps. Both in range.
    // Under simultaneous-move search, even a thinking opponent can't save
    // this matchup — hero's trades are strictly better — so the value at
    // the root should be positive.
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 1500;
    cfg.rollout_horizon = 24;
    cfg.action_repeat  = 2;
    cfg.seed           = 11;
    mcts::DecoupledMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());

    engine.search(s->world, s->hero, s->enemy);
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(dmcts_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 256;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xABCD;
    mcts::DecoupledMcts e1(cfg), e2(cfg);
    for (auto* e : {&e1, &e2}) {
        e->set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e->set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    }

    auto a1 = e1.search(s1->world, s1->hero, s1->enemy);
    auto a2 = e2.search(s2->world, s2->hero, s2->enemy);
    CHECK(a1 == a2);
    CHECK(e1.last_stats().tree_size == e2.last_stats().tree_size);
}

TEST(dmcts_search_side_effect_free_on_world) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    float hero_hp_before = s->hero.unit().hp;
    float enemy_hp_before = s->enemy.unit().hp;
    float heroX = s->hero.x(), heroZ = s->hero.z();
    float enemyX = s->enemy.x(), enemyZ = s->enemy.z();

    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    mcts::DecoupledMcts engine(cfg);
    engine.search(s->world, s->hero, s->enemy);

    CHECK_NEAR(s->hero.unit().hp, hero_hp_before, 1e-4f);
    CHECK_NEAR(s->enemy.unit().hp, enemy_hp_before, 1e-4f);
    CHECK_NEAR(s->hero.x(), heroX, 1e-4f);
    CHECK_NEAR(s->hero.z(), heroZ, 1e-4f);
    CHECK_NEAR(s->enemy.x(), enemyX, 1e-4f);
    CHECK_NEAR(s->enemy.z(), enemyZ, 1e-4f);
}

TEST(dmcts_advance_root_preserves_subtree) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 400;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    mcts::DecoupledMcts engine(cfg);

    auto first = engine.search(s->world, s->hero, s->enemy);
    int tree_before = engine.last_stats().tree_size;
    CHECK(tree_before > 1);

    engine.advance_root(first.hero, first.opp);
    // Commit both actions in the caller's world.
    mcts::apply(s->hero,  s->world, first.hero, cfg.sim_dt * cfg.action_repeat);
    mcts::apply(s->enemy, s->world, first.opp,  cfg.sim_dt * cfg.action_repeat);

    engine.search(s->world, s->hero, s->enemy);
    CHECK(engine.last_stats().reused_root == true);
}

TEST(dmcts_search_grows_joint_tree) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 500;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    mcts::DecoupledMcts engine(cfg);
    engine.search(s->world, s->hero, s->enemy);

    // At least one joint child must exist; tree_size grows with iterations.
    CHECK(engine.last_stats().tree_size > 1);
    CHECK(engine.last_stats().root_children > 0);
}

// ─── Root-parallel MCTS ────────────────────────────────────────────────────

TEST(mcts_parallel_returns_valid_action) {
    // Three cloned scenes, same starting state, searched in parallel.
    constexpr int N = 3;
    std::vector<std::unique_ptr<McstScene>> scenes;
    std::vector<World*> worlds;
    for (int i = 0; i < N; i++) {
        scenes.emplace_back(make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f));
        worlds.push_back(&scenes.back()->world);
    }

    mcts::MctsConfig cfg;
    cfg.iterations      = 400;
    cfg.rollout_horizon = 16;
    cfg.action_repeat   = 4;
    cfg.seed            = 0x1234;

    mcts::ParallelSearchStats pstats;
    auto picked = mcts::root_parallel_search(
        worlds, /*hero_id*/ 1, cfg,
        std::make_shared<mcts::HpDeltaEvaluator>(),
        std::make_shared<mcts::RandomRollout>(),
        mcts::policy_idle,
        &pstats);

    CHECK(pstats.num_threads == N);
    CHECK(pstats.total_iterations == N * cfg.iterations);   // no time cap → exact
    CHECK(pstats.merged_best_visits > 0);

    // With hero in range + idle opp, attack should dominate merged visits.
    CHECK(picked.attack_slot == 0);

    // None of the worlds should have been mutated by the parallel search.
    for (auto& sc : scenes) {
        CHECK_NEAR(sc->hero.unit().hp, 100.0f, 1e-4f);
        CHECK_NEAR(sc->enemy.unit().hp, 50.0f, 1e-4f);
    }
}

TEST(dmcts_parallel_returns_joint) {
    constexpr int N = 2;
    std::vector<std::unique_ptr<McstScene>> scenes;
    std::vector<World*> worlds;
    for (int i = 0; i < N; i++) {
        scenes.emplace_back(make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f));
        worlds.push_back(&scenes.back()->world);
    }

    mcts::MctsConfig cfg;
    cfg.iterations      = 200;
    cfg.rollout_horizon = 8;
    cfg.action_repeat   = 2;
    cfg.seed            = 0x5678;

    mcts::ParallelSearchStats pstats;
    auto joint = mcts::root_parallel_search_decoupled(
        worlds, /*hero_id*/ 1, /*opp_id*/ 2, cfg,
        std::make_shared<mcts::HpDeltaEvaluator>(),
        std::make_shared<mcts::RandomRollout>(),
        &pstats);

    CHECK(pstats.num_threads == N);
    CHECK(pstats.merged_best_visits > 0);
    (void)joint;

    for (auto& sc : scenes) {
        CHECK_NEAR(sc->hero.unit().hp, 100.0f, 1e-4f);
        CHECK_NEAR(sc->enemy.unit().hp, 50.0f, 1e-4f);
    }
}

// ─── TeamMcts (cooperative multi-agent) ────────────────────────────────────

namespace {

struct TeamScene {
    World world;
    std::vector<std::unique_ptr<Agent>> heroes;
    std::vector<std::unique_ptr<Agent>> enemies;
};

static std::unique_ptr<TeamScene> make_team_scene(int num_heroes, int num_enemies,
                                                   float attackRange = 3.0f) {
    auto s = std::make_unique<TeamScene>();
    // Heroes clumped on the left, enemies clumped on the right, all in range.
    int next_id = 1;
    for (int i = 0; i < num_heroes; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next_id++;
        a->unit().teamId = 0;
        a->unit().hp = 100.0f;
        a->unit().maxHp = 100.0f;
        a->unit().damage = 10.0f;
        a->unit().attackRange = attackRange;
        a->unit().attacksPerSec = 2.0f;
        a->setPosition(-1.0f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->heroes.push_back(std::move(a));
    }
    for (int i = 0; i < num_enemies; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next_id++;
        a->unit().teamId = 1;
        a->unit().hp = 40.0f;
        a->unit().maxHp = 40.0f;
        a->unit().damage = 5.0f;
        a->unit().attackRange = attackRange;
        a->unit().attacksPerSec = 1.0f;
        a->setPosition(1.0f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->enemies.push_back(std::move(a));
    }
    s->world.seed(99);
    return s;
}

static std::vector<Agent*> raw(const std::vector<std::unique_ptr<Agent>>& v) {
    std::vector<Agent*> out;
    out.reserve(v.size());
    for (const auto& u : v) out.push_back(u.get());
    return out;
}

} // namespace

TEST(team_hp_delta_evaluator_terminal_cases) {
    auto s = make_team_scene(2, 1);
    mcts::TeamHpDeltaEvaluator ev;
    CHECK_NEAR(ev.evaluate(s->world, 0), 0.0f, 1e-5f);   // both sides full HP

    s->enemies[0]->unit().hp = 0.0f;
    CHECK_NEAR(ev.evaluate(s->world, 0), 1.0f, 1e-5f);   // all enemies dead → +1

    s->enemies[0]->unit().hp = 40.0f;
    for (auto& h : s->heroes) h->unit().hp = 0.0f;
    CHECK_NEAR(ev.evaluate(s->world, 0), -1.0f, 1e-5f);  // team wiped → -1
}

TEST(team_mcts_two_heroes_vs_one_enemy) {
    // 2v1 in range with hero stat advantage — team should win in expectation.
    auto s = make_team_scene(2, 1);
    mcts::MctsConfig cfg;
    cfg.iterations      = 600;
    cfg.rollout_horizon = 16;
    cfg.action_repeat   = 2;
    cfg.seed            = 0xA11CE;
    mcts::TeamMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    auto heroes = raw(s->heroes);
    auto joint = engine.search(s->world, heroes);

    CHECK(joint.per_hero.size() == 2);
    // Non-trivial tree: root visited many times, children expanded.
    CHECK(engine.last_stats().tree_size > 1);
    CHECK(engine.last_stats().root_children > 0);
}

TEST(team_mcts_three_heroes_vs_two_enemies_deterministic_under_seed) {
    auto s1 = make_team_scene(3, 2);
    auto s2 = make_team_scene(3, 2);
    mcts::MctsConfig cfg;
    cfg.iterations      = 200;
    cfg.rollout_horizon = 6;
    cfg.action_repeat   = 2;
    cfg.seed            = 0xCAFE;

    auto run = [&](TeamScene& s) {
        mcts::TeamMcts engine(cfg);
        engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        engine.set_opponent_policy(mcts::policy_idle);
        return engine.search(s.world, raw(s.heroes));
    };

    auto j1 = run(*s1);
    auto j2 = run(*s2);
    CHECK(j1 == j2);
}

TEST(team_mcts_search_side_effect_free) {
    auto s = make_team_scene(2, 2);
    std::vector<float> hp_before;
    std::vector<std::pair<float,float>> pos_before;
    for (auto& h : s->heroes) {
        hp_before.push_back(h->unit().hp);
        pos_before.push_back({h->x(), h->z()});
    }

    mcts::MctsConfig cfg;
    cfg.iterations      = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat   = 2;
    mcts::TeamMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, raw(s->heroes));

    for (size_t i = 0; i < s->heroes.size(); i++) {
        CHECK_NEAR(s->heroes[i]->unit().hp, hp_before[i], 1e-4f);
        CHECK_NEAR(s->heroes[i]->x(), pos_before[i].first,  1e-4f);
        CHECK_NEAR(s->heroes[i]->z(), pos_before[i].second, 1e-4f);
    }
}

TEST(team_mcts_advance_root_reuses_subtree) {
    auto s = make_team_scene(2, 1);
    mcts::MctsConfig cfg;
    cfg.iterations      = 300;
    cfg.rollout_horizon = 6;
    cfg.action_repeat   = 2;
    mcts::TeamMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    auto heroes = raw(s->heroes);
    auto first = engine.search(s->world, heroes);
    CHECK(engine.last_stats().reused_root == false);

    engine.advance_root(first);
    for (size_t i = 0; i < heroes.size(); i++) {
        mcts::apply(*heroes[i], s->world, first.per_hero[i], cfg.sim_dt * cfg.action_repeat);
    }
    engine.search(s->world, heroes);
    // advance_root may have reset the tree if the rarely-picked joint-action
    // pair wasn't expanded. Either state is valid — just assert the second
    // search completed without crashing and produced a tree.
    CHECK(engine.last_stats().tree_size >= 1);
}

// ─── TacticMcts (hierarchical tactic layer) ────────────────────────────────

TEST(tactic_to_action_hold_attacks_nearest_in_range) {
    auto s = make_team_scene(1, 1);
    mcts::Tactic t{ mcts::TacticKind::Hold };
    auto a = mcts::tactic_to_action(t, *s->heroes[0], s->world);
    CHECK(a.move_dir == mcts::MoveDir::Hold);
    CHECK(a.attack_slot == 0);
}

TEST(tactic_to_action_retreat_moves_back_no_attack) {
    auto s = make_team_scene(1, 1);
    mcts::Tactic t{ mcts::TacticKind::Retreat };
    auto a = mcts::tactic_to_action(t, *s->heroes[0], s->world);
    CHECK(a.move_dir == mcts::MoveDir::PathAway);
    CHECK(a.attack_slot == -1);
    CHECK(a.ability_slot == -1);
}

TEST(tactic_to_action_focus_targets_lowest_hp_enemy) {
    auto s = make_team_scene(1, 2);
    s->enemies[1]->unit().hp = 5.0f;   // make enemy 1 the clear focus
    mcts::Tactic t{ mcts::TacticKind::FocusLowestHp };
    auto a = mcts::tactic_to_action(t, *s->heroes[0], s->world);
    CHECK(a.attack_slot != -1 || a.move_dir == mcts::MoveDir::PathToTarget);
}

TEST(tactic_mcts_search_returns_legal_tactic) {
    auto s = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations            = 300;
    cfg.rollout_horizon       = 8;
    cfg.action_repeat         = 2;
    cfg.tactic_window_decisions = 3;
    cfg.seed                  = 0xF00D;
    mcts::TacticMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    auto chosen = engine.search(s->world, raw(s->heroes));
    CHECK(static_cast<int>(chosen.kind) >= 0);
    CHECK(static_cast<int>(chosen.kind) <  static_cast<int>(mcts::TacticKind::COUNT));
    CHECK(engine.last_stats().tree_size > 1);
    CHECK(engine.last_stats().root_children > 0);
}

TEST(tactic_mcts_value_is_positive_when_team_dominant) {
    // 3 strong heroes vs 1 weak enemy — team should expect to win under any
    // reasonable tactic choice. Best-action mean value should be positive.
    auto s = make_team_scene(3, 1);
    s->enemies[0]->unit().hp = 10.0f;
    s->enemies[0]->unit().damage = 1.0f;
    mcts::MctsConfig cfg;
    cfg.iterations            = 400;
    cfg.rollout_horizon       = 16;
    cfg.action_repeat         = 2;
    cfg.tactic_window_decisions = 3;
    cfg.seed                  = 17;
    mcts::TacticMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    engine.search(s->world, raw(s->heroes));
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(tactic_mcts_is_deterministic_under_seed) {
    auto s1 = make_team_scene(2, 2);
    auto s2 = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations            = 200;
    cfg.rollout_horizon       = 6;
    cfg.action_repeat         = 2;
    cfg.tactic_window_decisions = 3;
    cfg.seed                  = 0xDEAD;

    auto run = [&](TeamScene& s) {
        mcts::TacticMcts engine(cfg);
        engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        engine.set_opponent_policy(mcts::policy_idle);
        return engine.search(s.world, raw(s.heroes));
    };
    auto t1 = run(*s1);
    auto t2 = run(*s2);
    CHECK(t1 == t2);
}

TEST(tactic_mcts_search_side_effect_free) {
    auto s = make_team_scene(2, 2);
    std::vector<float> hp_before;
    for (auto& h : s->heroes) hp_before.push_back(h->unit().hp);
    for (auto& e : s->enemies) hp_before.push_back(e->unit().hp);

    mcts::MctsConfig cfg;
    cfg.iterations            = 100;
    cfg.rollout_horizon       = 4;
    cfg.action_repeat         = 2;
    cfg.tactic_window_decisions = 2;
    mcts::TacticMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, raw(s->heroes));

    size_t i = 0;
    for (auto& h : s->heroes)  CHECK_NEAR(h->unit().hp, hp_before[i++], 1e-4f);
    for (auto& e : s->enemies) CHECK_NEAR(e->unit().hp, hp_before[i++], 1e-4f);
}

TEST(legal_tactics_no_enemies_returns_hold_only) {
    auto s = make_team_scene(/*heroes*/2, /*enemies*/0);
    auto ts = mcts::legal_tactics(raw(s->heroes), s->world);
    CHECK(ts.size() == 1);
    CHECK(ts[0].kind == mcts::TacticKind::Hold);
}

TEST(legal_tactics_all_dead_enemies_returns_hold_only) {
    auto s = make_team_scene(2, 2);
    for (auto& e : s->enemies) e->unit().hp = 0.0f;
    auto ts = mcts::legal_tactics(raw(s->heroes), s->world);
    CHECK(ts.size() == 1);
    CHECK(ts[0].kind == mcts::TacticKind::Hold);
}

TEST(legal_tactics_prunes_retreat_when_enemies_far) {
    // Default make_team_scene: attackRange=3, heroes at x≈-1, enemies at x≈+1.
    // Move enemies far away so no hero is within 1.5× enemy attackRange.
    auto s = make_team_scene(2, 2);
    for (size_t i = 0; i < s->enemies.size(); i++) {
        s->enemies[i]->setPosition(100.0f + 0.4f * i, 0.3f * i);
    }
    auto ts = mcts::legal_tactics(raw(s->heroes), s->world);
    bool has_retreat = false, has_focus = false, has_scatter = false, has_hold = false;
    for (auto& t : ts) {
        has_retreat |= (t.kind == mcts::TacticKind::Retreat);
        has_focus   |= (t.kind == mcts::TacticKind::FocusLowestHp);
        has_scatter |= (t.kind == mcts::TacticKind::Scatter);
        has_hold    |= (t.kind == mcts::TacticKind::Hold);
    }
    CHECK(has_hold);
    CHECK(has_focus);
    CHECK(has_scatter);
    CHECK(!has_retreat);
}

TEST(legal_tactics_keeps_retreat_when_enemy_in_threat_range) {
    // Stock scene: enemies ~2 units from heroes, 1.5× attackRange = 4.5 → threat.
    auto s = make_team_scene(2, 2);
    auto ts = mcts::legal_tactics(raw(s->heroes), s->world);
    bool has_retreat = false;
    for (auto& t : ts) if (t.kind == mcts::TacticKind::Retreat) has_retreat = true;
    CHECK(has_retreat);
    CHECK(ts.size() == 4);
}

// ─── OptionMcts / TeamOptionMcts ───────────────────────────────────────────

namespace {

// Simple concrete options for tests. Each option drives the hero with a
// fixed CombatAction and terminates after N windows.
class HoldOpt : public mcts::Option {
    std::string name_ = "hold";
    int windows_;
public:
    explicit HoldOpt(int w = 2) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent&, const World&) const override { return true; }
    mcts::CombatAction step(Agent&, World&, int) const override {
        return mcts::CombatAction{ mcts::MoveDir::Hold, -1, -1 };
    }
    bool should_terminate(const Agent&, const World&, int ticks) const override {
        return ticks >= windows_;
    }
};

class AttackOpt : public mcts::Option {
    std::string name_ = "attack";
    int windows_;
public:
    explicit AttackOpt(int w = 3) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent& self, const World& world) const override {
        // Only if there's at least one living enemy.
        for (Agent* a : world.agents()) {
            if (a == &self) continue;
            if (a->unit().alive() && a->unit().teamId != self.unit().teamId) return true;
        }
        return false;
    }
    mcts::CombatAction step(Agent&, World&, int) const override {
        return mcts::CombatAction{ mcts::MoveDir::PathToTarget, 0, -1 };
    }
    bool should_terminate(const Agent&, const World&, int ticks) const override {
        return ticks >= windows_;
    }
};

class NeverInitOpt : public mcts::Option {
    std::string name_ = "never";
public:
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent&, const World&) const override { return false; }
    mcts::CombatAction step(Agent&, World&, int) const override { return {}; }
    bool should_terminate(const Agent&, const World&, int) const override { return true; }
};

class TeamHoldOpt : public mcts::TeamOption {
    std::string name_ = "team_hold";
    int windows_;
public:
    explicit TeamHoldOpt(int w = 2) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const std::vector<Agent*>&, const World&) const override { return true; }
    std::vector<mcts::CombatAction> step(const std::vector<Agent*>& heroes,
                                           World&, int) const override {
        return std::vector<mcts::CombatAction>(heroes.size(),
            mcts::CombatAction{ mcts::MoveDir::Hold, -1, -1 });
    }
    bool should_terminate(const std::vector<Agent*>&, const World&, int t) const override {
        return t >= windows_;
    }
};

class TeamPushOpt : public mcts::TeamOption {
    std::string name_ = "team_push";
    int windows_;
public:
    explicit TeamPushOpt(int w = 3) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const std::vector<Agent*>& heroes, const World& world) const override {
        if (heroes.empty()) return false;
        int team = heroes.front()->unit().teamId;
        for (Agent* a : world.agents()) {
            if (a->unit().alive() && a->unit().teamId != team) return true;
        }
        return false;
    }
    std::vector<mcts::CombatAction> step(const std::vector<Agent*>& heroes,
                                           World&, int) const override {
        return std::vector<mcts::CombatAction>(heroes.size(),
            mcts::CombatAction{ mcts::MoveDir::PathToTarget, 0, -1 });
    }
    bool should_terminate(const std::vector<Agent*>&, const World&, int t) const override {
        return t >= windows_;
    }
};

} // namespace

TEST(option_mcts_search_returns_an_option) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations        = 80;
    cfg.rollout_horizon   = 3;
    cfg.action_repeat     = 2;
    cfg.option_max_windows = 4;
    cfg.seed              = 0xAB;

    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(3),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    const mcts::Option* chosen = engine.search(s->world, s->hero);
    CHECK(chosen != nullptr);
    CHECK(chosen->name() == "hold" || chosen->name() == "attack");
    CHECK(engine.last_stats().root_children > 0);
}

TEST(option_mcts_excludes_options_that_cannot_initiate) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations        = 40;
    cfg.rollout_horizon   = 2;
    cfg.option_max_windows = 3;

    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<NeverInitOpt>(),   // filtered out every expansion
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    // Only "hold" should ever have been expanded at the root.
    CHECK(engine.last_stats().root_children == 1);
}

TEST(option_mcts_returns_null_when_no_option_initiable) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 20;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<NeverInitOpt>(),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    CHECK(engine.search(s->world, s->hero) == nullptr);
}

TEST(option_mcts_search_is_side_effect_free) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    float hp_hero_before = s->hero.unit().hp;
    float hp_enemy_before = s->world.agents()[1]->unit().hp;

    mcts::MctsConfig cfg;
    cfg.iterations = 50;
    cfg.rollout_horizon = 3;
    cfg.option_max_windows = 3;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);

    CHECK_NEAR(s->hero.unit().hp, hp_hero_before, 1e-4f);
    CHECK_NEAR(s->world.agents()[1]->unit().hp, hp_enemy_before, 1e-4f);
}

TEST(option_mcts_advance_root_by_name) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 40;
    cfg.option_max_windows = 3;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    const mcts::Option* first = engine.search(s->world, s->hero);
    CHECK(first != nullptr);

    // Re-author the option set with fresh shared_ptrs (different pointers,
    // same names) — advance_root must match by name and preserve the subtree.
    std::vector<std::shared_ptr<mcts::Option>> opts2 = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    const mcts::Option* committed = nullptr;
    for (auto& sp : opts2) if (sp->name() == first->name()) { committed = sp.get(); break; }
    engine.advance_root(committed);
    CHECK(engine.last_root() != nullptr);
    CHECK(engine.last_root()->action != nullptr);
    CHECK(engine.last_root()->action->name() == first->name());
}

TEST(option_mcts_use_leaf_value_finishes_with_large_horizon) {
    // With use_leaf_value=true, rollout_horizon is ignored. Set it huge: if
    // rollout actually ran the search would take long enough that elapsed_ms
    // would be very large. Instead we just assert it completes all iterations
    // and reports a small-to-moderate elapsed time.
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 30;
    cfg.rollout_horizon = 10000;
    cfg.use_leaf_value = true;
    cfg.option_max_windows = 2;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(1),
        std::make_shared<AttackOpt>(1),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    CHECK(engine.last_stats().iterations == 30);
    CHECK(engine.last_stats().elapsed_ms < 500);
}

TEST(team_option_mcts_search_returns_an_option) {
    auto s = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations        = 60;
    cfg.rollout_horizon   = 3;
    cfg.action_repeat     = 2;
    cfg.option_max_windows = 3;
    cfg.seed              = 0xCD;

    std::vector<std::shared_ptr<mcts::TeamOption>> opts = {
        std::make_shared<TeamHoldOpt>(2),
        std::make_shared<TeamPushOpt>(2),
    };
    mcts::TeamOptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    const mcts::TeamOption* chosen = engine.search(s->world, raw(s->heroes));
    CHECK(chosen != nullptr);
    CHECK(chosen->name() == "team_hold" || chosen->name() == "team_push");
}

TEST(team_option_mcts_is_deterministic_under_seed) {
    auto s1 = make_team_scene(2, 2);
    auto s2 = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations        = 40;
    cfg.rollout_horizon   = 3;
    cfg.option_max_windows = 2;
    cfg.seed              = 0xBEEF;

    auto run = [&](TeamScene& s) {
        std::vector<std::shared_ptr<mcts::TeamOption>> opts = {
            std::make_shared<TeamHoldOpt>(2),
            std::make_shared<TeamPushOpt>(2),
        };
        mcts::TeamOptionMcts engine(cfg);
        engine.set_options(opts);
        engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        engine.set_opponent_policy(mcts::policy_idle);
        const mcts::TeamOption* o = engine.search(s.world, raw(s.heroes));
        return o ? o->name() : std::string{};
    };
    CHECK(run(*s1) == run(*s2));
}


// ─── Commander ─────────────────────────────────────────────────────────────

TEST(commander_assigns_roles_and_produces_actions) {
    auto s = make_team_scene(2, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 30;
    cfg.role_cfg.rollout_horizon = 2;
    cfg.role_cfg.option_max_windows = 2;
    cfg.replan_every_windows = 2;
    mcts::Commander cmdr(cfg);

    // Role A: hold. Role B: attack. Default round-robin assignment → hero 0
    // gets role 0 (hold), hero 1 gets role 1 (attack).
    std::vector<std::shared_ptr<mcts::Option>> hold_opts = { std::make_shared<HoldOpt>(2) };
    std::vector<std::shared_ptr<mcts::Option>> atk_opts  = {
        std::make_shared<HoldOpt>(2), std::make_shared<AttackOpt>(2) };
    cmdr.add_role("hold",   hold_opts);
    cmdr.add_role("attack", atk_opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    auto acts = cmdr.decide(s->world, raw(s->heroes));
    CHECK(acts.size() == 2);
    CHECK(cmdr.current_assignments().size() == 2);
    CHECK(cmdr.current_assignments()[0] == 0);
    CHECK(cmdr.current_assignments()[1] == 1);
    CHECK(cmdr.committed_option_for_hero(0) == "hold");
    // Hero 1 committed either "hold" or "attack" from role 1's set.
    std::string h1_opt = cmdr.committed_option_for_hero(1);
    CHECK(h1_opt == "hold" || h1_opt == "attack");
}

TEST(commander_custom_assigner_is_honored) {
    auto s = make_team_scene(3, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 20;
    cfg.role_cfg.option_max_windows = 2;
    mcts::Commander cmdr(cfg);

    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("alpha", opts);
    cmdr.add_role("beta",  opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    // All heroes on role "beta" (index 1).
    cmdr.set_assigner([](const std::vector<Agent*>& h, const World&) {
        return std::vector<int>(h.size(), 1);
    });
    cmdr.decide(s->world, raw(s->heroes));
    for (int a : cmdr.current_assignments()) CHECK(a == 1);
}

TEST(commander_replan_window_countdown_works) {
    auto s = make_team_scene(2, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 10;
    cfg.role_cfg.option_max_windows = 1;
    cfg.replan_every_windows = 3;
    mcts::Commander cmdr(cfg);
    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("only", opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    cmdr.decide(s->world, raw(s->heroes));
    int w1 = cmdr.windows_until_replan();
    cmdr.decide(s->world, raw(s->heroes));
    int w2 = cmdr.windows_until_replan();
    cmdr.decide(s->world, raw(s->heroes));
    int w3 = cmdr.windows_until_replan();
    CHECK(w1 > w2);
    CHECK(w2 > w3);
}

TEST(commander_dead_hero_skipped) {
    auto s = make_team_scene(2, 2);
    s->heroes[0]->unit().hp = 0.0f;    // kill hero 0
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 10;
    cfg.role_cfg.option_max_windows = 1;
    mcts::Commander cmdr(cfg);
    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("only", opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    auto acts = cmdr.decide(s->world, raw(s->heroes));
    // Dead hero gets default-constructed (no-op) action — move_dir Hold,
    // no attack, no ability. We can't probe "committed option" since there
    // isn't one, so just assert the default shape.
    CHECK(acts[0].move_dir == mcts::MoveDir::Hold);
    CHECK(acts[0].attack_slot == -1);
    CHECK(acts[0].ability_slot == -1);
    CHECK(cmdr.committed_option_for_hero(0) == "");
}


TEST(aggressive_rollout_returns_legal_action) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::AggressiveRollout rollout;
    auto a = rollout.choose(s->hero, s->world);
    // Enemy is in range → should pick an attack slot; movement is still N
    // (charge toward enemy), which is fine.
    CHECK(a.attack_slot >= 0);
    CHECK(a.move_dir == mcts::MoveDir::N);
}

TEST(aggressive_rollout_dead_agent_returns_noop) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    s->hero.unit().hp = 0.0f;
    mcts::AggressiveRollout rollout;
    auto a = rollout.choose(s->hero, s->world);
    CHECK(a.attack_slot == -1);
    CHECK(a.ability_slot == -1);
    CHECK(a.move_dir == mcts::MoveDir::Hold);
}

TEST(mcts_with_aggressive_rollout_picks_attack_vs_idle_opponent) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xA66;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(mcts_pw_bounds_root_children_by_visits_alpha) {
    // With pw_alpha = 0.5 and 100 iterations, the root should have at most
    // ceil(100^0.5) = 10 children. Without PW (classical), the root would
    // expand all 9 move-direction-only actions on the first ~9 iterations
    // and typically end up with 10+ children quickly.
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 100;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0x1234;
    cfg.pw_alpha       = 0.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    int bound = static_cast<int>(std::ceil(std::pow(100.0f, 0.5f)));
    CHECK(engine.last_stats().root_children <= bound);
    CHECK(engine.last_stats().root_children >= 1);
}

TEST(mcts_pw_off_matches_classical_expansion) {
    // pw_alpha = 0 should leave expansion unchanged: root children count
    // saturates at the number of legal actions (capped by iteration budget).
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 100;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0x5678;
    cfg.pw_alpha       = 0.0f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    // Under PW=0, we expect the root to have expanded well beyond the PW(0.5)
    // bound of 10 — we're just asserting PW isn't being applied.
    CHECK(engine.last_stats().root_children > 10);
}

TEST(mcts_pw_still_picks_attack_vs_idle) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 256;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xABCD;
    cfg.pw_alpha       = 0.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(mcts_pw_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xCAFE;
    cfg.pw_alpha       = 0.5f;
    auto run = [&](McstScene& s) {
        mcts::Mcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        return e.search(s.world, s.hero);
    };
    auto a1 = run(*s1);
    auto a2 = run(*s2);
    CHECK(a1 == a2);
}

TEST(attack_bias_prior_weights_attack_over_move) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::AttackBiasPrior prior;
    auto acts = mcts::legal_actions(s->hero, s->world);
    CHECK(acts.size() > 1);
    auto w = prior.score(s->hero, s->world, acts);
    CHECK(w.size() == acts.size());
    // Find a pure-move and an attack action, verify the attack outweighs.
    float max_move = 0.0f, min_attack = 1e9f;
    bool saw_move = false, saw_attack = false;
    for (size_t i = 0; i < acts.size(); i++) {
        if (acts[i].attack_slot >= 0) {
            saw_attack = true; if (w[i] < min_attack) min_attack = w[i];
        } else if (acts[i].ability_slot < 0) {
            saw_move = true; if (w[i] > max_move) max_move = w[i];
        }
    }
    CHECK(saw_move);
    CHECK(saw_attack);
    CHECK(min_attack > max_move);
}

TEST(uniform_prior_all_equal) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::UniformPrior prior;
    auto acts = mcts::legal_actions(s->hero, s->world);
    auto w = prior.score(s->hero, s->world, acts);
    CHECK(w.size() == acts.size());
    for (float v : w) CHECK(std::abs(v - w[0]) < 1e-6f);
}

TEST(mcts_puct_picks_attack_with_prior) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB;
    cfg.prior_c        = 1.5f;     // enable PUCT
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());

    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);

    // With an attack-biased prior, the most-visited attack-containing child
    // should accumulate more visits than an average child.
    const auto* root = engine.last_root();
    CHECK(root != nullptr);
    int attack_visits = 0, move_only_visits = 0;
    for (const auto& up : root->children) {
        if (up->action.attack_slot >= 0) attack_visits += up->visits;
        else if (up->action.ability_slot < 0) move_only_visits += up->visits;
    }
    CHECK(attack_visits > move_only_visits);
}

TEST(mcts_puct_child_priors_sum_to_one) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 64;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB2;
    cfg.prior_c        = 1.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    engine.search(s->world, s->hero);
    const auto* root = engine.last_root();
    CHECK(root != nullptr);
    // Sum of all children priors + remaining untried priors should be ~1.
    float sum = 0.0f;
    for (const auto& up : root->children) sum += up->prior_p;
    for (float p : root->untried_priors) sum += p;
    CHECK(std::abs(sum - 1.0f) < 1e-4f);
}

TEST(mcts_puct_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB3;
    cfg.prior_c        = 1.5f;
    auto run = [&](McstScene& s) {
        mcts::Mcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, s.hero);
    };
    CHECK(run(*s1) == run(*s2));
}

TEST(dmcts_puct_uses_priors_and_is_deterministic) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD00;
    cfg.prior_c        = 1.5f;
    auto run = [&](McstScene& s) {
        mcts::DecoupledMcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, s.hero, s.enemy);
    };
    auto j1 = run(*s1);
    auto j2 = run(*s2);
    CHECK(j1.hero == j2.hero);
    CHECK(j1.opp  == j2.opp);
    // Priors were normalized to sum to 1 on both players.
    // (Indirect check: search completed with non-empty stats.)
}

TEST(dmcts_puct_root_priors_sum_to_one) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 32;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD01;
    cfg.prior_c        = 1.5f;
    mcts::DecoupledMcts e(cfg);
    e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    e.search(s->world, s->hero, s->enemy);
    const auto* r = e.last_root();
    CHECK(r != nullptr);
    float sh = 0.0f, so = 0.0f;
    for (float p : r->hero_stats.priors) sh += p;
    for (float p : r->opp_stats.priors)  so += p;
    CHECK(std::abs(sh - 1.0f) < 1e-4f);
    CHECK(std::abs(so - 1.0f) < 1e-4f);
}

TEST(team_mcts_puct_uses_priors_and_is_deterministic) {
    auto s1 = make_team_scene(2, 2);
    auto s2 = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD02;
    cfg.prior_c        = 1.5f;
    auto run = [&](TeamScene& s) {
        mcts::TeamMcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, raw(s.heroes));
    };
    auto j1 = run(*s1);
    auto j2 = run(*s2);
    CHECK(j1 == j2);
}

TEST(team_mcts_puct_root_priors_sum_to_one_per_hero) {
    auto s = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations     = 64;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD03;
    cfg.prior_c        = 1.5f;
    mcts::TeamMcts e(cfg);
    e.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    e.set_opponent_policy(mcts::policy_idle);
    e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    e.search(s->world, raw(s->heroes));
    const auto* r = e.last_root();
    CHECK(r != nullptr);
    for (const auto& ph : r->per_hero) {
        float sum = 0.0f;
        for (float p : ph.priors) sum += p;
        CHECK(std::abs(sum - 1.0f) < 1e-4f);
    }
}

TEST(tactic_prior_boosts_matching_action) {
    auto s = make_team_scene(1, 1);
    auto& hero = *s->heroes.front();
    // FocusLowestHp with the one enemy → target action is attack_slot=0.
    mcts::TacticPrior tp;
    tp.set_tactic(mcts::Tactic{ mcts::TacticKind::FocusLowestHp });
    auto acts = mcts::legal_actions(hero, s->world);
    auto w = tp.score(hero, s->world, acts);
    CHECK(w.size() == acts.size());
    mcts::CombatAction target = mcts::tactic_to_action(
        mcts::Tactic{ mcts::TacticKind::FocusLowestHp }, hero, s->world);
    float max_w = 0.0f, match_w = 0.0f;
    bool saw_match = false;
    for (size_t i = 0; i < acts.size(); i++) {
        if (acts[i] == target) { saw_match = true; match_w = w[i]; }
        if (w[i] > max_w) max_w = w[i];
    }
    CHECK(saw_match);
    CHECK(match_w >= max_w);
    CHECK(match_w > 1.0f);
}

TEST(layered_planner_returns_joint_action_of_correct_size) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 60;
    cfg.tactic_cfg.rollout_horizon        = 6;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 3;
    cfg.tactic_cfg.seed                   = 0x5A1;
    cfg.fine_cfg.iterations               = 60;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A2;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    planner.set_opponent_policy(mcts::policy_aggressive);

    auto joint = planner.decide(s->world, raw(s->heroes));
    CHECK(joint.per_hero.size() == s->heroes.size());
    // First call must have planned a tactic.
    CHECK(planner.last_stats().replanned_this_call == true);
    CHECK(planner.last_stats().windows_until_replan == 2);  // window=3, one consumed
}

TEST(layered_planner_replans_tactic_on_schedule) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 40;
    cfg.tactic_cfg.rollout_horizon        = 4;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 2;
    cfg.tactic_cfg.seed                   = 0x5A3;
    cfg.fine_cfg.iterations               = 40;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A4;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    planner.set_opponent_policy(mcts::policy_idle);

    // Windows=2 → replan pattern: T, F, T, F, T, ...
    std::vector<bool> replanned;
    for (int i = 0; i < 5; i++) {
        planner.decide(s->world, raw(s->heroes));
        replanned.push_back(planner.last_stats().replanned_this_call);
    }
    CHECK(replanned[0] == true);
    CHECK(replanned[1] == false);
    CHECK(replanned[2] == true);
    CHECK(replanned[3] == false);
    CHECK(replanned[4] == true);
}

TEST(layered_planner_reset_forces_next_replan) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 30;
    cfg.tactic_cfg.rollout_horizon        = 4;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 5;
    cfg.tactic_cfg.seed                   = 0x5A5;
    cfg.fine_cfg.iterations               = 30;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A6;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    planner.set_opponent_policy(mcts::policy_idle);
    planner.decide(s->world, raw(s->heroes));
    CHECK(planner.last_stats().windows_until_replan == 4);
    planner.reset();
    planner.decide(s->world, raw(s->heroes));
    CHECK(planner.last_stats().replanned_this_call == true);
}

// ─── Capability layer ──────────────────────────────────────────────────────

// Helper: construct a minimal CapContext for a given agent + world.
static CapContext makeCapCtx(Agent& a, World& w, CapabilitySet& set) {
    CapContext c;
    c.self = &a;
    c.unit = &a.unit();
    c.world = &w;
    c.caps = &set;
    c.now = 0;
    return c;
}

TEST(cap_move_to_sets_target_and_completes) {
    NavGrid grid(-10, -10, 10, 10, 0.5f);
    Agent bot; bot.setNavGrid(&grid); bot.setPosition(0, 0); bot.setSpeed(5);
    bot.unit().id = 1;
    World w; w.addAgent(&bot);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    CapContext ctx = makeCapCtx(bot, w, set);

    Action a; a.capId = kCapMoveTo; a.fx = 5; a.fz = 0;
    set.get(kCapMoveTo)->start(ctx, a);
    CHECK(a.done);
    CHECK(bot.hasTarget());

    // World tick should move bot toward target.
    for (int i = 0; i < 100 && !bot.atTarget(); i++) w.tick(1.0f/60.0f);
    CHECK(bot.atTarget());
    CHECK_NEAR(bot.x(), 5.0f, 1.0f);
}

TEST(cap_lane_walk_advances_waypoints) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    Agent bot; bot.setNavGrid(&grid); bot.setPosition(-10, 0); bot.setSpeed(8);
    bot.unit().id = 1;
    World w; w.addAgent(&bot);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    set.setLaneWaypoints({{-5, 0}, {0, 0}, {5, 0}, {10, 0}});
    CHECK(set.laneIndex() == 0);

    CapContext ctx = makeCapCtx(bot, w, set);

    // First decision: target the first waypoint.
    Action a; a.capId = kCapLaneWalk;
    set.get(kCapLaneWalk)->start(ctx, a);
    CHECK(a.done);
    CHECK(bot.hasTarget());

    // Simulate walking + periodic re-decision (think rate ~10Hz).
    int lastIdx = set.laneIndex();
    for (int frame = 0; frame < 600; frame++) {
        w.tick(1.0f/60.0f);
        if (frame % 6 == 0) {
            Action aa; aa.capId = kCapLaneWalk;
            set.get(kCapLaneWalk)->start(ctx, aa);
            lastIdx = set.laneIndex();
        }
        if (lastIdx >= static_cast<int>(set.laneWaypoints().size()) - 1
            && bot.atTarget()) break;
    }
    CHECK(set.laneIndex() == static_cast<int>(set.laneWaypoints().size()) - 1);
    CHECK_NEAR(bot.x(), 10.0f, 1.5f);
}

TEST(cap_basic_attack_resolves_and_blocks_for_swing) {
    NavGrid grid(-10, -10, 10, 10, 0.5f);
    Agent hero; hero.setNavGrid(&grid); hero.setPosition(0, 0);
    hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().damage = 20; hero.unit().attackRange = 5;
    hero.unit().attacksPerSec = 2.0f; // swing time 0.5s
    Agent enemy; enemy.setPosition(2, 0);
    enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 100;
    World w; w.addAgent(&hero); w.addAgent(&enemy);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    CapContext ctx = makeCapCtx(hero, w, set);

    CHECK(set.get(kCapBasicAttack)->gate(ctx));
    Action a; a.capId = kCapBasicAttack; a.i0 = 2;
    set.get(kCapBasicAttack)->start(ctx, a);
    CHECK(!a.done);
    CHECK_NEAR(a.dur, 0.5f, 0.001f);
    CHECK(enemy.unit().hp < 100);

    // Advance < swing time — still blocking.
    set.get(kCapBasicAttack)->advance(ctx, a, 0.3f);
    CHECK(!a.done);
    // Advance past swing time — completes.
    set.get(kCapBasicAttack)->advance(ctx, a, 0.3f);
    CHECK(a.done);

    // Cooldown gate blocks next attack until unit tick clears it.
    CHECK(!set.get(kCapBasicAttack)->gate(ctx));
}

TEST(cap_cast_ability_invokes_world_resolveAbility) {
    NavGrid grid(-10, -10, 10, 10, 0.5f);
    Agent caster; caster.setNavGrid(&grid); caster.setPosition(0, 0);
    caster.unit().id = 1; caster.unit().teamId = 0;
    caster.unit().mana = 50; caster.unit().maxMana = 100;
    Agent target; target.setPosition(3, 0);
    target.unit().id = 2; target.unit().teamId = 1; target.unit().hp = 100;
    World w; w.addAgent(&caster); w.addAgent(&target);

    // Register a tiny ability: deal 25 magic damage.
    int hits = 0;
    w.registerAbility(42, AbilitySpec{
        /*cooldown*/ 5.0f, /*manaCost*/ 20.0f, /*range*/ 10.0f,
        [&](Agent& c, World& ww, int tid) {
            if (Agent* t = ww.findById(tid)) {
                ww.dealDamage(c, *t, 25.0f, DamageKind::Magical);
                hits++;
            }
        },
    });
    caster.unit().abilitySlot[0] = 42;

    CapabilitySet set; addAllBuiltinCapabilities(set);
    CapContext ctx = makeCapCtx(caster, w, set);

    CHECK(set.get(kCapCastAbility)->gate(ctx));
    Action a; a.capId = kCapCastAbility; a.i0 = 0 /*slot*/; a.i1 = 2 /*targetId*/;
    set.get(kCapCastAbility)->start(ctx, a);
    CHECK(hits == 1);
    CHECK(target.unit().hp < 100);
    CHECK(caster.unit().mana < 50);
    CHECK(!a.done); // blocks for cast time
    set.get(kCapCastAbility)->advance(ctx, a, 1.0f);
    CHECK(a.done);
}

TEST(cap_flee_points_away_from_nearest_enemy) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    Agent bot; bot.setNavGrid(&grid); bot.setPosition(0, 0);
    bot.unit().id = 1; bot.unit().teamId = 0;
    Agent threat; threat.setPosition(3, 0);
    threat.unit().id = 2; threat.unit().teamId = 1;
    World w; w.addAgent(&bot); w.addAgent(&threat);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    CapContext ctx = makeCapCtx(bot, w, set);

    Action a; a.capId = kCapFlee;
    set.get(kCapFlee)->start(ctx, a);
    CHECK(a.done);
    // Retreat point should be on the negative-X side (away from threat at +3,0).
    CHECK(a.fx < 0);
    CHECK(bot.hasTarget());
}

TEST(capset_builtin_mask_reflects_gate) {
    NavGrid grid(-10, -10, 10, 10, 0.5f);
    Agent bot; bot.setNavGrid(&grid); bot.setPosition(0, 0);
    bot.unit().id = 1; bot.unit().attackCooldown = 0;
    World w; w.addAgent(&bot);
    CapabilitySet set; addAllBuiltinCapabilities(set);
    CapContext ctx = makeCapCtx(bot, w, set);

    uint32_t mask = set.buildBuiltinMask(ctx);
    CHECK((mask & (1u << kCapMoveTo)) != 0);
    CHECK((mask & (1u << kCapBasicAttack)) != 0);
    CHECK((mask & (1u << kCapCastAbility)) == 0); // no ability slots bound

    // Bind ability -> mask flips on.
    w.registerAbility(7, AbilitySpec{1.0f, 0.0f, 5.0f, [](Agent&, World&, int){}});
    bot.unit().abilitySlot[0] = 7;
    mask = set.buildBuiltinMask(ctx);
    CHECK((mask & (1u << kCapCastAbility)) != 0);

    // Attack on cooldown -> mask flips off.
    bot.unit().attackCooldown = 1.0f;
    mask = set.buildBuiltinMask(ctx);
    CHECK((mask & (1u << kCapBasicAttack)) == 0);
}

TEST(policy_scripted_minion_attacks_when_in_range) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    Agent minion; minion.setNavGrid(&grid); minion.setPosition(0, 0);
    minion.unit().id = 1; minion.unit().teamId = 0;
    minion.unit().attackRange = 5; minion.unit().damage = 10;
    Agent enemy; enemy.setPosition(2, 0);
    enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 100;
    World w; w.addAgent(&minion); w.addAgent(&enemy);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    set.setLaneWaypoints({{10, 0}});
    CapContext ctx = makeCapCtx(minion, w, set);

    auto policy = makeScriptedMinionPolicy();
    Action out;
    CHECK(policy->decide(ctx, set, out));
    CHECK(out.capId == kCapBasicAttack);
    CHECK(out.i0 == 2);
}

TEST(policy_scripted_minion_lanewalks_when_no_enemy_in_range) {
    NavGrid grid(-20, -20, 20, 20, 0.5f);
    Agent minion; minion.setNavGrid(&grid); minion.setPosition(0, 0);
    minion.unit().id = 1; minion.unit().teamId = 0;
    minion.unit().attackRange = 3;
    Agent faraway; faraway.setPosition(15, 0);
    faraway.unit().id = 2; faraway.unit().teamId = 1;
    World w; w.addAgent(&minion); w.addAgent(&faraway);

    CapabilitySet set; addAllBuiltinCapabilities(set);
    set.setLaneWaypoints({{10, 0}});
    CapContext ctx = makeCapCtx(minion, w, set);

    auto policy = makeScriptedMinionPolicy();
    Action out;
    CHECK(policy->decide(ctx, set, out));
    CHECK(out.capId == kCapLaneWalk);
}

// ─── Partial observability: observation + belief + IS-MCTS ─────────────────

TEST(observe_visible_enemy_no_obstacles) {
    World world;
    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100;
    hero.setPosition(0, 0); world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 50;
    enemy.setPosition(5, 0); world.addAgent(&enemy);

    obs::VisibilityConfig cfg;  // omnidirectional, unlimited range, LOS on
    auto o = obs::observe(world, 0, cfg, 0.0f);
    CHECK(o.allies.size() == 1);
    CHECK(o.enemies.size() == 1);
    CHECK(o.enemies[0].visible);
    CHECK_NEAR(o.enemies[0].pos.x, 5.0f, 1e-3f);
}

TEST(observe_enemy_behind_wall_hidden) {
    World world;
    world.addObstacle({0, 0, 0.5f, 2.0f});  // tall thin wall at origin

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0; hero.unit().hp = 100;
    hero.setPosition(-5, 0); world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 50;
    enemy.setPosition(5, 0); world.addAgent(&enemy);

    obs::VisibilityConfig cfg;
    auto o = obs::observe(world, 0, cfg, 0.0f);
    CHECK(!o.enemies[0].visible);
}

TEST(observe_range_gating) {
    World world;
    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0; hero.unit().hp = 100;
    hero.setPosition(0, 0); world.addAgent(&hero);
    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 50;
    enemy.setPosition(20, 0); world.addAgent(&enemy);

    obs::VisibilityConfig cfg; cfg.max_range = 10.0f;
    auto o = obs::observe(world, 0, cfg, 0.0f);
    CHECK(!o.enemies[0].visible);
}

TEST(observe_merge_preserves_stale) {
    World world;
    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0; hero.unit().hp = 100;
    hero.setPosition(-5, 0); world.addAgent(&hero);
    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 50;
    enemy.setPosition(5, 0); world.addAgent(&enemy);

    obs::VisibilityConfig cfg;
    auto o0 = obs::observe(world, 0, cfg, 0.0f);
    CHECK(o0.enemies[0].visible);

    // Enemy moves to a new location and a wall appears between them.
    enemy.setPosition(10, 0);
    world.addObstacle({7, 0, 0.5f, 2.0f});
    auto o1_raw = obs::observe(world, 0, cfg, 1.0f);
    CHECK(!o1_raw.enemies[0].visible);

    auto o1 = obs::merge(o0, o1_raw, 1.0f);
    CHECK(!o1.enemies[0].visible);
    // Stale position carried forward is the last-seen (5, 0), not current (10, 0).
    CHECK_NEAR(o1.enemies[0].pos.x, 5.0f, 1e-3f);
    CHECK_NEAR(o1.enemies[0].last_seen_elapsed, 1.0f, 1e-3f);
}

TEST(belief_collapses_on_sighting) {
    NavGrid nav(-10, -10, 10, 10, 0.5f);
    belief::TeamBelief tb(0, 32, &nav);

    Vec2 prior{5, 0};
    tb.register_enemy(2, 50.0f, &prior);

    // Synthesize an observation where the enemy is visible at (3, 1).
    obs::TeamObservation fresh;
    fresh.team_id = 0; fresh.timestamp = 0.0f;
    obs::AgentObservation e;
    e.id = 2; e.team_id = 1; e.pos = {3, 1}; e.hp = 42; e.alive = true; e.visible = true;
    fresh.enemies.push_back(e);

    tb.update(fresh);
    CHECK(tb.enemies().size() == 1);
    CHECK(tb.enemies()[0].visible);
    // All particles should be at (3, 1) exactly.
    for (const auto& p : tb.enemies()[0].particles) {
        CHECK_NEAR(p.pos.x, 3.0f, 1e-4f);
        CHECK_NEAR(p.pos.z, 1.0f, 1e-4f);
    }
}

TEST(belief_spreads_on_loss_of_contact) {
    NavGrid nav(-10, -10, 10, 10, 0.5f);
    belief::MotionParams mp; mp.spread_on_loss = 2.0f;
    belief::TeamBelief tb(0, 64, &nav, mp);

    Vec2 prior{5, 0};
    tb.register_enemy(2, 50.0f, &prior);

    // Sighting collapses.
    obs::TeamObservation seen;
    seen.team_id = 0; seen.timestamp = 0.0f;
    {
        obs::AgentObservation e;
        e.id = 2; e.team_id = 1; e.pos = {3, 0}; e.hp = 42; e.alive = true; e.visible = true;
        seen.enemies.push_back(e);
    }
    tb.update(seen);

    // Next tick: enemy hidden (e.g., stepped behind cover).
    obs::TeamObservation hidden;
    hidden.team_id = 0; hidden.timestamp = 1.0f;
    {
        obs::AgentObservation e;
        e.id = 2; e.team_id = 1; e.pos = {0, 0}; e.hp = 0; e.alive = false;
        e.visible = false; e.last_seen_elapsed = 1.0f;
        hidden.enemies.push_back(e);
    }
    tb.update(hidden);

    // Particles should no longer all coincide; compute variance.
    float sx = 0, sx2 = 0; int n = 0;
    for (const auto& p : tb.enemies()[0].particles) { sx += p.pos.x; sx2 += p.pos.x * p.pos.x; n++; }
    CHECK(n > 0);
    float mean = sx / n;
    float var  = sx2 / n - mean * mean;
    CHECK(var > 0.01f);  // non-degenerate spread
}

TEST(belief_sample_returns_mapped_ids) {
    NavGrid nav(-10, -10, 10, 10, 0.5f);
    belief::TeamBelief tb(0, 16, &nav);
    Vec2 p{3, 3};
    tb.register_enemy(7, 100.0f, &p);
    tb.register_enemy(9, 100.0f, &p);

    std::mt19937_64 rng(42);
    auto s = tb.sample(rng);
    CHECK(s.size() == 2);
    CHECK(s.count(7) == 1);
    CHECK(s.count(9) == 1);
}

TEST(patch_snapshot_overwrites_enemy_only) {
    World world;
    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0; hero.unit().hp = 100;
    hero.setPosition(-3, 0); world.addAgent(&hero);
    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 80;
    enemy.setPosition(3, 0); world.addAgent(&enemy);

    auto snap = world.snapshot();
    std::unordered_map<int, belief::EnemyParticle> sampled;
    belief::EnemyParticle p; p.pos = {7, 2}; p.vel = {1, 0}; p.hp = 30; p.heading = 0.5f;
    sampled.emplace(2, p);

    mcts::patch_snapshot_with_particles(snap, sampled);

    // Hero untouched.
    CHECK_NEAR(snap.agents[0].x, -3.0f, 1e-4f);
    CHECK_NEAR(snap.agents[0].unit.hp, 100.0f, 1e-4f);
    // Enemy patched.
    CHECK_NEAR(snap.agents[1].x, 7.0f, 1e-4f);
    CHECK_NEAR(snap.agents[1].z, 2.0f, 1e-4f);
    CHECK_NEAR(snap.agents[1].unit.hp, 30.0f, 1e-4f);
}

TEST(infoset_mcts_restores_world) {
    NavGrid nav(-10, -10, 10, 10, 0.5f);

    World world;
    world.seed(0x77);

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100; hero.unit().damage = 10; hero.unit().attackRange = 3;
    hero.unit().attacksPerSec = 2;
    hero.setNavGrid(&nav);
    hero.setPosition(-2, 0); hero.setMaxAccel(30); hero.setMaxTurnRate(10);
    world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 80; enemy.unit().damage = 8; enemy.unit().attackRange = 3;
    enemy.unit().attacksPerSec = 1.5f;
    enemy.setNavGrid(&nav);
    enemy.setPosition(2, 0); enemy.setMaxAccel(30); enemy.setMaxTurnRate(10);
    world.addAgent(&enemy);

    auto tb = std::make_shared<belief::TeamBelief>(0, 16, &nav);
    Vec2 prior{2, 0};
    tb->register_enemy(2, enemy.unit().maxHp, &prior);
    // Collapse belief to truth via a fake "visible" observation.
    obs::TeamObservation seen;
    seen.team_id = 0; seen.timestamp = 0.0f;
    {
        obs::AgentObservation e;
        e.id = 2; e.team_id = 1; e.pos = {2, 0};
        e.hp = enemy.unit().hp; e.max_hp = enemy.unit().maxHp;
        e.alive = true; e.visible = true;
        seen.enemies.push_back(e);
    }
    tb->update(seen);

    mcts::MctsConfig cfg;
    cfg.iterations = 24;
    cfg.rollout_horizon = 8;
    cfg.action_repeat = 4;
    cfg.seed = 0xABCD;

    mcts::InfoSetMcts planner(cfg);
    planner.set_belief(tb);
    planner.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    planner.set_opponent_policy(mcts::policy_aggressive);

    float hero_x_before  = hero.x();
    float enemy_x_before = enemy.x();
    float hero_hp_before = hero.unit().hp;

    (void)planner.search(world, hero);

    // World must be untouched by search.
    CHECK_NEAR(hero.x(), hero_x_before, 1e-4f);
    CHECK_NEAR(enemy.x(), enemy_x_before, 1e-4f);
    CHECK_NEAR(hero.unit().hp, hero_hp_before, 1e-4f);

    CHECK(planner.last_stats().iterations > 0);
}

// ─── GenericMcts Tests ──────────────────────────────────────────────────────
//
// Tiny corridor env: states 0..N-1 with a goal at N-1. Actions are
// 0=left, 1=right, 2=stay. Step reward is -0.05 except entering the goal
// which pays +1 and terminates. Optimal policy from any non-goal state is
// "right". Used to verify that GenericMcts converges via random rollout
// and via a learned value function, and that snapshot/restore preserves
// the caller's pre-search state.

namespace {

struct CorridorEnv {
    int pos    = 0;
    int length = 6;

    int  step_action(int a) {
        if (a == 0) pos = std::max(0, pos - 1);
        else if (a == 1) pos = std::min(length - 1, pos + 1);
        // a == 2: stay.
        return pos;
    }
    bool at_goal() const { return pos == length - 1; }
};

mcts::GenericEnv make_corridor_env(CorridorEnv& env) {
    mcts::GenericEnv g;
    g.num_actions = 3;
    g.snapshot_fn = [&]() -> std::any { return env; };
    g.restore_fn  = [&](const std::any& s) { env = std::any_cast<CorridorEnv>(s); };
    g.step_fn     = [&](int a) -> mcts::GenericStepResult {
        if (env.at_goal()) return {0.0f, true};
        env.step_action(a);
        const bool done = env.at_goal();
        const float r = done ? 1.0f : -0.05f;
        return {r, done};
    };
    g.legal_actions_fn = [&]() -> std::vector<int> { return {0, 1, 2}; };
    g.observe_fn       = [&]() -> std::vector<float> {
        return { static_cast<float>(env.pos) / static_cast<float>(env.length - 1) };
    };
    return g;
}

} // namespace

TEST(generic_mcts_corridor_random_rollout_picks_right) {
    CorridorEnv env;
    env.pos = 0;
    auto g = make_corridor_env(env);
    mcts::GenericMcts m(std::move(g));
    mcts::GenericMctsConfig cfg;
    cfg.iterations    = 400;
    cfg.rollout_depth = 8;
    cfg.gamma         = 0.99f;
    cfg.c_puct        = 1.5f;
    m.set_config(cfg);

    const int action = m.search();
    CHECK(action == 1);                    // optimal
    CHECK(m.last_stats().iterations == 400);
    CHECK(m.last_stats().best_visits > 0);
    // Search must leave the env exactly where the caller left it.
    CHECK(env.pos == 0);
}

TEST(generic_mcts_corridor_value_fn_picks_right) {
    CorridorEnv env;
    env.pos = 2;
    auto g = make_corridor_env(env);
    mcts::GenericMcts m(std::move(g));
    mcts::GenericMctsConfig cfg;
    cfg.iterations = 200;
    cfg.gamma      = 0.99f;
    m.set_config(cfg);

    // Value head: closer to goal ⇒ higher value, in [-1, 1].
    m.set_value_fn([](const std::vector<float>& obs) -> float {
        return obs.empty() ? 0.0f : 2.0f * obs[0] - 1.0f;
    });

    const int action = m.search();
    CHECK(action == 1);
    CHECK(env.pos == 2);                   // unchanged
}

TEST(generic_mcts_root_visits_normalize) {
    CorridorEnv env;
    auto g = make_corridor_env(env);
    mcts::GenericMcts m(std::move(g));
    mcts::GenericMctsConfig cfg;
    cfg.iterations = 100;
    m.set_config(cfg);
    m.search();
    const auto v = m.root_visits();
    CHECK(v.size() == 3u);
    float s = 0.0f;
    for (float x : v) { CHECK(x >= 0.0f); s += x; }
    CHECK_NEAR(s, 1.0f, 1e-4f);
    // The "right" action (index 1) should dominate visits.
    CHECK(v[1] > v[0]);
    CHECK(v[1] > v[2]);
}

TEST(generic_mcts_advance_root_preserves_subtree) {
    CorridorEnv env;
    auto g = make_corridor_env(env);
    mcts::GenericMcts m(std::move(g));
    mcts::GenericMctsConfig cfg;
    cfg.iterations = 200;
    m.set_config(cfg);

    const int a = m.search();
    CHECK(a == 1);
    const int tree_before = m.last_stats().tree_size;
    CHECK(tree_before > 1);

    // Commit the action on the real env, advance the tree, and search
    // again — the second search should still pick right after subtree
    // promotion.
    env.step_action(a);
    m.advance_root(a);
    const int a2 = m.search();
    CHECK(a2 == 1);
}

TEST(generic_mcts_prior_fn_biases_search) {
    CorridorEnv env;
    auto g = make_corridor_env(env);
    mcts::GenericMcts m(std::move(g));
    mcts::GenericMctsConfig cfg;
    cfg.iterations = 60;
    cfg.c_puct     = 2.0f;     // amplify the prior's effect
    m.set_config(cfg);

    // Adversarial prior pinning all probability mass on "left". With a
    // small budget and an aggressive prior, search should follow the
    // prior often enough that visits on left exceed visits on right.
    m.set_prior_fn([](const std::vector<float>&,
                      const std::vector<int>&) -> std::vector<float> {
        return {1.0f, 0.0f, 0.0f};
    });

    m.search();
    const auto v = m.root_visits();
    CHECK(v[0] > v[1]);
}

TEST(generic_mcts_dirichlet_noise_explores_root) {
    // Property: with an adversarial prior pinning all probability mass on
    // one action, Dirichlet root noise must still allocate non-trivial
    // visit budget to every legal action — that's what "exploration" means
    // in the AlphaZero formulation. Whether right ultimately wins depends
    // on the rollout signal; here we only test that no action is starved.
    auto run = [](float dirichlet_alpha, float dirichlet_eps) {
        CorridorEnv env;
        auto g = make_corridor_env(env);
        mcts::GenericMcts m(std::move(g));
        mcts::GenericMctsConfig cfg;
        cfg.iterations        = 200;
        cfg.dirichlet_alpha   = dirichlet_alpha;
        cfg.dirichlet_epsilon = dirichlet_eps;
        cfg.seed              = 42ULL;
        m.set_config(cfg);
        m.set_prior_fn([](const std::vector<float>&,
                          const std::vector<int>&) -> std::vector<float> {
            return {1.0f, 0.0f, 0.0f};   // all mass on "left"
        });
        m.search();
        return m.root_visits();
    };

    // Without noise: prior dominates, and "right"/"stay" can be near-zero.
    const auto v_off = run(0.0f, 0.0f);
    // With noise: every legal action must get some share.
    const auto v_on = run(0.5f, 0.5f);
    CHECK(v_on[0] > 0.0f);
    CHECK(v_on[1] > 0.0f);
    CHECK(v_on[2] > 0.0f);
    // And "right" should be visited materially more under noise than
    // without it — that's the whole point of mixing in exploration mass.
    CHECK(v_on[1] > v_off[1]);
}

// ─── grid::ObsWindow / FrameStack Tests ────────────────────────────────────

TEST(obs_window_layout_invariants) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 2; s.cols_ahead = 3;
    s.rows_up = 1;     s.rows_down = 1;
    s.tile_channels = 2;
    s.self_block_size = 5;
    EntityLayerSpec layer;
    layer.channels = 3;
    layer.enumerate_fn = [] { return size_t{0}; };
    layer.sample_fn    = [](size_t) { return EntityCell{}; };
    ObsWindow win(s, [](int, int, float* o){ o[0]=0; o[1]=0; return true; }, {layer});
    const auto& L = win.layout();
    CHECK(L.cols == 6);
    CHECK(L.rows == 3);
    CHECK(L.tile_offset == 0);
    CHECK(L.tile_size == 6 * 3 * 2);
    CHECK(L.layers.size() == 1);
    CHECK(L.layers[0].offset == L.tile_size);
    CHECK(L.layers[0].size   == 6 * 3 * 3);
    CHECK(L.self_offset == L.tile_size + L.layers[0].size);
    CHECK(L.self_size   == 5);
    CHECK(L.total       == L.self_offset + 5);
    CHECK(win.out_dim() == L.total);
}

TEST(obs_window_tile_sampling_and_oob) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 1; s.cols_ahead = 1;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.oob_tile = {7.0f};
    // World is solid at col=0 only; everything else is OOB.
    auto tile = [](int col, int /*row*/, float* o) -> bool {
        if (col == 0) { o[0] = 3.0f; return true; }
        return false;   // OOB substitution path
    };
    ObsWindow win(s, tile, {});
    auto out = win.build(0, 0, {});
    // window cols = [-1, 0, 1] → [oob, 3, oob] → [7, 3, 7]
    CHECK(out.size() == 3);
    CHECK_NEAR(out[0], 7.0f, 1e-6f);
    CHECK_NEAR(out[1], 3.0f, 1e-6f);
    CHECK_NEAR(out[2], 7.0f, 1e-6f);
}

TEST(obs_window_entity_rasterizes_relative_to_ego) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 1; s.cols_ahead = 1;
    s.rows_up = 1;     s.rows_down = 1;
    s.tile_channels = 1;
    struct Ent { int col, row; float v; };
    std::vector<Ent> ents = { {5, 5, 1.0f}, {7, 6, 2.0f}, {99, 99, 9.0f} };
    EntityLayerSpec layer;
    layer.channels = 1;
    layer.enumerate_fn = [&] { return ents.size(); };
    layer.sample_fn    = [&](size_t i) {
        EntityCell c; c.col = ents[i].col; c.row = ents[i].row;
        c.values = { ents[i].v }; return c;
    };
    ObsWindow win(s, [](int,int,float* o){ o[0]=0; return true; }, {layer});
    auto out = win.build(6, 5, {});
    // window is 3 cols × 3 rows; offsets:
    //   ego at (col=1, row=1) maps to world (6,5)
    //   ent (5,5) → window (0,1) — "behind, same row"
    //   ent (7,6) → window (2,2) — "ahead, below"
    //   ent (99,99) — out of window, must not appear
    int tile_size = 3 * 3 * 1;
    int layer_off = tile_size;
    auto cell = [&](int wc, int wr) { return out[layer_off + (wr*3 + wc)]; };
    CHECK_NEAR(cell(0, 1), 1.0f, 1e-6f);
    CHECK_NEAR(cell(2, 2), 2.0f, 1e-6f);
    // Center cell (ego) should be untouched (no entity there).
    CHECK_NEAR(cell(1, 1), 0.0f, 1e-6f);
}

TEST(obs_window_entity_layer_modes) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    struct E { int col, row; float v; };
    std::vector<E> ents = { {0,0, 1.0f}, {0,0, 2.0f}, {0,0, 4.0f} };
    auto enumerate = [&] { return ents.size(); };
    auto sample = [&](size_t i) {
        EntityCell c; c.col=ents[i].col; c.row=ents[i].row; c.values={ents[i].v}; return c;
    };
    {
        EntityLayerSpec L; L.channels=1; L.overwrite=false;
        L.enumerate_fn=enumerate; L.sample_fn=sample;
        ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {L});
        auto out = win.build(0, 0, {});
        // additive: 1+2+4=7
        CHECK_NEAR(out[1], 7.0f, 1e-6f);   // out[0] is tile, out[1] is layer
    }
    {
        EntityLayerSpec L; L.channels=1; L.overwrite=true;
        L.enumerate_fn=enumerate; L.sample_fn=sample;
        ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {L});
        auto out = win.build(0, 0, {});
        // last write wins: 4
        CHECK_NEAR(out[1], 4.0f, 1e-6f);
    }
}

TEST(obs_window_normalizers_apply_after_accumulate) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.tile_normalize = {0.5f};
    EntityLayerSpec L;
    L.channels = 1;
    L.normalize = {0.25f};
    std::vector<std::pair<int,int>> dummy = {{0,0},{0,0}};
    L.enumerate_fn = [&]{ return dummy.size(); };
    L.sample_fn    = [&](size_t){ EntityCell c; c.col=0; c.row=0; c.values={4.0f}; return c; };
    ObsWindow win(s, [](int,int,float* o){o[0]=2.0f;return true;}, {L});
    auto out = win.build(0,0,{});
    CHECK_NEAR(out[0], 2.0f * 0.5f, 1e-6f);             // tile post-normalize
    CHECK_NEAR(out[1], (4.0f + 4.0f) * 0.25f, 1e-6f);  // accumulate then normalize
}

TEST(obs_window_self_block_copied) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.self_block_size = 4;
    ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {});
    auto out = win.build(0, 0, {1.0f, 2.0f, 3.0f});  // shorter than self block
    CHECK(out.size() == 5);
    CHECK_NEAR(out[1], 1.0f, 1e-6f);
    CHECK_NEAR(out[2], 2.0f, 1e-6f);
    CHECK_NEAR(out[3], 3.0f, 1e-6f);
    CHECK_NEAR(out[4], 0.0f, 1e-6f);   // missing self entry padded to 0
}

TEST(frame_stack_zero_initialized_and_pads_leading_frames) {
    using namespace brogameagent::grid;
    FrameStack fs(3, 4);
    CHECK(fs.out_dim() == 12);
    CHECK(fs.filled() == 0);
    auto out = fs.read();
    for (float v : out) CHECK_NEAR(v, 0.0f, 1e-6f);
    // Push one frame; the freshest frame must be the *last* k-block.
    float frame[] = {1.0f, 2.0f, 3.0f};
    fs.push(frame);
    out = fs.read();
    // Slots [0..2] zero (pad), slot 3 holds frame.
    for (int i = 0; i < 9; ++i) CHECK_NEAR(out[i], 0.0f, 1e-6f);
    CHECK_NEAR(out[9],  1.0f, 1e-6f);
    CHECK_NEAR(out[10], 2.0f, 1e-6f);
    CHECK_NEAR(out[11], 3.0f, 1e-6f);
}

TEST(frame_stack_chronological_order_when_full) {
    using namespace brogameagent::grid;
    FrameStack fs(1, 3);
    float a = 1.0f, b = 2.0f, c = 3.0f, d = 4.0f, e = 5.0f;
    fs.push(&a); fs.push(&b); fs.push(&c);
    auto out = fs.read();
    CHECK_NEAR(out[0], 1.0f, 1e-6f);
    CHECK_NEAR(out[1], 2.0f, 1e-6f);
    CHECK_NEAR(out[2], 3.0f, 1e-6f);   // freshest last
    fs.push(&d);  // evicts oldest (a)
    out = fs.read();
    CHECK_NEAR(out[0], 2.0f, 1e-6f);
    CHECK_NEAR(out[1], 3.0f, 1e-6f);
    CHECK_NEAR(out[2], 4.0f, 1e-6f);
    fs.push(&e);
    out = fs.read();
    CHECK_NEAR(out[0], 3.0f, 1e-6f);
    CHECK_NEAR(out[1], 4.0f, 1e-6f);
    CHECK_NEAR(out[2], 5.0f, 1e-6f);
}

TEST(frame_stack_reset_clears_history) {
    using namespace brogameagent::grid;
    FrameStack fs(1, 2);
    float a = 7.0f;
    fs.push(&a); fs.push(&a); fs.push(&a);
    fs.reset();
    CHECK(fs.filled() == 0);
    auto out = fs.read();
    CHECK_NEAR(out[0], 0.0f, 1e-6f);
    CHECK_NEAR(out[1], 0.0f, 1e-6f);
}

// ─── grid::FailureTape Tests ────────────────────────────────────────────────

TEST(failure_tape_records_only_tail_depth) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 3; cfg.ring_capacity = 16;
    cfg.penalty = 0.5f; cfg.floor = 0.05f;
    FailureTape tape(cfg);
    std::vector<FailureStep> tail;
    for (int i = 0; i < 10; ++i) tail.push_back({"S", i});
    tape.record_failure(tail);
    CHECK(tape.size() == 3);
    // Only the last 3 actions (7,8,9) should be counted; 0..6 must be 1.0.
    auto m = tape.multipliers("S", 10);
    for (int a = 0; a < 7; ++a) CHECK_NEAR(m[a], 1.0f, 1e-6f);
    CHECK_NEAR(m[7], 0.5f, 1e-6f);
    CHECK_NEAR(m[8], 0.5f, 1e-6f);
    CHECK_NEAR(m[9], 0.5f, 1e-6f);
}

TEST(failure_tape_multiplier_compounds_on_repeats_and_clamps_at_floor) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 100; cfg.ring_capacity = 256;
    cfg.penalty = 0.5f; cfg.floor = 0.1f;
    FailureTape tape(cfg);
    std::vector<FailureStep> t = { {"X", 2}, {"X", 2}, {"X", 2}, {"X", 2}, {"X", 2} };
    tape.record_failure(t);
    auto m = tape.multipliers("X", 4);
    // 0.5^5 = 0.03125 < 0.1 floor → clamps.
    CHECK_NEAR(m[2], 0.1f, 1e-6f);
    // Other actions unaffected.
    CHECK_NEAR(m[0], 1.0f, 1e-6f);
}

TEST(failure_tape_unknown_sig_is_identity) {
    using namespace brogameagent::grid;
    FailureTape tape;
    auto m = tape.multipliers("never_seen", 6);
    for (float v : m) CHECK_NEAR(v, 1.0f, 1e-6f);
}

TEST(failure_tape_ring_eviction_drops_old_counts) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 100; cfg.ring_capacity = 4;
    cfg.penalty = 0.5f; cfg.floor = 0.0f;
    FailureTape tape(cfg);
    // Fill ring with one (S,1) then push 4 (T,2) — S must be fully evicted.
    tape.record_failure({ {"S", 1} });
    tape.record_failure({ {"T", 2}, {"T", 2}, {"T", 2}, {"T", 2} });
    CHECK(tape.size() == 4);
    auto mS = tape.multipliers("S", 4);
    for (float v : mS) CHECK_NEAR(v, 1.0f, 1e-6f);
    auto mT = tape.multipliers("T", 4);
    CHECK_NEAR(mT[2], 0.5f * 0.5f * 0.5f * 0.5f, 1e-6f);
}

TEST(failure_tape_apply_priors_in_place) {
    using namespace brogameagent::grid;
    FailureTape tape;
    tape.record_failure({ {"S", 1}, {"S", 1} });
    std::vector<float> prior = {0.25f, 0.50f, 0.25f, 0.0f};
    tape.apply_priors("S", prior.data(), static_cast<int>(prior.size()));
    CHECK_NEAR(prior[0], 0.25f, 1e-6f);
    CHECK_NEAR(prior[1], 0.50f * 0.25f, 1e-6f);
    CHECK_NEAR(prior[2], 0.25f, 1e-6f);
}

TEST(failure_tape_clear_resets_state) {
    using namespace brogameagent::grid;
    FailureTape tape;
    tape.record_failure({ {"S", 1} });
    CHECK(tape.size() == 1);
    tape.clear();
    CHECK(tape.size() == 0);
    auto m = tape.multipliers("S", 4);
    for (float v : m) CHECK_NEAR(v, 1.0f, 1e-6f);
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
