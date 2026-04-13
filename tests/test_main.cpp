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
