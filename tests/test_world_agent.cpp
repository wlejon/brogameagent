#include "test_common.h"

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
    float speed = std::sqrt(v.x * v.x + v.y * v.y);
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
