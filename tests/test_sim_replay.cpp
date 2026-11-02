#include "test_common.h"

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
    auto f1 = r.frame(1);
    const auto& e = f1.events[0];
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
    auto f0 = r.frame(0);
    const auto& pp = f0.projectiles[0];
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
