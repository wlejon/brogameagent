#include "test_common.h"

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
        CHECK_NEAR(p.pos.y, 1.0f, 1e-4f);
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
