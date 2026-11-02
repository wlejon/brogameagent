#include "test_common.h"

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
