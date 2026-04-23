// 14_fog_of_war — partial-observability MCTS against a hidden enemy.
//
// Demonstrates:
//   - obs::observe / obs::merge : building + maintaining a team-scoped
//     TeamObservation under FOV + LOS + range limits.
//   - belief::TeamBelief        : particle filter over hidden enemy state
//     with negative-information pruning against observer geometry.
//   - mcts::InfoSetMcts         : determinized IS-MCTS that samples from the
//     belief per iteration and runs standard MCTS against the sampled world.
//
// Scenario: hero at (-6, 0), enemy at (+6, 0), an AABB wall in between.
// The wall occludes direct LOS; the hero must plan around the uncertainty.

#include "brogameagent/agent.h"
#include "brogameagent/belief.h"
#include "brogameagent/info_set_mcts.h"
#include "brogameagent/mcts.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>

using namespace brogameagent;

int main() {
    // --- World, nav grid, and a wall that occludes the enemy from the hero.
    World world;
    world.seed(0x7091);

    NavGrid nav(-12, -12, 12, 12, 0.5f);

    AABB wall{ 0.0f, 0.0f, 0.5f, 4.0f };
    world.addObstacle(wall);
    nav.addObstacle(wall, 0.5f);

    // Hero (team 0).
    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100; hero.unit().damage = 10;
    hero.unit().attackRange = 4; hero.unit().attacksPerSec = 2;
    hero.setNavGrid(&nav);
    hero.setPosition(-6, 0); hero.setMaxAccel(30); hero.setMaxTurnRate(10);
    world.addAgent(&hero);

    // Enemy (team 1) — hidden behind the wall.
    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 80; enemy.unit().damage = 8;
    enemy.unit().attackRange = 4; enemy.unit().attacksPerSec = 1.5f;
    enemy.setNavGrid(&nav);
    enemy.setPosition(6, 2); enemy.setMaxAccel(30); enemy.setMaxTurnRate(10);
    world.addAgent(&enemy);

    // --- Visibility config for the hero's team.
    obs::VisibilityConfig vis;
    vis.fov_radians = 6.283185f;   // omnidirectional for simplicity
    vis.max_range   = 12.0f;
    vis.check_los   = true;

    // --- Belief: 64 particles per enemy, seeded around a (stale) initial prior.
    auto team_belief = std::make_shared<belief::TeamBelief>(
        /*team_id=*/0, /*num_particles=*/64, &nav,
        belief::MotionParams{ /*max_speed*/ 6.0f,
                               /*accel_std*/ 4.0f,
                               /*spread_on_loss*/ 3.0f },
        /*rng_seed=*/0xD00DU);
    Vec2 prior_center{ 6, 0 };
    team_belief->register_enemy(enemy.unit().id, enemy.unit().maxHp, &prior_center);

    // Fold the very first observation in (enemy is hidden — wall occludes).
    auto obs0 = obs::observe(world, 0, vis, /*now=*/0.0f);
    team_belief->update(obs0);

    std::printf("initial ESS: %.3f, enemy visible now: %s\n",
        team_belief->effective_sample_size(),
        team_belief->enemies().front().visible ? "yes" : "no");

    // --- IS-MCTS config. Modest iteration count to keep the demo snappy.
    mcts::MctsConfig cfg;
    cfg.iterations      = 256;
    cfg.rollout_horizon = 16;
    cfg.action_repeat   = 4;
    cfg.seed            = 0xFACE;
    cfg.prior_c         = 1.5f;
    cfg.pw_alpha        = 0.5f;

    mcts::InfoSetMcts planner(cfg);
    planner.set_belief(team_belief);
    planner.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    planner.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    planner.set_opponent_policy(mcts::policy_scripted);

    // Plan one decision under the fog.
    auto action = planner.search(world, hero);
    const auto& s = planner.last_stats();
    std::printf("fog search: iters=%d tree=%d best_visits=%d best_mean=%+.3f ess=%.3f elapsed=%dms\n",
        s.iterations, s.tree_size, s.best_visits, s.best_mean, s.mean_ess, s.elapsed_ms);
    std::printf("chosen: move_dir=%d attack_slot=%d ability_slot=%d\n",
        (int)action.move_dir, (int)action.attack_slot, (int)action.ability_slot);

    // Walk a few ticks so the belief can demonstrate propagation + update.
    const float dt = 0.064f;  // 4 sim ticks per decision window
    float now = 0.0f;
    for (int window = 0; window < 5; window++) {
        mcts::apply(hero, world, action, dt);
        mcts::apply(enemy, world, mcts::policy_scripted(enemy, world), dt);
        world.stepProjectiles(dt);
        world.cullProjectiles();
        now += dt;

        team_belief->propagate(world, vis, dt);
        auto o = obs::observe(world, 0, vis, now);
        team_belief->update(o);

        const auto& e = team_belief->enemies().front();
        std::printf("t=%.2f visible=%d last_seen=%.2f ESS=%.3f hero=(%.2f,%.2f) truth_enemy=(%.2f,%.2f)\n",
            now, (int)e.visible, e.last_seen_elapsed,
            team_belief->effective_sample_size(),
            hero.x(), hero.z(), enemy.x(), enemy.z());

        planner.advance_root(action);
        action = planner.search(world, hero);
    }

    return 0;
}
