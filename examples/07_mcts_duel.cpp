// 07_mcts_duel — full single-agent Mcts vs a scripted opponent.
//
// Demonstrates:
//   - Mcts engine + MctsConfig.
//   - AggressiveRollout, AttackBiasPrior, PUCT (prior_c), progressive widening.
//   - advance_root for tree reuse across consecutive decisions.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>

using namespace brogameagent;

int main() {
    World world;
    world.seed(0x1010);

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100; hero.unit().damage = 10; hero.unit().attackRange = 3;
    hero.unit().attacksPerSec = 2;
    hero.setPosition(-1, 0); hero.setMaxAccel(30); hero.setMaxTurnRate(10);
    world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 80; enemy.unit().damage = 8; enemy.unit().attackRange = 3;
    enemy.unit().attacksPerSec = 1.5f;
    enemy.setPosition(1, 0); enemy.setMaxAccel(30); enemy.setMaxTurnRate(10);
    world.addAgent(&enemy);

    mcts::MctsConfig cfg;
    cfg.iterations      = 256;
    cfg.rollout_horizon = 16;
    cfg.action_repeat   = 4;
    cfg.seed            = 0xABC;
    cfg.prior_c         = 1.5f;    // enable PUCT
    cfg.pw_alpha        = 0.5f;    // progressive widening: go deeper, not wider

    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    // Plan one decision.
    mcts::CombatAction a = engine.search(world, hero);
    const auto& s = engine.last_stats();
    std::printf("search: iters=%d tree=%d elapsed=%dms best_visits=%d best_mean=%+.3f\n",
        s.iterations, s.tree_size, s.elapsed_ms, s.best_visits, s.best_mean);
    std::printf("chosen: move_dir=%d attack_slot=%d ability_slot=%d\n",
        (int)a.move_dir, (int)a.attack_slot, (int)a.ability_slot);

    // Commit the action and reuse the tree for a follow-up decision.
    engine.advance_root(a);
    mcts::CombatAction a2 = engine.search(world, hero);
    std::printf("follow-up reused_root=%d new best_mean=%+.3f\n",
        engine.last_stats().reused_root, engine.last_stats().best_mean);
    (void)a2;
    return 0;
}
