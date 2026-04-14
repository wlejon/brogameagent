// 08_decoupled_1v1 — simultaneous-move 1v1 via DecoupledMcts.
//
// Use this when both players act at the same instant without seeing each
// other's choice. Hero maximises the evaluator; opponent minimises (the
// engine handles sign-flipping internally).
//
// Output: the joint action — the hero action to commit plus the planner's
// best-response prediction for the opponent.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>

using namespace brogameagent;

int main() {
    World world;
    world.seed(0xDEC0);

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100; hero.unit().damage = 10; hero.unit().attackRange = 3;
    hero.unit().attacksPerSec = 2;
    hero.setPosition(-1, 0); hero.setMaxAccel(30); hero.setMaxTurnRate(10);
    world.addAgent(&hero);

    Agent opp; opp.unit().id = 2; opp.unit().teamId = 1;
    opp.unit().hp = 100; opp.unit().damage = 10; opp.unit().attackRange = 3;
    opp.unit().attacksPerSec = 2;
    opp.setPosition(1, 0); opp.setMaxAccel(30); opp.setMaxTurnRate(10);
    world.addAgent(&opp);

    mcts::MctsConfig cfg;
    cfg.iterations      = 400;
    cfg.rollout_horizon = 12;
    cfg.action_repeat   = 4;
    cfg.seed            = 0xD1D;
    cfg.prior_c         = 1.5f;

    mcts::DecoupledMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());

    auto joint = engine.search(world, hero, opp);
    const auto& s = engine.last_stats();
    std::printf("search: iters=%d tree=%d elapsed=%dms best_mean=%+.3f\n",
        s.iterations, s.tree_size, s.elapsed_ms, s.best_mean);
    std::printf("hero commits: move=%d attack=%d\n",
        (int)joint.hero.move_dir, (int)joint.hero.attack_slot);
    std::printf("opp predicted: move=%d attack=%d\n",
        (int)joint.opp.move_dir,  (int)joint.opp.attack_slot);
    return 0;
}
