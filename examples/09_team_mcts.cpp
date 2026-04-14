// 09_team_mcts — cooperative multi-agent planning.
//
// Two heroes coordinate against two scripted enemies. TeamMcts picks a
// joint action (one CombatAction per hero) that maximises a shared team
// value. Enemies are driven by the OpponentPolicy during expansion/rollout.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>
#include <vector>

using namespace brogameagent;

int main() {
    World world;
    world.seed(0x7EA);

    std::vector<Agent> heroes(2), enemies(2);
    int next_id = 1;
    for (int i = 0; i < 2; i++) {
        auto& h = heroes[i];
        h.unit().id = next_id++;  h.unit().teamId = 0;
        h.unit().hp = 100; h.unit().damage = 10; h.unit().attackRange = 3;
        h.unit().attacksPerSec = 2;
        h.setPosition(-1.5f + 0.4f * i, 0.3f * i);
        h.setMaxAccel(30); h.setMaxTurnRate(10);
        world.addAgent(&h);
    }
    for (int i = 0; i < 2; i++) {
        auto& e = enemies[i];
        e.unit().id = next_id++;  e.unit().teamId = 1;
        e.unit().hp = 60;  e.unit().damage = 6;  e.unit().attackRange = 3;
        e.unit().attacksPerSec = 1;
        e.setPosition(1.5f + 0.4f * i, 0.3f * i);
        e.setMaxAccel(30); e.setMaxTurnRate(10);
        world.addAgent(&e);
    }
    std::vector<Agent*> team{ &heroes[0], &heroes[1] };

    mcts::MctsConfig cfg;
    cfg.iterations      = 200;
    cfg.rollout_horizon = 10;
    cfg.action_repeat   = 4;
    cfg.seed            = 0x7EA7;
    cfg.prior_c         = 1.5f;

    mcts::TeamMcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    auto joint = engine.search(world, team);
    const auto& s = engine.last_stats();
    // TeamMcts leaves best_mean at 0 by design — team value isn't well-defined
    // as a single per-hero mean. Inspect the root's per_hero stats for detail.
    std::printf("team search: iters=%d tree=%d elapsed=%dms root_children=%d\n",
        s.iterations, s.tree_size, s.elapsed_ms, s.root_children);
    for (size_t i = 0; i < joint.per_hero.size(); i++) {
        auto& a = joint.per_hero[i];
        std::printf("hero %d: move=%d attack=%d ability=%d\n",
            heroes[i].unit().id, (int)a.move_dir,
            (int)a.attack_slot, (int)a.ability_slot);
    }
    return 0;
}
