// 06_snapshot_rollout — poor man's MCTS (1-ply).
//
// The "fork N futures, score each, pick the winner" pattern that MCTS
// formalises. Demonstrates:
//   - World::snapshot / restore as the forking primitive.
//   - mcts::apply + mcts::CombatAction as a discrete action space.
//   - Hand-rolled rollout + evaluator for one decision.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>

using namespace brogameagent;

int main() {
    World world;
    world.seed(0xC0FFEE);

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().hp = 100; hero.unit().damage = 10; hero.unit().attackRange = 3;
    hero.unit().attacksPerSec = 2;
    hero.setPosition(0, 0); hero.setMaxAccel(30); hero.setMaxTurnRate(10);
    world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 50; enemy.unit().damage = 5; enemy.unit().attackRange = 3;
    enemy.unit().attacksPerSec = 1;
    enemy.setPosition(1, 0); enemy.setMaxAccel(30); enemy.setMaxTurnRate(10);
    world.addAgent(&enemy);

    // Three candidate hero actions for this decision window.
    mcts::CombatAction hold;
    mcts::CombatAction attack;  attack.attack_slot = 0;
    mcts::CombatAction retreat; retreat.move_dir = mcts::MoveDir::S;
    std::pair<const char*, mcts::CombatAction> options[] = {
        {"hold",    hold},
        {"attack",  attack},
        {"retreat", retreat},
    };

    WorldSnapshot saved = world.snapshot();
    mcts::HpDeltaEvaluator eval;

    std::printf("1-ply rollout evaluation:\n");
    const char* best_name = nullptr;
    float best_value = -2.0f;
    for (auto& [name, act] : options) {
        world.restore(saved);
        // Apply the action for a handful of ticks; opponent plays aggressive.
        const float dt = 0.05f;
        for (int t = 0; t < 20; t++) {
            mcts::apply(hero,  world, act, dt);
            mcts::apply(enemy, world, mcts::policy_aggressive(enemy, world), dt);
            world.stepProjectiles(dt);
            world.cullProjectiles();
            if (!hero.unit().alive() || !enemy.unit().alive()) break;
        }
        float v = eval.evaluate(world, hero.unit().id);
        std::printf("  %-8s  -> value %+ .3f  (hero hp=%.1f, enemy hp=%.1f)\n",
            name, v, hero.unit().hp, enemy.unit().hp);
        if (v > best_value) { best_value = v; best_name = name; }
    }
    world.restore(saved);
    std::printf("best action: %s (value %+.3f)\n", best_name, best_value);
    return 0;
}
