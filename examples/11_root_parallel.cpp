// 11_root_parallel — root-parallel MCTS over N std::threads.
//
// Each thread runs an independent Mcts on its own World copy with a
// seed-offset, then root visit counts are merged across trees to pick
// the final action. Real wall-clock speedup on multi-core boxes, at the
// cost of some noise vs one huge tree.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>
#include <vector>

using namespace brogameagent;

namespace {
struct Scene {
    World world;
    Agent hero;
    Agent enemy;
};

std::unique_ptr<Scene> build() {
    auto s = std::make_unique<Scene>();
    s->hero.unit().id = 1; s->hero.unit().teamId = 0;
    s->hero.unit().hp = 100; s->hero.unit().damage = 10;
    s->hero.unit().attackRange = 3; s->hero.unit().attacksPerSec = 2;
    s->hero.setPosition(-1, 0); s->hero.setMaxAccel(30); s->hero.setMaxTurnRate(10);

    s->enemy.unit().id = 2; s->enemy.unit().teamId = 1;
    s->enemy.unit().hp = 80; s->enemy.unit().damage = 8;
    s->enemy.unit().attackRange = 3; s->enemy.unit().attacksPerSec = 1.5f;
    s->enemy.setPosition(1, 0); s->enemy.setMaxAccel(30); s->enemy.setMaxTurnRate(10);

    s->world.addAgent(&s->hero);
    s->world.addAgent(&s->enemy);
    s->world.seed(0xCAFE);
    return s;
}
} // namespace

int main() {
    // Four identical scenes, one per thread.
    std::vector<std::unique_ptr<Scene>> scenes;
    std::vector<World*> worlds;
    for (int i = 0; i < 4; i++) {
        scenes.push_back(build());
        worlds.push_back(&scenes.back()->world);
    }

    mcts::MctsConfig cfg;
    cfg.iterations      = 200;    // per thread
    cfg.rollout_horizon = 12;
    cfg.action_repeat   = 4;
    cfg.seed            = 0xDEAD;  // per-thread seeds = seed + thread_idx

    mcts::ParallelSearchStats pstats;
    auto action = mcts::root_parallel_search(
        worlds,
        /*hero_id*/ 1,
        cfg,
        std::make_shared<mcts::HpDeltaEvaluator>(),
        std::make_shared<mcts::AggressiveRollout>(),
        /*opponent_policy*/ mcts::policy_aggressive,
        &pstats);

    std::printf("root-parallel: %d threads, %d total iters, %d ms, merged_best_visits=%d\n",
        pstats.num_threads, pstats.total_iterations, pstats.elapsed_ms,
        pstats.merged_best_visits);
    std::printf("chosen: move=%d attack=%d ability=%d\n",
        (int)action.move_dir, (int)action.attack_slot, (int)action.ability_slot);
    return 0;
}
