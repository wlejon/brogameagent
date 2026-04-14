// 04_obs_mask_reward — the three policy-facing builders.
//
// Demonstrates:
//   - observation::build: ego-centric feature vector.
//   - action_mask::build: which attacks/abilities are legal right now.
//   - RewardTracker: per-tick deltas (damage dealt/taken, kills, distance).

#include "brogameagent/action_mask.h"
#include "brogameagent/agent.h"
#include "brogameagent/observation.h"
#include "brogameagent/reward.h"
#include "brogameagent/world.h"

#include <cstdio>

using namespace brogameagent;

int main() {
    World world;

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.unit().attackRange = 3.0f; hero.unit().damage = 10.0f;
    hero.unit().attacksPerSec = 2.0f;
    hero.setPosition(0.0f, 0.0f);
    world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 30.0f; enemy.setPosition(1.5f, 0.0f);
    world.addAgent(&enemy);

    // --- Observation ---
    float obs[observation::TOTAL];
    observation::build(hero, world, obs);
    std::printf("observation size: %d\n", observation::TOTAL);
    std::printf("self hp frac: %.2f  attack cd: %.2f\n", obs[0], obs[2]);
    // First enemy slot starts at SELF_FEATURES; layout is [valid, relX, relZ, dist, hp, inRange]
    int e0 = observation::SELF_FEATURES;
    std::printf("enemy[0]: valid=%.0f relX=%.2f relZ=%.2f dist=%.2f inRange=%.0f\n",
        obs[e0+0], obs[e0+1], obs[e0+2], obs[e0+3], obs[e0+5]);

    // --- Action mask ---
    float mask[action_mask::TOTAL];
    int   enemy_ids[action_mask::N_ENEMY_SLOTS];
    action_mask::build(hero, world, mask, enemy_ids);
    std::printf("attack slot 0 legal: %s (target id=%d)\n",
        mask[0] > 0 ? "yes" : "no", enemy_ids[0]);

    // --- Reward tracker: capture deltas across an attack. ---
    RewardTracker rt;
    rt.reset(hero, world);
    world.resolveAttack(hero, enemy.unit().id);
    auto d = rt.consume(hero, world);
    std::printf("delta: dealt=%.1f taken=%.1f kills=%d deaths=%d dist=%.2f\n",
        d.damageDealt, d.damageTaken, d.kills, d.deaths, d.distanceTravelled);
    return 0;
}
