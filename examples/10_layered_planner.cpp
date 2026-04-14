// 10_layered_planner — hierarchical TacticMcts over TeamMcts.
//
// Coarse tactic planning (Hold / FocusLowestHp / Scatter / Retreat) runs
// every `tactic_window_decisions` fine decisions; fine per-hero TeamMcts
// runs every decision, biased toward the committed tactic via TacticPrior.
//
// Calls decide() several times so the replan schedule is visible.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <cstdio>
#include <memory>
#include <vector>

using namespace brogameagent;

namespace {
const char* tactic_name(mcts::TacticKind k) {
    switch (k) {
        case mcts::TacticKind::Hold:          return "Hold";
        case mcts::TacticKind::FocusLowestHp: return "FocusLowestHp";
        case mcts::TacticKind::Scatter:       return "Scatter";
        case mcts::TacticKind::Retreat:       return "Retreat";
        default:                              return "?";
    }
}
} // namespace

int main() {
    World world;
    world.seed(0x1A7E);

    std::vector<Agent> heroes(2), enemies(2);
    int next_id = 1;
    for (int i = 0; i < 2; i++) {
        auto& h = heroes[i];
        h.unit().id = next_id++; h.unit().teamId = 0;
        h.unit().hp = 100; h.unit().damage = 10; h.unit().attackRange = 3;
        h.unit().attacksPerSec = 2;
        h.setPosition(-1.5f + 0.4f * i, 0.3f * i);
        h.setMaxAccel(30); h.setMaxTurnRate(10);
        world.addAgent(&h);
    }
    for (int i = 0; i < 2; i++) {
        auto& e = enemies[i];
        e.unit().id = next_id++; e.unit().teamId = 1;
        e.unit().hp = 50; e.unit().damage = 5; e.unit().attackRange = 3;
        e.unit().attacksPerSec = 1;
        e.setPosition(1.5f + 0.4f * i, 0.3f * i);
        e.setMaxAccel(30); e.setMaxTurnRate(10);
        world.addAgent(&e);
    }
    std::vector<Agent*> team{ &heroes[0], &heroes[1] };

    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 60;
    cfg.tactic_cfg.rollout_horizon        = 6;
    cfg.tactic_cfg.action_repeat          = 4;
    cfg.tactic_cfg.tactic_window_decisions = 3;  // replan every 3 fine decisions
    cfg.tactic_cfg.seed                   = 0x1A71;
    cfg.fine_cfg.iterations               = 80;
    cfg.fine_cfg.rollout_horizon          = 8;
    cfg.fine_cfg.action_repeat            = 4;
    cfg.fine_cfg.seed                     = 0x1A72;
    cfg.fine_cfg.prior_c                  = 1.5f;  // TacticPrior requires PUCT

    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    planner.set_opponent_policy(mcts::policy_aggressive);

    // Six calls so we see two full tactic windows.
    for (int step = 0; step < 6; step++) {
        auto joint = planner.decide(world, team);
        const auto& s = planner.last_stats();
        std::printf("step %d: tactic=%-13s replanned=%s windows_left=%d",
            step, tactic_name(s.committed_tactic.kind),
            s.replanned_this_call ? "yes" : "no ", s.windows_until_replan);
        std::printf("  |  hero[0] move=%d atk=%d  hero[1] move=%d atk=%d\n",
            (int)joint.per_hero[0].move_dir, (int)joint.per_hero[0].attack_slot,
            (int)joint.per_hero[1].move_dir, (int)joint.per_hero[1].attack_slot);
    }
    return 0;
}
