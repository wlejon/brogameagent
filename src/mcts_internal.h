#pragma once

#include "brogameagent/mcts.h"

namespace brogameagent::mcts {

bool is_terminal_for(const Agent& hero, const World& world);

std::vector<float> compute_priors(
    const IPrior* prior,
    const Agent& self,
    const World& world,
    const std::vector<CombatAction>& actions);

void step_team_window(
    World& world,
    const std::vector<Agent*>& heroes,
    const std::vector<CombatAction>& hero_actions,
    const OpponentPolicy& opponent,
    float dt,
    int action_repeat,
    int team_id);

bool team_terminal(const World& world, int team_id);

} // namespace brogameagent::mcts
