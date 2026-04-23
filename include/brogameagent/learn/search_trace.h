#pragma once

#include "replay_buffer.h"
#include "brogameagent/mcts.h"

#include <vector>

namespace brogameagent::learn {

// ─── Extract per-decision targets from a completed Mcts search ────────────
//
// Given the root Node of a finished search, convert its children's visit
// counts into the three factored policy targets (move / attack / ability).
// Each child's action contributes its visit fraction to the three factored
// distributions independently.
//
// This is the "expert distribution" the NN will be trained to match
// (ExIt / AlphaZero framing).

void targets_from_root(
    const mcts::Node& root,
    float target_move[nn::FactoredPolicyHead::N_MOVE],
    float target_attack[nn::FactoredPolicyHead::N_ATTACK],
    float target_ability[nn::FactoredPolicyHead::N_ABILITY]);

// Build one Situation struct from a (world, hero, root) at the current
// decision. value_target is filled in later when the episode return is
// known — caller is expected to patch .value_target before pushing to the
// replay buffer.
Situation make_situation(const World& world, const Agent& hero,
                         const mcts::Node& root);

} // namespace brogameagent::learn
