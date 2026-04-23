#include "brogameagent/learn/search_trace.h"
#include "brogameagent/action_mask.h"
#include "brogameagent/agent.h"
#include "brogameagent/observation.h"
#include "brogameagent/world.h"

#include <cstring>

namespace brogameagent::learn {

using nn::FactoredPolicyHead;

void targets_from_root(
    const mcts::Node& root,
    float target_move[FactoredPolicyHead::N_MOVE],
    float target_attack[FactoredPolicyHead::N_ATTACK],
    float target_ability[FactoredPolicyHead::N_ABILITY])
{
    for (int i = 0; i < FactoredPolicyHead::N_MOVE;    ++i) target_move[i]    = 0.0f;
    for (int i = 0; i < FactoredPolicyHead::N_ATTACK;  ++i) target_attack[i]  = 0.0f;
    for (int i = 0; i < FactoredPolicyHead::N_ABILITY; ++i) target_ability[i] = 0.0f;

    int total = 0;
    for (const auto& c : root.children) total += c->visits;
    if (total <= 0) return;
    const float inv = 1.0f / static_cast<float>(total);

    for (const auto& c : root.children) {
        const float w = static_cast<float>(c->visits) * inv;
        int mi = static_cast<int>(c->action.move_dir);
        if (mi < 0 || mi >= FactoredPolicyHead::N_MOVE) mi = 0;  // collapse path-kinds
        target_move[mi] += w;

        const int ai = (c->action.attack_slot < 0)
                        ? (FactoredPolicyHead::N_ATTACK - 1)
                        : c->action.attack_slot;
        target_attack[ai] += w;

        const int bi = (c->action.ability_slot < 0)
                        ? (FactoredPolicyHead::N_ABILITY - 1)
                        : c->action.ability_slot;
        target_ability[bi] += w;
    }
}

Situation make_situation(const World& world, const Agent& hero, const mcts::Node& root) {
    Situation s;
    observation::build(hero, world, s.obs.data());

    int enemy_ids[action_mask::N_ENEMY_SLOTS];
    float mask_buf[action_mask::TOTAL];
    action_mask::build(hero, world, mask_buf, enemy_ids);
    for (int i = 0; i < action_mask::N_ENEMY_SLOTS;   ++i) s.atk_mask[i]  = mask_buf[i];
    for (int i = 0; i < action_mask::N_ABILITY_SLOTS; ++i)
        s.abil_mask[i] = mask_buf[action_mask::N_ENEMY_SLOTS + i];

    targets_from_root(root, s.target_move.data(), s.target_attack.data(),
                      s.target_ability.data());
    s.value_target = 0.0f;
    return s;
}

} // namespace brogameagent::learn
