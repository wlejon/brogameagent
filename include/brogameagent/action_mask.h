#pragma once

#include "observation.h"
#include "unit.h"

namespace brogameagent {

class Agent;
class World;

/// Validity mask for a policy's discrete outputs. Lets the NN know which
/// attack targets / abilities are legal to pick *right now*, so its softmax
/// can be renormalized over just the legal set.
///
/// Enemy slots line up 1:1 with the enemy slots in the observation vector
/// (sorted nearest-first), so "output head k picks enemy slot k" matches
/// "observation slot k describes enemy slot k".
namespace action_mask {

constexpr int N_ENEMY_SLOTS   = observation::K_ENEMIES;
constexpr int N_ABILITY_SLOTS = Unit::MAX_ABILITIES;
constexpr int TOTAL           = N_ENEMY_SLOTS + N_ABILITY_SLOTS;

/// Fill `out[0..TOTAL)` with 1.0 for legal choices, 0.0 for illegal.
///
/// Layout:
///   [0 .. N_ENEMY_SLOTS)     attackable enemy in slot k (alive, in attackRange,
///                            attack cooldown ready)
///   [N_ENEMY_SLOTS ..)       ability slot s is castable (bound, cooldown ready,
///                            mana sufficient — range is not checked here since
///                            the target is supplied at cast time)
///
/// Also returns the enemy IDs associated with each slot (caller-provided
/// out array of size N_ENEMY_SLOTS, -1 for empty slots) so the policy can
/// translate "fire slot k" into a concrete Unit::id for the action.
void build(const Agent& self, const World& world,
           float* outMask, int* outEnemyIds);

} // namespace action_mask
} // namespace brogameagent
