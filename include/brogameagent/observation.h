#pragma once

namespace brogameagent {

class Agent;
class World;

/// Ego-centric observation vector for a NN policy.
///
/// Layout (all floats, all roughly in [-1, 1] unless noted):
///
///   SELF block (SELF_FEATURES = 14):
///     [0]      hp / maxHp                     in [0,1]
///     [1]      mana / maxMana                 in [0,1] (0 if maxMana==0)
///     [2]      attackCooldown normalized      in [0,1]
///     [3..10]  abilityCooldowns[0..7] normalized   in [0,1]
///     [11]     speed / moveSpeed              in [0,1]
///     [12]     sin(aimYaw - yaw)              heading vs aim (strafe-aware)
///     [13]     cos(aimYaw - yaw)
///
///   ENEMY block: K_ENEMIES * ENEMY_FEATURES, sorted nearest-first.
///   Absent slots are zeroed (valid flag=0).
///     [0] valid (1 if slot is populated, else 0)
///     [1] rel X in agent's local frame / OBS_RANGE                  (-1..1)
///     [2] rel Z in agent's local frame / OBS_RANGE                  (-1..1)
///     [3] distance / OBS_RANGE                                       (0..1)
///     [4] enemy hp / maxHp                                           (0..1)
///     [5] inAttackRange flag (self's attackRange)                    (0 or 1)
///
///   ALLY block: K_ALLIES * ALLY_FEATURES, sorted nearest-first.
///     [0] valid
///     [1] rel X local / OBS_RANGE
///     [2] rel Z local / OBS_RANGE
///     [3] distance / OBS_RANGE
///     [4] ally hp / maxHp
namespace observation {

constexpr int K_ENEMIES = 5;
constexpr int K_ALLIES  = 4;
constexpr int SELF_FEATURES  = 14;
constexpr int ENEMY_FEATURES = 6;
constexpr int ALLY_FEATURES  = 5;

/// Range (world units) used to normalize positions/distances.
/// Anything further than this saturates.
constexpr float OBS_RANGE = 50.0f;

constexpr int TOTAL =
    SELF_FEATURES
    + K_ENEMIES * ENEMY_FEATURES
    + K_ALLIES  * ALLY_FEATURES;

/// Fill `out[0..TOTAL)` with the ego-centric observation for `self`.
/// Caller owns the buffer; must have room for at least TOTAL floats.
void build(const Agent& self, const World& world, float* out);

} // namespace observation
} // namespace brogameagent
