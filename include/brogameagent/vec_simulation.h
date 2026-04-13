#pragma once

#include "agent.h"
#include "world.h"
#include "reward.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace brogameagent {

/// Batched 1v1 environment for self-play training.
///
/// Holds N independent Worlds; each world has exactly two Agents (hero on
/// team 0, opponent on team 1). All per-env random state lives in a
/// per-env RNG so rollouts are deterministic given a base seed.
///
/// The contract Python (or a C++ trainer) follows per training tick:
///
///   1. vec.observe(HERO_ID, hero_obs);                  // (N, OBS_TOTAL) write
///   2. vec.action_mask(HERO_ID, hero_mask, hero_ids);   // (N, MASK), (N, K)
///   3. (same two calls for OPPONENT_ID)
///   4. policy forward → hero_actions (N), opp_actions (N)
///   5. vec.apply_actions(HERO_ID, hero_actions);
///   6. vec.apply_actions(OPPONENT_ID, opp_actions);
///   7. vec.step();                                       // tick world, advance projectiles
///   8. vec.dones(done, winner); vec.rewards(rh, ro);     // per-env outcomes
///   9. vec.reset_done();                                 // reseed finished envs
///
/// Observations / masks for each agent re-use observation::build and
/// action_mask::build, so the policy sees the exact same vector layout as
/// the single-env path.
class VecSimulation {
public:
    static constexpr int HERO_ID     = 1;
    static constexpr int OPPONENT_ID = 2;

    // Built-in ability ids (also the default slot bindings 0..7).
    enum AbilityId : int {
        ABILITY_FIREBALL  = 0,   // direct damage, ranged single-target
        ABILITY_POISON    = 1,   // DoT, ranged single-target
        ABILITY_HEAL      = 2,   // direct heal self
        ABILITY_REGEN     = 3,   // HoT self
        ABILITY_STONESKIN = 4,   // armor + magic resist buff self
        ABILITY_FURY      = 5,   // damage multiplier buff self
        ABILITY_HASTE     = 6,   // move-speed buff self
        ABILITY_SHROUD    = 7,   // stealth (dodge chance) self
    };

    struct Config {
        int     numEnvs            = 64;
        float   arenaHalfSize      = 12.0f;  // square arena: [-half, +half]
        float   minSpawnDist       = 4.0f;
        float   maxSpawnDist       = 10.0f;
        float   dt                 = 0.016f;        // 16ms sim chunk
        int     maxStepsPerEpisode = 600;           // sim ticks (~9.6 s of game time)

        // Stats applied to BOTH agents on reset (symmetric 1v1).
        float   hp                = 100.0f;
        float   maxMana           = 100.0f;
        float   manaRegenPerSec   = 1.0f;
        float   damage            = 5.0f;       // basic attack damage
        float   attackRange       = 2.5f;
        float   attacksPerSec     = 2.0f;
        float   moveSpeed         = 6.0f;
        float   maxAccel          = 40.0f;
        float   maxTurnRate       = 6.2831853f * 1.5f; // 1.5 turns/sec
        float   radius            = 0.4f;

        // Reward shaping (same for both agents — symmetric).
        float   rewardDamageDealt    = 1.0f;
        float   rewardDamageTakenMul = -0.5f;
        float   rewardKill           = 100.0f;
        float   rewardDeath          = -100.0f;
        float   rewardStep           = -0.02f;
        // Applied to BOTH agents on timeout (winner==-1). Makes "stall to draw"
        // strictly worse than fighting without swamping the value function.
        float   rewardTimeout        = -30.0f;
    };

    explicit VecSimulation(const Config& cfg);
    ~VecSimulation();
    VecSimulation(const VecSimulation&)            = delete;
    VecSimulation& operator=(const VecSimulation&) = delete;

    int numEnvs() const { return cfg_.numEnvs; }
    const Config& config() const { return cfg_; }

    /// Seed every env with `baseSeed + envIdx` and reset them all to fresh
    /// scenarios. Call once before training begins.
    void seedAndReset(uint64_t baseSeed);

    /// Reset only envs whose `done` flag is currently set (typically the
    /// envs that just ended in the previous step). Increments per-env
    /// episode counter. Idempotent if no env is done.
    void resetDone();

    /// Force-reset a single env (e.g. when the trainer wants to swap
    /// opponents and start fresh).
    void resetEnv(int envIdx);

    /// Write (N, observation::TOTAL) observations for the given agent id
    /// (HERO_ID or OPPONENT_ID) into `out`. Caller-allocated, contiguous.
    void observe(int agentId, float* out) const;

    /// Write (N, action_mask::TOTAL) mask + (N, N_ENEMY_SLOTS) enemy ids.
    void actionMask(int agentId, float* outMask, int* outEnemyIds) const;

    /// Apply an array of N actions, one per env, to the given agent id.
    /// Skips envs whose agent is dead.
    void applyActions(int agentId, const AgentAction* actions);

    /// Advance every env's world: projectiles, cull, step counter, terminal
    /// detection, reward accumulation since the last apply/step pair.
    /// Call once per training tick AFTER applying both agents' actions.
    void step();

    /// Per-env: 1 if the episode just ended (this step or earlier and not
    /// yet reset), 0 otherwise. Winner is HERO_ID, OPPONENT_ID, or -1
    /// (timeout / draw).
    void dones(int* outDone, int* outWinner) const;

    /// Drain accumulated reward for both sides and zero the internal
    /// accumulator. Call once per training tick after step().
    void rewards(float* outHero, float* outOpponent);

    /// Per-env step counts (total steps elapsed in the current episode).
    void stepCounts(int* out) const;

    /// Per-env episode counters (total episodes finished across all time).
    void episodeCounts(int* out) const;

    // Direct access for advanced use (recording with Recorder, debug).
    World&       world(int i);
    const World& world(int i) const;
    Agent&       hero(int i);
    Agent&       opponent(int i);

private:
    struct EnvState {
        World           world;
        Agent           hero;
        Agent           opponent;
        std::mt19937_64 rng{0};
        RewardTracker   heroRT;
        RewardTracker   oppRT;
        int             stepCount  = 0;
        int             episodeIdx = 0;
        bool            doneFlag   = false;
        int             winner     = -1;
        // Reward accumulator (drained by rewards()).
        float           heroReward = 0.0f;
        float           oppReward  = 0.0f;
    };

    Config cfg_;
    std::vector<std::unique_ptr<EnvState>> envs_;

    void initEnv_(EnvState& env);
    void resetEnvImpl_(EnvState& env);
};

} // namespace brogameagent
