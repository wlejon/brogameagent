#include "brogameagent/vec_simulation.h"
#include "brogameagent/observation.h"
#include "brogameagent/action_mask.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {

namespace {

// Register the 8 built-in abilities on a world.
//
// All durations are in seconds, costs in mana, ranges in world units.
// Single-target damage/DoT abilities use the AgentAction::attackTargetId
// as their target; self-cast abilities ignore the target argument.
void registerBuiltinAbilities(World& w) {
    using A = VecSimulation;

    // Fireball — direct magical damage, single target.
    {
        AbilitySpec s;
        s.cooldown = 4.0f; s.manaCost = 8.0f; s.range = 6.0f;
        s.fn = [](Agent& caster, World& world, int targetId) {
            Agent* tgt = world.findById(targetId);
            if (!tgt || !tgt->unit().alive()) return;
            world.dealDamage(caster, *tgt, 25.0f, DamageKind::Magical);
        };
        w.registerAbility(A::ABILITY_FIREBALL, std::move(s));
    }

    // Poison — magical damage-over-time, single target.
    {
        AbilitySpec s;
        s.cooldown = 6.0f; s.manaCost = 8.0f; s.range = 5.0f;
        s.fn = [](Agent& caster, World& world, int targetId) {
            Agent* tgt = world.findById(targetId);
            if (!tgt || !tgt->unit().alive()) return;
            Unit& u = tgt->unit();
            u.dotDps       = 5.0f;
            u.dotRemaining = 4.0f;
            u.dotKind      = DamageKind::Magical;
            u.dotSourceId  = caster.unit().id;
        };
        w.registerAbility(A::ABILITY_POISON, std::move(s));
    }

    // Heal — direct self-heal.
    {
        AbilitySpec s;
        s.cooldown = 5.0f; s.manaCost = 8.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.hp = std::min(u.hp + 25.0f, u.maxHp);
        };
        w.registerAbility(A::ABILITY_HEAL, std::move(s));
    }

    // Regen — heal-over-time on self.
    {
        AbilitySpec s;
        s.cooldown = 8.0f; s.manaCost = 8.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.hotRate      = 5.0f;
            u.hotRemaining = 5.0f;
        };
        w.registerAbility(A::ABILITY_REGEN, std::move(s));
    }

    // Stoneskin — additive armor + MR buff on self.
    {
        AbilitySpec s;
        s.cooldown = 10.0f; s.manaCost = 10.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.armorBonus                  = 25.0f;
            u.armorBonusRemaining         = 5.0f;
            u.magicResistBonus            = 25.0f;
            u.magicResistBonusRemaining   = 5.0f;
        };
        w.registerAbility(A::ABILITY_STONESKIN, std::move(s));
    }

    // Fury — multiplicative damage buff on self.
    {
        AbilitySpec s;
        s.cooldown = 8.0f; s.manaCost = 10.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.damageMul          = 1.5f;
            u.damageMulRemaining = 4.0f;
        };
        w.registerAbility(A::ABILITY_FURY, std::move(s));
    }

    // Haste — multiplicative move-speed buff on self.
    {
        AbilitySpec s;
        s.cooldown = 8.0f; s.manaCost = 6.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.moveSpeedMul          = 1.5f;
            u.moveSpeedMulRemaining = 4.0f;
        };
        w.registerAbility(A::ABILITY_HASTE, std::move(s));
    }

    // Shroud — stealth (dodge chance) on self.
    {
        AbilitySpec s;
        s.cooldown = 12.0f; s.manaCost = 10.0f; s.range = 0.0f;
        s.fn = [](Agent& caster, World&, int) {
            Unit& u = caster.unit();
            u.stealthChance          = 0.5f;
            u.stealthChanceRemaining = 3.0f;
        };
        w.registerAbility(A::ABILITY_SHROUD, std::move(s));
    }
}

} // namespace

VecSimulation::VecSimulation(const Config& cfg) : cfg_(cfg) {
    envs_.reserve(static_cast<size_t>(cfg_.numEnvs));
    for (int i = 0; i < cfg_.numEnvs; i++) {
        auto e = std::make_unique<EnvState>();
        initEnv_(*e);
        envs_.push_back(std::move(e));
    }
}

VecSimulation::~VecSimulation() = default;

void VecSimulation::initEnv_(EnvState& env) {
    env.hero.unit().id     = HERO_ID;
    env.hero.unit().teamId = 0;
    env.opponent.unit().id     = OPPONENT_ID;
    env.opponent.unit().teamId = 1;

    env.world.addAgent(&env.hero);
    env.world.addAgent(&env.opponent);

    // Per-world ability registry (cheap; abilities are stored in a hash map).
    registerBuiltinAbilities(env.world);
}

static void applyStats(Agent& a, const VecSimulation::Config& cfg) {
    Unit& u = a.unit();
    u.hp                = cfg.hp;
    u.maxHp             = cfg.hp;
    u.maxMana           = cfg.maxMana;
    u.mana              = cfg.maxMana;
    u.manaRegenPerSec   = cfg.manaRegenPerSec;
    u.damage            = cfg.damage;
    u.attackRange       = cfg.attackRange;
    u.attacksPerSec     = cfg.attacksPerSec;
    u.moveSpeed         = cfg.moveSpeed;
    u.radius            = cfg.radius;
    u.attackCooldown    = 0.0f;
    for (int i = 0; i < Unit::MAX_ABILITIES; i++) {
        u.abilityCooldowns[i] = 0.0f;
        // Default binding: slot i holds ability id i (Fireball..Shroud).
        u.abilitySlot[i] = (i < 8) ? i : -1;
    }
    // Clear all timed effects.
    u.armorBonus = u.magicResistBonus = 0.0f;
    u.armorBonusRemaining = u.magicResistBonusRemaining = 0.0f;
    u.damageMul = u.attacksMul = u.moveSpeedMul = 1.0f;
    u.damageMulRemaining = u.attacksMulRemaining = u.moveSpeedMulRemaining = 0.0f;
    u.stealthChance = 0.0f; u.stealthChanceRemaining = 0.0f;
    u.dotDps = u.dotRemaining = 0.0f;
    u.dotSourceId = -1;
    u.hotRate = u.hotRemaining = 0.0f;

    a.setSpeed(cfg.moveSpeed);
    a.setRadius(cfg.radius);
    a.setMaxAccel(cfg.maxAccel);
    a.setMaxTurnRate(cfg.maxTurnRate);
}

void VecSimulation::resetEnvImpl_(EnvState& env) {
    // Sample a random spawn line through the origin: the two agents start
    // facing each other across a uniformly-sampled distance and angle.
    std::uniform_real_distribution<float> angleDist(-3.14159265f, 3.14159265f);
    std::uniform_real_distribution<float> distDist(cfg_.minSpawnDist, cfg_.maxSpawnDist);
    float angle = angleDist(env.rng);
    float dist  = distDist(env.rng);
    float half  = 0.5f * dist;

    float hx =  std::cos(angle) * half;
    float hz =  std::sin(angle) * half;
    float ox = -hx;
    float oz = -hz;

    // Clamp into arena.
    auto clamp = [&](float v) {
        if (v >  cfg_.arenaHalfSize) return  cfg_.arenaHalfSize;
        if (v < -cfg_.arenaHalfSize) return -cfg_.arenaHalfSize;
        return v;
    };
    hx = clamp(hx); hz = clamp(hz);
    ox = clamp(ox); oz = clamp(oz);

    applyStats(env.hero, cfg_);
    applyStats(env.opponent, cfg_);
    env.hero.setPosition(hx, hz);
    env.opponent.setPosition(ox, oz);

    env.world.clearEvents();
    env.heroRT.reset(env.hero, env.world);
    env.oppRT.reset(env.opponent, env.world);

    env.stepCount  = 0;
    env.doneFlag   = false;
    env.winner     = -1;
    env.heroReward = 0.0f;
    env.oppReward  = 0.0f;
}

void VecSimulation::seedAndReset(uint64_t baseSeed) {
    for (size_t i = 0; i < envs_.size(); i++) {
        envs_[i]->rng.seed(baseSeed + static_cast<uint64_t>(i));
        envs_[i]->world.seed(baseSeed + 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(i));
        envs_[i]->episodeIdx = 0;
        resetEnvImpl_(*envs_[i]);
    }
}

void VecSimulation::resetDone() {
    for (auto& e : envs_) {
        if (e->doneFlag) {
            e->episodeIdx++;
            resetEnvImpl_(*e);
        }
    }
}

void VecSimulation::resetEnv(int envIdx) {
    if (envIdx < 0 || envIdx >= cfg_.numEnvs) return;
    envs_[envIdx]->episodeIdx++;
    resetEnvImpl_(*envs_[envIdx]);
}

void VecSimulation::observe(int agentId, float* out) const {
    const int stride = observation::TOTAL;
    for (size_t i = 0; i < envs_.size(); i++) {
        const Agent& a = (agentId == HERO_ID) ? envs_[i]->hero : envs_[i]->opponent;
        observation::build(a, envs_[i]->world, out + i * stride);
    }
}

void VecSimulation::actionMask(int agentId, float* outMask, int* outEnemyIds) const {
    const int mStride  = action_mask::TOTAL;
    const int idStride = action_mask::N_ENEMY_SLOTS;
    for (size_t i = 0; i < envs_.size(); i++) {
        const Agent& a = (agentId == HERO_ID) ? envs_[i]->hero : envs_[i]->opponent;
        action_mask::build(a, envs_[i]->world,
                           outMask + i * mStride,
                           outEnemyIds + i * idStride);
    }
}

void VecSimulation::applyActions(int agentId, const AgentAction* actions) {
    for (size_t i = 0; i < envs_.size(); i++) {
        EnvState& e = *envs_[i];
        Agent& a = (agentId == HERO_ID) ? e.hero : e.opponent;
        if (!a.unit().alive()) continue;
        e.world.applyAction(a, actions[i], cfg_.dt);
    }
}

void VecSimulation::step() {
    for (size_t i = 0; i < envs_.size(); i++) {
        EnvState& e = *envs_[i];

        // Advance projectiles and clean up dead ones. Agent cooldowns and
        // movement were already integrated by applyActions (Agent::applyAction
        // ticks cooldowns internally). DoT/HoT, however, must be ticked here
        // because they emit events / mutate HP and aren't policy-driven.
        e.world.applyDotHot(e.hero,     cfg_.dt);
        e.world.applyDotHot(e.opponent, cfg_.dt);
        e.world.stepProjectiles(cfg_.dt);
        e.world.cullProjectiles();
        e.stepCount++;

        // Drain reward deltas into accumulators.
        auto dh = e.heroRT.consume(e.hero, e.world);
        auto doo = e.oppRT.consume(e.opponent, e.world);
        e.heroReward += cfg_.rewardDamageDealt    * dh.damageDealt
                      + cfg_.rewardDamageTakenMul * dh.damageTaken
                      + cfg_.rewardKill           * dh.kills
                      + cfg_.rewardDeath          * dh.deaths
                      + cfg_.rewardStep;
        e.oppReward  += cfg_.rewardDamageDealt    * doo.damageDealt
                      + cfg_.rewardDamageTakenMul * doo.damageTaken
                      + cfg_.rewardKill           * doo.kills
                      + cfg_.rewardDeath          * doo.deaths
                      + cfg_.rewardStep;

        // Termination check. Winner = the surviving alive agent; -1 if both
        // alive (timeout) or both dead (mutual KO this step). Any winner==-1
        // outcome is treated as a loss for BOTH sides via rewardTimeout —
        // standing still must be strictly worse than fighting.
        if (!e.doneFlag) {
            bool heroAlive = e.hero.unit().alive();
            bool oppAlive  = e.opponent.unit().alive();
            bool terminal  = (!heroAlive || !oppAlive)
                          || (e.stepCount >= cfg_.maxStepsPerEpisode);
            if (terminal) {
                e.doneFlag = true;
                if (heroAlive && !oppAlive)      e.winner = HERO_ID;
                else if (oppAlive && !heroAlive) e.winner = OPPONENT_ID;
                else                             e.winner = -1;

                if (e.winner < 0) {
                    e.heroReward += cfg_.rewardTimeout;
                    e.oppReward  += cfg_.rewardTimeout;
                }
            }
        }
    }
}

void VecSimulation::dones(int* outDone, int* outWinner) const {
    for (size_t i = 0; i < envs_.size(); i++) {
        outDone[i]   = envs_[i]->doneFlag ? 1 : 0;
        outWinner[i] = envs_[i]->winner;
    }
}

void VecSimulation::rewards(float* outHero, float* outOpponent) {
    for (size_t i = 0; i < envs_.size(); i++) {
        outHero[i]     = envs_[i]->heroReward;
        outOpponent[i] = envs_[i]->oppReward;
        envs_[i]->heroReward = 0.0f;
        envs_[i]->oppReward  = 0.0f;
    }
}

void VecSimulation::stepCounts(int* out) const {
    for (size_t i = 0; i < envs_.size(); i++) out[i] = envs_[i]->stepCount;
}

void VecSimulation::episodeCounts(int* out) const {
    for (size_t i = 0; i < envs_.size(); i++) out[i] = envs_[i]->episodeIdx;
}

World& VecSimulation::world(int i) { return envs_[i]->world; }
const World& VecSimulation::world(int i) const { return envs_[i]->world; }
Agent& VecSimulation::hero(int i) { return envs_[i]->hero; }
Agent& VecSimulation::opponent(int i) { return envs_[i]->opponent; }

} // namespace brogameagent
