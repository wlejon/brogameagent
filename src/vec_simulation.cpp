#include "brogameagent/vec_simulation.h"
#include "brogameagent/observation.h"
#include "brogameagent/action_mask.h"

#include <cmath>

namespace brogameagent {

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
}

static void applyStats(Agent& a, const VecSimulation::Config& cfg) {
    a.unit().hp            = cfg.hp;
    a.unit().maxHp         = cfg.hp;
    a.unit().damage        = cfg.damage;
    a.unit().attackRange   = cfg.attackRange;
    a.unit().attacksPerSec = cfg.attacksPerSec;
    a.unit().moveSpeed     = cfg.moveSpeed;
    a.unit().radius        = cfg.radius;
    a.unit().attackCooldown = 0.0f;
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
        // ticks cooldowns internally).
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
