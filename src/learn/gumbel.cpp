#include "brogameagent/learn/gumbel.h"
#include "brogameagent/nn/heads.h"

#include <algorithm>
#include <cmath>

namespace brogameagent::learn {

// splitmix64 — same primitive used in nn::ops. Inline here to avoid header
// churn; both are one-way deterministic RNG so no cross-contamination risk.
static inline uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline float u01(uint64_t& s) {
    return static_cast<float>((splitmix(s) >> 40)) / 16777216.0f;
}
// Gumbel(0,1) sample via inverse CDF: g = -log(-log(U)) for U ~ Uniform(0,1).
// Clamp away from 0/1 to avoid log(0).
static inline float gumbel01(uint64_t& s) {
    float u = u01(s);
    if (u < 1e-7f) u = 1e-7f;
    if (u > 1.0f - 1e-7f) u = 1.0f - 1e-7f;
    return -std::log(-std::log(u));
}

std::vector<float> GumbelNoisePrior::score(
    const Agent& self, const World& world,
    const std::vector<mcts::CombatAction>& actions) const
{
    // Inner scores are weights (not logits). Convert to pseudo-logits via
    // log(w + eps), add Gumbel noise, re-exponentiate. The engine normalises
    // these internally, so absolute scale doesn't matter.
    auto base = inner_->score(self, world, actions);
    std::vector<float> out(base.size(), 0.0f);
    for (size_t i = 0; i < base.size(); ++i) {
        const float w = std::max(base[i], 1e-6f);
        const float logit = std::log(w);
        const float noise = gumbel01(rng_state_) * scale_;
        // Undo log → exp so the engine's sum-normalisation handles it.
        out[i] = std::exp(logit + noise);
    }
    return out;
}

void gumbel_improved_policy(
    const mcts::Node& root,
    float target_move[9],
    float target_attack[6],
    float target_ability[9])
{
    const int N_MOVE = nn::FactoredPolicyHead::N_MOVE;
    const int N_ATK  = nn::FactoredPolicyHead::N_ATTACK;
    const int N_AB   = nn::FactoredPolicyHead::N_ABILITY;

    for (int i = 0; i < N_MOVE; ++i) target_move[i]    = 0.0f;
    for (int i = 0; i < N_ATK;  ++i) target_attack[i]  = 0.0f;
    for (int i = 0; i < N_AB;   ++i) target_ability[i] = 0.0f;

    // π'(a) ∝ exp(log(1 + visits(a)) + completedQ(a))  — simplified form.
    // The "+ log(1 + visits)" is equivalent to visit fractions for high
    // budgets but smoother for small-count children; the Q bump rewards
    // children that were good (not just popular) — closer to the paper's
    // completed-Q advantage.
    struct Score { float logit; int mi, ai, bi; };
    std::vector<Score> scores;
    scores.reserve(root.children.size());
    float maxl = -1e30f;
    for (const auto& c : root.children) {
        const float l = std::log(1.0f + static_cast<float>(c->visits)) + c->mean();
        int mi = static_cast<int>(c->action.move_dir);
        if (mi < 0 || mi >= N_MOVE) mi = 0;
        const int ai = (c->action.attack_slot < 0)  ? (N_ATK - 1) : c->action.attack_slot;
        const int bi = (c->action.ability_slot < 0) ? (N_AB  - 1) : c->action.ability_slot;
        scores.push_back({l, mi, ai, bi});
        if (l > maxl) maxl = l;
    }
    float total = 0.0f;
    for (auto& s : scores) { s.logit = std::exp(s.logit - maxl); total += s.logit; }
    if (total <= 0.0f) return;
    const float inv = 1.0f / total;
    for (const auto& s : scores) {
        const float w = s.logit * inv;
        target_move[s.mi]    += w;
        target_attack[s.ai]  += w;
        target_ability[s.bi] += w;
    }
}

} // namespace brogameagent::learn
