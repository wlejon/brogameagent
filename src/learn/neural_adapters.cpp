#include "brogameagent/learn/neural_adapters.h"
#include "brogameagent/action_mask.h"
#include "brogameagent/agent.h"
#include "brogameagent/observation.h"
#include "brogameagent/world.h"
#include "brogameagent/nn/heads.h"

#include <algorithm>
#include <cmath>

namespace brogameagent::learn {

void NeuralEvaluator::maybe_reload_() const {
    if (!handle_) return;
    uint64_t v = 0;
    auto blob = handle_->snapshot(&v);
    if (!blob || v == loaded_version_) return;
    net_->load(*blob);
    loaded_version_ = v;
}

float NeuralEvaluator::evaluate(const World& world, int heroId) const {
    maybe_reload_();
    const Agent* hero = world.findById(heroId);
    if (!hero) return 0.0f;
    if (!hero->unit().alive()) return -1.0f;

    nn::Tensor obs = nn::Tensor::vec(observation::TOTAL);
    observation::build(*hero, world, obs.ptr());

    float v = 0.0f;
    nn::Tensor logits = nn::Tensor::vec(net_->policy_logits());
    // SingleHeroNet is not thread-safe due to activation caches; callers use
    // one evaluator per thread if parallel. const_cast is safe w.r.t. the
    // library's contract — evaluate() is logically const but the net's
    // internal scratch is mutable.
    const_cast<nn::SingleHeroNet&>(*net_).forward(obs, v, logits);
    if (v >  1.0f) v =  1.0f;
    if (v < -1.0f) v = -1.0f;
    return v;
}

void NeuralPrior::maybe_reload_() const {
    if (!handle_) return;
    uint64_t v = 0;
    auto blob = handle_->snapshot(&v);
    if (!blob || v == loaded_version_) return;
    net_->load(*blob);
    loaded_version_ = v;
}

std::vector<float> NeuralPrior::score(
    const Agent& self, const World& world,
    const std::vector<mcts::CombatAction>& actions) const
{
    maybe_reload_();
    std::vector<float> out(actions.size(), 0.0f);
    if (actions.empty()) return out;

    // One forward pass.
    nn::Tensor obs = nn::Tensor::vec(observation::TOTAL);
    observation::build(self, world, obs.ptr());

    float v_unused = 0.0f;
    nn::Tensor logits = nn::Tensor::vec(net_->policy_logits());
    const_cast<nn::SingleHeroNet&>(*net_).forward(obs, v_unused, logits);

    // Optional temperature scaling on logits.
    if (temperature_ > 0.0f && std::fabs(temperature_ - 1.0f) > 1e-4f) {
        const float inv = 1.0f / temperature_;
        for (int i = 0; i < logits.size(); ++i) logits[i] *= inv;
    }

    // Build action_mask for the softmax — legal attack/ability slots.
    float attack_mask[action_mask::N_ENEMY_SLOTS] = {0};
    float ability_mask[action_mask::N_ABILITY_SLOTS] = {0};
    int enemy_ids[action_mask::N_ENEMY_SLOTS];
    float mask_buf[action_mask::TOTAL];
    action_mask::build(self, world, mask_buf, enemy_ids);
    for (int i = 0; i < action_mask::N_ENEMY_SLOTS; ++i) attack_mask[i] = mask_buf[i];
    for (int i = 0; i < action_mask::N_ABILITY_SLOTS; ++i)
        ability_mask[i] = mask_buf[action_mask::N_ENEMY_SLOTS + i];

    nn::Tensor probs = nn::Tensor::vec(net_->policy_logits());
    nn::factored_softmax(logits, probs, attack_mask, ability_mask);

    const int N_MOVE = nn::FactoredPolicyHead::N_MOVE;
    const int N_ATK  = nn::FactoredPolicyHead::N_ATTACK;
    const int N_AB   = nn::FactoredPolicyHead::N_ABILITY;
    const float* p_move    = probs.ptr() + 0;
    const float* p_attack  = probs.ptr() + N_MOVE;
    const float* p_ability = probs.ptr() + N_MOVE + N_ATK;

    // Enumerate actions; translate to factor indices.
    // MoveDir enum values 0..8 match N_MOVE. PathToTarget(9)/PathAway(10)
    // collapse into the nearest cardinal dirs — we treat both as "Hold" index
    // for the prior (the scripted rollout will reinterpret them at rollout
    // time). This bias is intentional: the policy head does not model paths.
    for (size_t i = 0; i < actions.size(); ++i) {
        const auto& a = actions[i];
        int mi = static_cast<int>(a.move_dir);
        if (mi < 0 || mi >= N_MOVE) mi = 0;  // treat path-kinds as Hold
        const int ai = (a.attack_slot < 0) ? (N_ATK - 1) : a.attack_slot;
        const int bi = (a.ability_slot < 0) ? (N_AB - 1) : a.ability_slot;

        float joint = p_move[mi] * p_attack[ai] * p_ability[bi];
        // Mix with a uniform prior so exploration doesn't collapse on tight
        // distributions (Dirichlet-lite).
        joint = (1.0f - uniform_mix_) * joint + uniform_mix_ / static_cast<float>(actions.size());
        out[i] = joint;
    }
    return out;
}

} // namespace brogameagent::learn
