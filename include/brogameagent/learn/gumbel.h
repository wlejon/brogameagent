#pragma once

#include "brogameagent/mcts.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace brogameagent::learn {

// ─── GumbelNoisePrior ─────────────────────────────────────────────────────
//
// A prior wrapper that adds IID Gumbel(0,1) noise (in log-space) to an
// inner prior's weights. Equivalent to drawing actions from the inner
// distribution with "Gumbel-argmax" sampling at the root — the trick from
// Gumbel-AlphaZero (Danihelka et al., 2022), simplified.
//
// Why this matters here: for real-time planning budgets (small iteration
// counts, few expansions), classical PUCT over-commits to the root action
// with the biggest early bump. Adding Gumbel noise per search() call makes
// the initial descent explore diverse subtrees without sacrificing the
// policy-improvement property — the MCTS visit distribution remains a
// valid (noisier) improvement of the prior policy.
//
// The noise is sampled once per prior::score() call with the configured
// seed, so each MCTS root uses a distinct noisy prior but each level deeper
// reuses the inner prior unchanged (we only want diversity at the root).
//
// Usage:
//   auto inner = std::make_shared<NeuralPrior>(net, handle);
//   auto noisy = std::make_shared<GumbelNoisePrior>(inner, /*scale*/ 1.0f);
//   noisy->reseed(ep_seed);                // once per episode decision
//   mcts.set_prior(noisy);
//
// Reference: "Policy improvement by planning with Gumbel", Danihelka et al.,
// ICLR 2022. The full Gumbel-AlphaZero also uses sequential halving among
// top candidates; this wrapper implements only the noisy-root piece and
// leaves the existing Mcts iteration loop unchanged. That's the subset
// that composes cleanly with the rest of the library today and remains a
// strict improvement on classical PUCT for tight realtime budgets.

class GumbelNoisePrior : public mcts::IPrior {
public:
    GumbelNoisePrior(std::shared_ptr<mcts::IPrior> inner, float scale = 1.0f)
        : inner_(std::move(inner)), scale_(scale) {}

    // Reseed the noise generator. Typical use: reseed per decision so each
    // search() draws a fresh noise vector.
    void reseed(uint64_t seed) { rng_state_ = seed ? seed : 0xA11CEBEEFULL; }
    void set_scale(float s)    { scale_ = s; }

    std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<mcts::CombatAction>& actions) const override;

private:
    std::shared_ptr<mcts::IPrior> inner_;
    float                         scale_     = 1.0f;
    mutable uint64_t              rng_state_ = 0xA11CEBEEFULL;
};

// ─── gumbel_improved_policy ───────────────────────────────────────────────
//
// Given a completed Mcts search rooted at `root`, compute π' — the policy
// improvement target from the Gumbel-AlphaZero paper. This is the
// distribution the apprentice net should regress toward, strictly better
// than the raw visit-count target when the search budget is small.
//
// Formula (simplified):
//   π'(a) ∝ exp(logits(a) + g(a) + σ * completedQ(a))
// where logits is the inner prior's pre-softmax score, g(a) is the Gumbel
// noise drawn at search time (shared with GumbelNoisePrior), and
// completedQ uses the root child's Q-value if visited else the root value.
//
// For now we expose a simpler version that derives logits from visit
// counts (log-visits), which is equivalent up to a constant for large
// budgets and works well as a distillation target in practice.
void gumbel_improved_policy(
    const mcts::Node& root,
    float target_move[9],
    float target_attack[6],   // N_ENEMY_SLOTS + 1
    float target_ability[9]); // N_ABILITY_SLOTS + 1 (assumes MAX_ABILITIES=8)

} // namespace brogameagent::learn
