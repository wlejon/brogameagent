#pragma once

#include "brogameagent/mcts.h"
#include "brogameagent/nn/net.h"

#include <memory>
#include <vector>

namespace brogameagent::learn {

// ─── NeuralEvaluator ──────────────────────────────────────────────────────
//
// IEvaluator adapter: returns the value head's prediction on the observation
// of the hero, from the hero's perspective. Value is already in [-1, 1].
//
// Owns a SingleHeroNet; optionally reloads it from a WeightsHandle at the
// start of each evaluate() when the handle version advances. That's how the
// "net changes while the game plays" pattern surfaces at the planner
// boundary: the planner just calls evaluate() and gets the latest weights
// if the trainer published a new version.

class NeuralEvaluator : public mcts::IEvaluator {
public:
    explicit NeuralEvaluator(std::shared_ptr<nn::SingleHeroNet> net,
                             nn::WeightsHandle* handle = nullptr)
        : net_(std::move(net)), handle_(handle) {}

    float evaluate(const World& world, int heroId) const override;

private:
    void maybe_reload_() const;

    mutable std::shared_ptr<nn::SingleHeroNet> net_;
    nn::WeightsHandle*                         handle_ = nullptr;
    mutable uint64_t                           loaded_version_ = 0;
};

// ─── NeuralPrior ──────────────────────────────────────────────────────────
//
// IPrior adapter: scores a candidate CombatAction vector by running one
// forward pass, softmaxing the factored heads against the legal-action mask,
// and multiplying the per-factor probabilities.
//
// Joint prior P(a) = p_move[a.move_dir] * p_attack[a.attack_slot-or-no-op]
//                   * p_ability[a.ability_slot-or-no-op]
//
// The engine normalizes priors internally, so absolute magnitudes don't need
// to sum to 1. Zero priors are allowed on actions the network considers
// near-impossible; the engine falls back to uniform if everything is 0.

class NeuralPrior : public mcts::IPrior {
public:
    explicit NeuralPrior(std::shared_ptr<nn::SingleHeroNet> net,
                         nn::WeightsHandle* handle = nullptr)
        : net_(std::move(net)), handle_(handle) {}

    std::vector<float> score(
        const Agent& self, const World& world,
        const std::vector<mcts::CombatAction>& actions) const override;

    void set_temperature(float t) { temperature_ = t; }
    void set_uniform_mix(float m) { uniform_mix_ = m; }

private:
    void maybe_reload_() const;

    mutable std::shared_ptr<nn::SingleHeroNet> net_;
    nn::WeightsHandle*                         handle_ = nullptr;
    mutable uint64_t                           loaded_version_ = 0;
    float                                      temperature_    = 1.0f;
    float                                      uniform_mix_    = 0.05f;  // Dirichlet-lite
};

} // namespace brogameagent::learn
