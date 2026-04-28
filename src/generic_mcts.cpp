#include "brogameagent/generic_mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace brogameagent::mcts {

// ─── Node ──────────────────────────────────────────────────────────────────
// Visits / values / priors are stored as parallel vectors indexed by action,
// mirroring the JS reference implementation. Children are owned via
// unique_ptr so subtree promotion (advance_root) is a single move.

class GenericMcts::Node {
public:
    std::vector<std::unique_ptr<Node>> children; // size = num_actions
    std::vector<uint8_t> legal;                   // 1 if action is legal at this state
    std::vector<float>   P;                       // prior probabilities, normalized
    std::vector<int>     N;                       // visits per action
    std::vector<float>   W;                       // sum-of-discounted-returns per action

    int   visits             = 0;     // sum of N[a]
    bool  expanded           = false;
    bool  terminal           = false;
    float reward_from_parent = 0.0f;  // r(s_parent, a → this); used in backprop

    void init(int num_actions) {
        children.resize(num_actions);
        legal.assign(num_actions, 0);
        P.assign(num_actions, 0.0f);
        N.assign(num_actions, 0);
        W.assign(num_actions, 0.0f);
    }
};

namespace {

// Marsaglia–Tsang Gamma(α, 1) sampler. Used for Dirichlet root noise.
// Recursive boost identity handles α<1 by sampling Gamma(α+1) · U^(1/α).
float sample_gamma(std::mt19937_64& rng, float alpha) {
    if (alpha < 1.0f) {
        std::uniform_real_distribution<float> u01(1e-12f, 1.0f);
        return sample_gamma(rng, alpha + 1.0f)
             * std::pow(u01(rng), 1.0f / alpha);
    }
    const float d = alpha - 1.0f / 3.0f;
    const float c = 1.0f / std::sqrt(9.0f * d);
    std::normal_distribution<float>       normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (;;) {
        float x;
        float v;
        do {
            x = normal(rng);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;
        const float u = u01(rng);
        if (u < 1.0f - 0.0331f * x * x * x * x) return d * v;
        if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v))) return d * v;
    }
}

// Symmetric Dirichlet(α) over a subset of legal action indices. Output is
// sized num_actions; entries off the legal set stay zero.
std::vector<float> sample_dirichlet(std::mt19937_64& rng,
                                    const std::vector<int>& legal,
                                    int num_actions, float alpha) {
    std::vector<float> out(static_cast<size_t>(num_actions), 0.0f);
    if (legal.empty()) return out;
    float sum = 0.0f;
    for (int a : legal) {
        const float g = sample_gamma(rng, alpha);
        out[static_cast<size_t>(a)] = g;
        sum += g;
    }
    if (sum > 0.0f) {
        for (int a : legal) out[static_cast<size_t>(a)] /= sum;
    } else {
        const float u = 1.0f / static_cast<float>(legal.size());
        for (int a : legal) out[static_cast<size_t>(a)] = u;
    }
    return out;
}

// PUCT child selection. Returns -1 only if no legal action exists at all.
int pick_action(const GenericMcts::Node& node, float c_puct) {
    const float sqrt_parent =
        std::sqrt(static_cast<float>(std::max(1, node.visits)));
    int   best_a     = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    const int n = static_cast<int>(node.legal.size());
    for (int a = 0; a < n; ++a) {
        if (!node.legal[a]) continue;
        const int   Na = node.N[a];
        const float Q  = Na > 0
            ? node.W[a] / static_cast<float>(Na)
            : 0.0f;
        const float U = c_puct * node.P[a] * sqrt_parent
                      / (1.0f + static_cast<float>(Na));
        const float s = Q + U;
        if (s > best_score) {
            best_score = s;
            best_a     = a;
        }
    }
    return best_a;
}

} // namespace

// ─── ctor / dtor / move ─────────────────────────────────────────────────────
// Out-of-line because Node is incomplete in the header.

GenericMcts::GenericMcts(GenericEnv env)
    : env_(std::move(env)) {}

GenericMcts::~GenericMcts() = default;
GenericMcts::GenericMcts(GenericMcts&&) noexcept = default;
GenericMcts& GenericMcts::operator=(GenericMcts&&) noexcept = default;

void GenericMcts::reset() {
    root_.reset();
    tree_size_ = 0;
}

void GenericMcts::advance_root(int action) {
    if (!root_ || action < 0 || action >= env_.num_actions) {
        reset();
        return;
    }
    auto& slot = root_->children[static_cast<size_t>(action)];
    if (!slot) { reset(); return; }
    root_ = std::move(slot);
    // Tree size becomes approximate after move; the next search will treat
    // it as the start-of-search baseline. Cheap to live with.
}

std::vector<float> GenericMcts::root_visits() const {
    std::vector<float> out(static_cast<size_t>(env_.num_actions), 0.0f);
    if (!root_) return out;
    float s = 0.0f;
    for (int a = 0; a < env_.num_actions; ++a) {
        out[static_cast<size_t>(a)] = static_cast<float>(root_->N[a]);
        s += out[static_cast<size_t>(a)];
    }
    if (s > 0.0f) {
        for (auto& v : out) v /= s;
    }
    return out;
}

int GenericMcts::search() {
    const int num_actions = env_.num_actions;
    if (num_actions <= 0) return -1;

    if (!root_) {
        root_ = std::make_unique<Node>();
        root_->init(num_actions);
        tree_size_ = 1;
    }

    auto expand = [&](Node& node) {
        const auto legal = env_.legal_actions_fn();
        for (int a : legal) {
            if (a >= 0 && a < num_actions) {
                node.legal[static_cast<size_t>(a)] = 1;
            }
        }
        if (prior_fn_) {
            const auto obs   = env_.observe_fn();
            const auto probs = prior_fn_(obs, legal);
            float s = 0.0f;
            for (int a = 0; a < num_actions; ++a) {
                if (!node.legal[a]) continue;
                const float p = (a < static_cast<int>(probs.size()))
                    ? std::max(0.0f, probs[static_cast<size_t>(a)])
                    : 0.0f;
                node.P[a]  = p;
                s         += p;
            }
            if (s > 0.0f) {
                for (int a = 0; a < num_actions; ++a) node.P[a] /= s;
            } else if (!legal.empty()) {
                // Net produced all zeros over the legal set — fall back to
                // uniform so PUCT still has signal.
                const float u = 1.0f / static_cast<float>(legal.size());
                for (int a : legal) node.P[static_cast<size_t>(a)] = u;
            }
        } else if (!legal.empty()) {
            const float u = 1.0f / static_cast<float>(legal.size());
            for (int a : legal) node.P[static_cast<size_t>(a)] = u;
        }
        node.expanded = true;
    };

    std::mt19937_64 rng(cfg_.seed);

    auto random_rollout = [&]() -> float {
        float g = 0.0f;
        float discount = 1.0f;
        for (int i = 0; i < cfg_.rollout_depth; ++i) {
            const auto legal = env_.legal_actions_fn();
            if (legal.empty()) break;
            std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
            const int a = legal[d(rng)];
            const auto out = env_.step_fn(a);
            g        += discount * out.reward;
            discount *= cfg_.gamma;
            if (out.done) return g;
        }
        return g;
    };

    // One snapshot covers the whole search.
    const std::any root_snap = env_.snapshot_fn();

    Node& root = *root_;
    if (!root.expanded) expand(root);

    // Mix Dirichlet noise into the root prior post-expansion so we operate
    // on the normalized prior. Only entries on the legal set are touched.
    if (cfg_.dirichlet_alpha > 0.0f && cfg_.dirichlet_epsilon > 0.0f) {
        std::vector<int> legal_idx;
        legal_idx.reserve(static_cast<size_t>(num_actions));
        for (int a = 0; a < num_actions; ++a) {
            if (root.legal[static_cast<size_t>(a)]) legal_idx.push_back(a);
        }
        if (!legal_idx.empty()) {
            const auto noise = sample_dirichlet(
                rng, legal_idx, num_actions, cfg_.dirichlet_alpha);
            const float eps = cfg_.dirichlet_epsilon;
            for (int a : legal_idx) {
                root.P[static_cast<size_t>(a)] =
                    (1.0f - eps) * root.P[static_cast<size_t>(a)]
                  + eps * noise[static_cast<size_t>(a)];
            }
        }
    }

    struct PathStep { Node* parent; int action; };
    std::vector<PathStep> path;
    path.reserve(64);

    int it = 0;
    for (; it < cfg_.iterations; ++it) {
        env_.restore_fn(root_snap);
        path.clear();

        Node* node = &root;
        while (true) {
            const int a = pick_action(*node, cfg_.c_puct);
            if (a < 0) break; // no legal action at this state
            path.push_back({node, a});
            auto& slot = node->children[static_cast<size_t>(a)];
            if (!slot) {
                // Edge unexpanded: take the action and create the child.
                const auto out = env_.step_fn(a);
                auto child = std::make_unique<Node>();
                child->init(num_actions);
                child->reward_from_parent = out.reward;
                child->terminal           = out.done;
                slot = std::move(child);
                tree_size_++;
                node = slot.get();
                break;
            }
            // Replay the action on the live env to descend.
            env_.step_fn(a);
            node = slot.get();
            if (node->terminal) break;
        }

        float leaf_value = 0.0f;
        if (!node->terminal) {
            if (!node->expanded) expand(*node);
            leaf_value = value_fn_
                ? value_fn_(env_.observe_fn())
                : random_rollout();
        }

        // Backup. Each edge contributes its r(s, a); the leaf value is
        // bootstrapped at the bottom. Walking backward keeps Q(s, a) as
        // the discounted return starting from taking a in s.
        float g = leaf_value;
        for (auto i = path.rbegin(); i != path.rend(); ++i) {
            Node*       parent = i->parent;
            const int   a      = i->action;
            const Node* child  = parent->children[static_cast<size_t>(a)].get();
            g = child->reward_from_parent + cfg_.gamma * g;
            parent->N[static_cast<size_t>(a)] += 1;
            parent->W[static_cast<size_t>(a)] += g;
            parent->visits                    += 1;
        }
    }

    // Leave env where the caller had it.
    env_.restore_fn(root_snap);

    int best_a = -1;
    int best_n = -1;
    for (int a = 0; a < num_actions; ++a) {
        if (root.N[static_cast<size_t>(a)] > best_n) {
            best_n = root.N[static_cast<size_t>(a)];
            best_a = a;
        }
    }
    stats_.iterations  = it;
    stats_.tree_size   = tree_size_;
    stats_.best_visits = best_n;
    stats_.best_action = best_a;
    return best_a;
}

} // namespace brogameagent::mcts
