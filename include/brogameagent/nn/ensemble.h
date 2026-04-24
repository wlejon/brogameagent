#pragma once

#include "net.h"
#include "tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brogameagent::nn {

// ─── EnsembleNet ──────────────────────────────────────────────────────────
//
// Owns N SingleHeroNet members seeded as base.seed + i. forward_mean()
// aggregates value mean/std and logits mean across members (for epistemic
// disagreement signals). forward_one() dispatches to member i for training.

class EnsembleNet {
public:
    EnsembleNet() = default;

    void init(int N, SingleHeroNet::Config base);

    int num_members() const { return static_cast<int>(members_.size()); }
    SingleHeroNet&       member(int i)       { return members_[i]; }
    const SingleHeroNet& member(int i) const { return members_[i]; }

    // Runs all members; value_mean, value_std across members, logits_mean elementwise.
    void forward_mean(const Tensor& x, float& value_mean, float& value_std, Tensor& logits_mean);

    void forward_one(int i, const Tensor& x, float& value, Tensor& logits) {
        members_[i].forward(x, value, logits);
    }

    // Serialization: 4-byte count, then per-member [4-byte size | blob].
    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

    bool save_file(const std::string& path) const;
    bool load_file(const std::string& path);

private:
    std::vector<SingleHeroNet> members_;
    SingleHeroNet::Config base_{};
};

} // namespace brogameagent::nn
