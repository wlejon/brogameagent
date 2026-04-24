#pragma once

#include "heads.h"
#include "set_transformer.h"
#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn {

// ─── SingleHeroNetST ──────────────────────────────────────────────────────
//
// Clone of SingleHeroNet using SetTransformerEncoder. Same API surface so
// benchmarks can swap encoders head-to-head.

class SingleHeroNetST {
public:
    struct Config {
        SetTransformerEncoder::Config enc{};
        int trunk_hidden = 64;
        int value_hidden = 32;
        uint64_t seed = 0xC0DE1234ULL;
    };

    SingleHeroNetST() = default;
    void init(const Config& cfg);
    const Config& config() const { return cfg_; }

    void forward(const Tensor& x, float& value, Tensor& logits);
    void backward(float dValue, const Tensor& dLogits);

    void zero_grad();
    void sgd_step(float lr, float momentum);

    int embed_dim() const { return enc_.out_dim(); }
    int trunk_dim() const { return cfg_.trunk_hidden; }
    int policy_logits() const { return head_.total_logits(); }
    int num_params() const;

    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

private:
    Config cfg_{};
    SetTransformerEncoder enc_;
    Linear trunk_;
    Relu   trunk_act_;
    ValueHead value_head_;
    FactoredPolicyHead head_;

    Tensor enc_out_;
    Tensor trunk_raw_, trunk_act_out_;
    Tensor logits_scratch_;
};

} // namespace brogameagent::nn
