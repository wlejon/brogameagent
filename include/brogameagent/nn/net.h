#pragma once

#include "encoder.h"
#include "heads.h"
#include "tensor.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brogameagent::nn {

// ─── SingleHeroNet ────────────────────────────────────────────────────────
//
// DeepSetsEncoder  →  trunk Linear + ReLU  →  { ValueHead, FactoredPolicyHead }.
//
// Trunk is one hidden layer of configurable width; keeps depth shallow by
// design. Net returns (value, logits) in one forward pass and expects
// backward to be called once with (dValue, dLogits).

class SingleHeroNet {
public:
    struct Config {
        DeepSetsEncoder::Config enc{};
        int trunk_hidden = 64;
        int value_hidden = 32;
        uint64_t seed = 0xC0DE1234ULL;
    };

    SingleHeroNet() = default;
    void init(const Config& cfg);
    const Config& config() const { return cfg_; }

    // Single-sample forward. `x` is an observation::TOTAL vector.
    // Writes scalar value and total_logits()-sized logits tensor.
    void forward(const Tensor& x, float& value, Tensor& logits);

    // Backward. dValue is scalar grad on the value output; dLogits is the
    // gradient on the logits tensor as produced by (probs - target). After
    // backward, sgd_step() applies the update.
    void backward(float dValue, const Tensor& dLogits);

    void zero_grad();
    void sgd_step(float lr, float momentum);

    int embed_dim() const { return enc_.out_dim(); }
    int trunk_dim() const { return cfg_.trunk_hidden; }
    int policy_logits() const { return head_.total_logits(); }
    int num_params() const;

    // Binary serialization.
    std::vector<uint8_t> save() const;
    void load(const std::vector<uint8_t>& blob);

    // Load only the encoder portion of the net from a blob in
    // DeepSetsEncoder::save_to format (no magic/version header — just the
    // raw encoder sub-blob). Used to pull pretrained AE encoder weights
    // into a fresh SingleHeroNet before supervised fine-tuning.
    void load_encoder_only(const std::vector<uint8_t>& enc_blob);

private:
    Config cfg_{};
    DeepSetsEncoder enc_;
    Linear trunk_;
    Relu   trunk_act_;
    ValueHead value_head_;
    FactoredPolicyHead head_;

    Tensor enc_out_;
    Tensor trunk_raw_, trunk_act_out_;
    Tensor logits_scratch_;
};

// ─── Weights hot-swap ─────────────────────────────────────────────────────
//
// A published weights snapshot is just a byte blob (same format SingleHeroNet
// produces/consumes). Publishers build a new blob and call publish(); readers
// call snapshot() to atomically obtain the current blob. This works across
// threads: publish() is rare, snapshot() is hot. std::shared_ptr gives us
// the RCU-grace property for free.

class WeightsHandle {
public:
    WeightsHandle() = default;

    void publish(std::vector<uint8_t> blob, uint64_t version) {
        auto p = std::make_shared<Entry>();
        p->blob    = std::move(blob);
        p->version = version;
        std::lock_guard<std::mutex> lk(mu_);
        current_ = std::move(p);
    }

    std::shared_ptr<const std::vector<uint8_t>> snapshot(uint64_t* out_version = nullptr) const {
        std::shared_ptr<const Entry> p;
        {
            std::lock_guard<std::mutex> lk(mu_);
            p = current_;
        }
        if (!p) return nullptr;
        if (out_version) *out_version = p->version;
        // wrap the blob reference-counted via aliasing.
        return std::shared_ptr<const std::vector<uint8_t>>(p, &p->blob);
    }

    uint64_t version() const {
        std::lock_guard<std::mutex> lk(mu_);
        return current_ ? current_->version : 0;
    }

private:
    struct Entry {
        std::vector<uint8_t> blob;
        uint64_t version = 0;
    };
    mutable std::mutex mu_;
    std::shared_ptr<const Entry> current_;
};

} // namespace brogameagent::nn
