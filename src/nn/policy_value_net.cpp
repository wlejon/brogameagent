#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

static constexpr uint32_t kMagic   = 0x564E5642; // "BVNV" — Bro policy-Value Net Variant
// v2 adds a length-prefixed head_sizes vector after the version word. v1 blobs
// load as single-head with head_sizes = {num_actions}.
static constexpr uint32_t kVersion = 2;

void PolicyValueNet::init(const Config& cfg) {
    cfg_ = cfg;
    assert(cfg.in_dim > 0);
    assert(!cfg.hidden.empty());
    assert(cfg.value_hidden > 0);

    // Resolve head shape. Default: one head spanning num_actions. With
    // head_sizes provided, num_actions either matches the sum or is auto-
    // populated from it. head_offsets gets a trailing sentinel = num_actions.
    head_sizes_.clear();
    head_offsets_.clear();
    if (cfg.head_sizes.empty()) {
        assert(cfg.num_actions > 0);
        head_sizes_.push_back(cfg.num_actions);
    } else {
        int sum = 0;
        for (int h : cfg.head_sizes) {
            assert(h > 0);
            sum += h;
            head_sizes_.push_back(h);
        }
        if (cfg_.num_actions == 0) cfg_.num_actions = sum;
        else assert(cfg_.num_actions == sum);
    }
    head_offsets_.reserve(head_sizes_.size() + 1);
    int off = 0;
    for (int h : head_sizes_) { head_offsets_.push_back(off); off += h; }
    head_offsets_.push_back(off);
    assert(off == cfg_.num_actions);

    uint64_t seed = cfg_.seed;

    // Trunk.
    trunk_.clear();
    trunk_acts_.clear();
    trunk_raw_.clear();
    trunk_act_.clear();
    trunk_.resize(cfg.hidden.size());
    trunk_acts_.resize(cfg.hidden.size());
    trunk_raw_.resize(cfg.hidden.size());
    trunk_act_.resize(cfg.hidden.size());

    int prev = cfg_.in_dim;
    for (size_t i = 0; i < cfg_.hidden.size(); ++i) {
        trunk_[i].init(prev, cfg_.hidden[i], seed);
        trunk_raw_[i].resize(cfg_.hidden[i], 1);
        trunk_act_[i].resize(cfg_.hidden[i], 1);
        prev = cfg_.hidden[i];
    }

    // Value head.
    v_fc1_.init(prev, cfg_.value_hidden, seed);
    v_fc2_.init(cfg_.value_hidden, 1, seed);
    v_h_raw_.resize(cfg_.value_hidden, 1);
    v_h_act_.resize(cfg_.value_hidden, 1);
    v_pre_tanh_.resize(1, 1);
    v_post_tanh_.resize(1, 1);

    // Policy head: one Linear emitting all heads concatenated.
    p_fc_.init(prev, cfg_.num_actions, seed);
}

void PolicyValueNet::forward(const Tensor& x, float& value, Tensor& logits) {
    // Trunk.
    const Tensor* h = &x;
    for (size_t i = 0; i < trunk_.size(); ++i) {
        trunk_[i].forward(*h, trunk_raw_[i]);
        trunk_acts_[i].forward(trunk_raw_[i], trunk_act_[i]);
        h = &trunk_act_[i];
    }

    // Value head: Linear → ReLU → Linear → tanh.
    v_fc1_.forward(*h, v_h_raw_);
    v_act_.forward(v_h_raw_, v_h_act_);
    v_fc2_.forward(v_h_act_, v_pre_tanh_);
    v_tanh_.forward(v_pre_tanh_, v_post_tanh_);
    value = v_post_tanh_[0];

    // Policy head.
    p_fc_.forward(*h, logits);
}

void PolicyValueNet::backward(float dValue, const Tensor& dLogits) {
    const int trunk_out = trunk_dim();

    // ── Value head backward ───────────────────────────────────────────────
    Tensor dPostTanh = Tensor::vec(1);
    dPostTanh[0] = dValue;
    Tensor dPreTanh = Tensor::vec(1);
    v_tanh_.backward(dPostTanh, dPreTanh);

    Tensor dVAct = Tensor::vec(cfg_.value_hidden);
    v_fc2_.backward(dPreTanh, dVAct);

    Tensor dVRaw = Tensor::vec(cfg_.value_hidden);
    v_act_.backward(dVAct, dVRaw);

    Tensor dTrunkFromV = Tensor::vec(trunk_out);
    v_fc1_.backward(dVRaw, dTrunkFromV);

    // ── Policy head backward ──────────────────────────────────────────────
    Tensor dTrunkFromP = Tensor::vec(trunk_out);
    p_fc_.backward(dLogits, dTrunkFromP);

    // ── Sum gradients into the trunk's last activation ───────────────────
    Tensor dHAct = Tensor::vec(trunk_out);
    for (int i = 0; i < trunk_out; ++i)
        dHAct[i] = dTrunkFromV[i] + dTrunkFromP[i];

    // ── Walk trunk backwards ──────────────────────────────────────────────
    for (int li = static_cast<int>(trunk_.size()) - 1; li >= 0; --li) {
        const int w = cfg_.hidden[li];
        Tensor dHRaw = Tensor::vec(w);
        trunk_acts_[li].backward(dHAct, dHRaw);

        const int prev_w = (li == 0) ? cfg_.in_dim : cfg_.hidden[li - 1];
        Tensor dPrev = Tensor::vec(prev_w);
        trunk_[li].backward(dHRaw, dPrev);

        // dPrev becomes dHAct for the next iteration; for li==0 it's dX which
        // we discard (no upstream).
        dHAct = std::move(dPrev);
    }
}

void PolicyValueNet::zero_grad() {
    for (auto& l : trunk_) l.zero_grad();
    v_fc1_.zero_grad();
    v_fc2_.zero_grad();
    p_fc_.zero_grad();
}

void PolicyValueNet::sgd_step(float lr, float momentum) {
    for (auto& l : trunk_) l.sgd_step(lr, momentum);
    v_fc1_.sgd_step(lr, momentum);
    v_fc2_.sgd_step(lr, momentum);
    p_fc_.sgd_step(lr, momentum);
}

int PolicyValueNet::num_params() const {
    int n = 0;
    for (const auto& l : trunk_) n += l.num_params();
    n += v_fc1_.num_params();
    n += v_fc2_.num_params();
    n += p_fc_.num_params();
    return n;
}

std::vector<uint8_t> PolicyValueNet::save() const {
    std::vector<uint8_t> out;
    out.resize(sizeof(uint32_t) * 2);
    std::memcpy(out.data(),                    &kMagic,   sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t), &kVersion, sizeof(uint32_t));

    // v2: length-prefixed head_sizes. Single-head nets serialize as
    // {num_actions} so loaders can validate without a special case.
    const uint32_t n_heads = static_cast<uint32_t>(head_sizes_.size());
    const size_t base = out.size();
    out.resize(base + sizeof(uint32_t) * (1 + n_heads));
    std::memcpy(out.data() + base, &n_heads, sizeof(uint32_t));
    for (uint32_t i = 0; i < n_heads; ++i) {
        const uint32_t h = static_cast<uint32_t>(head_sizes_[i]);
        std::memcpy(out.data() + base + sizeof(uint32_t) * (1 + i), &h, sizeof(uint32_t));
    }

    for (const auto& l : trunk_) l.save_to(out);
    v_fc1_.save_to(out);
    v_fc2_.save_to(out);
    p_fc_.save_to(out);
    return out;
}

void PolicyValueNet::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= sizeof(uint32_t) * 2);
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic,   blob.data(),                    sizeof(uint32_t));
    std::memcpy(&version, blob.data() + sizeof(uint32_t), sizeof(uint32_t));
    assert(magic == kMagic);
    assert(version == 1 || version == 2);
    size_t offset = sizeof(uint32_t) * 2;

    if (version >= 2) {
        // Validate that the saved head shape matches what init() resolved.
        // A loader that wants to *adopt* a different shape should re-init
        // before calling load(); we don't reshape silently.
        assert(blob.size() >= offset + sizeof(uint32_t));
        uint32_t n_heads = 0;
        std::memcpy(&n_heads, blob.data() + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        assert(n_heads == head_sizes_.size());
        assert(blob.size() >= offset + sizeof(uint32_t) * n_heads);
        for (uint32_t i = 0; i < n_heads; ++i) {
            uint32_t h = 0;
            std::memcpy(&h, blob.data() + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            assert(static_cast<int>(h) == head_sizes_[i]);
        }
    }
    // v1 has no head metadata — we treat the saved net as single-head with
    // head_sizes == {num_actions}. The current init() must match this if the
    // caller wants to load a v1 blob.

    for (auto& l : trunk_) l.load_from(blob.data(), offset, blob.size());
    v_fc1_.load_from(blob.data(), offset, blob.size());
    v_fc2_.load_from(blob.data(), offset, blob.size());
    p_fc_.load_from(blob.data(), offset, blob.size());
}

} // namespace brogameagent::nn
