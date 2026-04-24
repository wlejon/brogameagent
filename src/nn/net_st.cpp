#include "brogameagent/nn/net_st.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void SingleHeroNetST::init(const Config& cfg) {
    cfg_ = cfg;
    uint64_t seed = cfg.seed;
    enc_.init(cfg.enc, seed);
    trunk_.init(enc_.out_dim(), cfg.trunk_hidden, seed);
    value_head_.init(cfg.trunk_hidden, cfg.value_hidden, seed);
    head_.init(cfg.trunk_hidden, seed);

    enc_out_.resize(enc_.out_dim(), 1);
    trunk_raw_.resize(cfg.trunk_hidden, 1);
    trunk_act_out_.resize(cfg.trunk_hidden, 1);
    logits_scratch_.resize(head_.total_logits(), 1);
}

void SingleHeroNetST::forward(const Tensor& x, float& value, Tensor& logits) {
    enc_.forward(x, enc_out_);
    trunk_.forward(enc_out_, trunk_raw_);
    trunk_act_.forward(trunk_raw_, trunk_act_out_);
    value_head_.forward(trunk_act_out_, value);
    head_.forward(trunk_act_out_, logits);
}

void SingleHeroNetST::backward(float dValue, const Tensor& dLogits) {
    Tensor dTrunkV = Tensor::vec(cfg_.trunk_hidden);
    value_head_.backward(dValue, dTrunkV);
    Tensor dTrunkP = Tensor::vec(cfg_.trunk_hidden);
    head_.backward(dLogits, dTrunkP);

    Tensor dTrunkAct = Tensor::vec(cfg_.trunk_hidden);
    for (int i = 0; i < cfg_.trunk_hidden; ++i)
        dTrunkAct[i] = dTrunkV[i] + dTrunkP[i];

    Tensor dTrunkRaw = Tensor::vec(cfg_.trunk_hidden);
    trunk_act_.backward(dTrunkAct, dTrunkRaw);

    Tensor dEnc = Tensor::vec(enc_.out_dim());
    trunk_.backward(dTrunkRaw, dEnc);

    Tensor dX = Tensor::vec(observation::TOTAL);
    enc_.backward(dEnc, dX);
}

void SingleHeroNetST::zero_grad() {
    enc_.zero_grad();
    trunk_.zero_grad();
    value_head_.zero_grad();
    head_.zero_grad();
}

void SingleHeroNetST::sgd_step(float lr, float momentum) {
    enc_.sgd_step(lr, momentum);
    trunk_.sgd_step(lr, momentum);
    value_head_.sgd_step(lr, momentum);
    head_.sgd_step(lr, momentum);
}

int SingleHeroNetST::num_params() const {
    return enc_.num_params() + trunk_.num_params()
         + value_head_.num_params() + head_.num_params();
}

static constexpr uint32_t kMagicST = 0x54534742; // "BGST" LE
static constexpr uint32_t kVersionST = 1;

std::vector<uint8_t> SingleHeroNetST::save() const {
    std::vector<uint8_t> out;
    const size_t header = sizeof(uint32_t) * 2;
    out.resize(header);
    std::memcpy(out.data(), &kMagicST, sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t), &kVersionST, sizeof(uint32_t));
    enc_.save_to(out);
    trunk_.save_to(out);
    value_head_.save_to(out);
    head_.save_to(out);
    return out;
}

void SingleHeroNetST::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= sizeof(uint32_t) * 2);
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic,   blob.data(),                   sizeof(uint32_t));
    std::memcpy(&version, blob.data() + sizeof(uint32_t), sizeof(uint32_t));
    assert(magic == kMagicST);
    assert(version == kVersionST);
    size_t offset = sizeof(uint32_t) * 2;
    enc_.load_from(blob.data(), offset, blob.size());
    trunk_.load_from(blob.data(), offset, blob.size());
    value_head_.load_from(blob.data(), offset, blob.size());
    head_.load_from(blob.data(), offset, blob.size());
}

} // namespace brogameagent::nn
