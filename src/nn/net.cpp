#include "brogameagent/nn/net.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

void SingleHeroNet::init(const Config& cfg) {
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

void SingleHeroNet::forward(const Tensor& x, float& value, Tensor& logits) {
    enc_.forward(x, enc_out_);
    trunk_.forward(enc_out_, trunk_raw_);
    trunk_act_.forward(trunk_raw_, trunk_act_out_);
    value_head_.forward(trunk_act_out_, value);
    head_.forward(trunk_act_out_, logits);
}

void SingleHeroNet::backward(float dValue, const Tensor& dLogits) {
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
    // dX is the gradient on the raw observation; we discard it (no upstream).
}

void SingleHeroNet::zero_grad() {
    enc_.zero_grad();
    trunk_.zero_grad();
    value_head_.zero_grad();
    head_.zero_grad();
}

void SingleHeroNet::sgd_step(float lr, float momentum) {
    enc_.sgd_step(lr, momentum);
    trunk_.sgd_step(lr, momentum);
    value_head_.sgd_step(lr, momentum);
    head_.sgd_step(lr, momentum);
}

void SingleHeroNet::adam_step(float lr, float b1, float b2, float eps, int step) {
    enc_.adam_step(lr, b1, b2, eps, step);
    trunk_.adam_step(lr, b1, b2, eps, step);
    value_head_.adam_step(lr, b1, b2, eps, step);
    head_.adam_step(lr, b1, b2, eps, step);
}

int SingleHeroNet::num_params() const {
    return enc_.num_params() + trunk_.num_params()
         + value_head_.num_params() + head_.num_params();
}

// Format: magic("BGNN") + version + (per-circuit tensor records via circuits).
static constexpr uint32_t kMagic = 0x4E4E4742; // "BGNN" LE
static constexpr uint32_t kVersion = 1;

std::vector<uint8_t> SingleHeroNet::save() const {
    std::vector<uint8_t> out;
    const size_t header = sizeof(uint32_t) * 2;
    out.resize(header);
    std::memcpy(out.data(), &kMagic, sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t), &kVersion, sizeof(uint32_t));
    enc_.save_to(out);
    trunk_.save_to(out);
    value_head_.save_to(out);
    head_.save_to(out);
    return out;
}

void SingleHeroNet::load_encoder_only(const std::vector<uint8_t>& enc_blob) {
    size_t offset = 0;
    enc_.load_from(enc_blob.data(), offset, enc_blob.size());
}

void SingleHeroNet::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= sizeof(uint32_t) * 2);
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic,   blob.data(),                   sizeof(uint32_t));
    std::memcpy(&version, blob.data() + sizeof(uint32_t), sizeof(uint32_t));
    assert(magic == kMagic);
    assert(version == kVersion);
    size_t offset = sizeof(uint32_t) * 2;
    enc_.load_from(blob.data(), offset, blob.size());
    trunk_.load_from(blob.data(), offset, blob.size());
    value_head_.load_from(blob.data(), offset, blob.size());
    head_.load_from(blob.data(), offset, blob.size());
}

} // namespace brogameagent::nn
