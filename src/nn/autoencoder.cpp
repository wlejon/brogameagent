#include "brogameagent/nn/autoencoder.h"

#include <cassert>
#include <cstring>

namespace brogameagent::nn {

static constexpr uint32_t kAeMagic   = 0x45414742; // "BGAE" LE
static constexpr uint32_t kAeVersion = 1;

void DeepSetsAutoencoder::init(const Config& cfg) {
    cfg_ = cfg;
    uint64_t seed = cfg.seed;
    enc_.init(cfg.enc, seed);
    DeepSetsDecoder::Config dcfg;
    dcfg.embed_dim = cfg.enc.embed_dim;
    dcfg.hidden    = cfg.dec_hidden;
    dec_.init(dcfg, seed);

    embed_.resize(enc_.out_dim(), 1);
    dEmbed_.resize(enc_.out_dim(), 1);
}

void DeepSetsAutoencoder::forward(const Tensor& x, Tensor& x_hat) {
    enc_.forward(x, embed_);
    dec_.forward(embed_, x_hat);
}

void DeepSetsAutoencoder::backward(const Tensor& dX_hat) {
    dec_.backward(dX_hat, dEmbed_);
    Tensor dX_obs = Tensor::vec(observation::TOTAL);
    enc_.backward(dEmbed_, dX_obs);
    // dX_obs discarded — no upstream consumer for the raw observation grad.
}

void DeepSetsAutoencoder::zero_grad() {
    enc_.zero_grad();
    dec_.zero_grad();
}

void DeepSetsAutoencoder::sgd_step(float lr, float momentum) {
    enc_.sgd_step(lr, momentum);
    dec_.sgd_step(lr, momentum);
}

int DeepSetsAutoencoder::num_params() const {
    return enc_.num_params() + dec_.num_params();
}

std::vector<uint8_t> DeepSetsAutoencoder::save() const {
    std::vector<uint8_t> out;
    const size_t header = sizeof(uint32_t) * 2;
    out.resize(header);
    std::memcpy(out.data(), &kAeMagic, sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t), &kAeVersion, sizeof(uint32_t));
    enc_.save_to(out);
    dec_.save_to(out);
    return out;
}

void DeepSetsAutoencoder::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= sizeof(uint32_t) * 2);
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic,   blob.data(),                   sizeof(uint32_t));
    std::memcpy(&version, blob.data() + sizeof(uint32_t), sizeof(uint32_t));
    assert(magic == kAeMagic);
    assert(version == kAeVersion);
    size_t offset = sizeof(uint32_t) * 2;
    enc_.load_from(blob.data(), offset, blob.size());
    dec_.load_from(blob.data(), offset, blob.size());
}

std::vector<uint8_t> DeepSetsAutoencoder::save_encoder() const {
    std::vector<uint8_t> out;
    enc_.save_to(out);
    return out;
}

// ─── reconstruction_loss ──────────────────────────────────────────────────

float reconstruction_loss(const Tensor& x, const Tensor& x_hat, Tensor& dX_hat) {
    assert(x.size() == observation::TOTAL);
    assert(x_hat.size() == observation::TOTAL);
    assert(dX_hat.size() == observation::TOTAL);
    dX_hat.zero();

    double sum_sq = 0.0;
    int    count  = 0;

    // Self block — always contributes.
    for (int j = 0; j < observation::SELF_FEATURES; ++j) {
        const float d = x_hat[j] - x[j];
        sum_sq += 0.5 * d * d;
        dX_hat[j] = d;
        ++count;
    }

    // Enemy slots — mask out invalid slots completely.
    const int off_e = observation::SELF_FEATURES;
    for (int k = 0; k < observation::K_ENEMIES; ++k) {
        const int base = off_e + k * observation::ENEMY_FEATURES;
        const bool valid = x[base] > 0.5f;
        if (!valid) continue;   // zero loss + zero grad for padding
        for (int j = 0; j < observation::ENEMY_FEATURES; ++j) {
            const float d = x_hat[base + j] - x[base + j];
            sum_sq += 0.5 * d * d;
            dX_hat[base + j] = d;
            ++count;
        }
    }

    // Ally slots.
    const int off_a = off_e + observation::K_ENEMIES * observation::ENEMY_FEATURES;
    for (int k = 0; k < observation::K_ALLIES; ++k) {
        const int base = off_a + k * observation::ALLY_FEATURES;
        const bool valid = x[base] > 0.5f;
        if (!valid) continue;
        for (int j = 0; j < observation::ALLY_FEATURES; ++j) {
            const float d = x_hat[base + j] - x[base + j];
            sum_sq += 0.5 * d * d;
            dX_hat[base + j] = d;
            ++count;
        }
    }

    if (count == 0) return 0.0f;
    return static_cast<float>(sum_sq / static_cast<double>(count));
}

void copy_encoder_weights(const DeepSetsAutoencoder& src, SingleHeroNet& dst) {
    auto blob = src.save_encoder();
    dst.load_encoder_only(blob);
}

} // namespace brogameagent::nn
