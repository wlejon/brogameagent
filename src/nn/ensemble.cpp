#include "brogameagent/nn/ensemble.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace brogameagent::nn {

void EnsembleNet::init(int N, SingleHeroNet::Config base) {
    base_ = base;
    members_.clear();
    members_.resize(N);
    for (int i = 0; i < N; ++i) {
        SingleHeroNet::Config c = base;
        c.seed = base.seed + static_cast<uint64_t>(i);
        members_[i].init(c);
    }
}

void EnsembleNet::forward_mean(const Tensor& x, float& value_mean, float& value_std, Tensor& logits_mean) {
    const int N = num_members();
    assert(N > 0);
    assert(logits_mean.size() == members_[0].policy_logits());
    logits_mean.zero();
    std::vector<float> vals(N, 0.0f);
    Tensor lg = Tensor::vec(members_[0].policy_logits());
    for (int i = 0; i < N; ++i) {
        members_[i].forward(x, vals[i], lg);
        for (int j = 0; j < lg.size(); ++j) logits_mean[j] += lg[j];
    }
    const float inv = 1.0f / static_cast<float>(N);
    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += vals[i];
    mean *= inv;
    float var = 0.0f;
    for (int i = 0; i < N; ++i) { const float d = vals[i] - mean; var += d * d; }
    var *= inv;
    for (int j = 0; j < logits_mean.size(); ++j) logits_mean[j] *= inv;
    value_mean = mean;
    value_std = std::sqrt(var);
}

std::vector<uint8_t> EnsembleNet::save() const {
    std::vector<uint8_t> out;
    const int32_t count = num_members();
    const size_t head = sizeof(int32_t);
    out.resize(head);
    std::memcpy(out.data(), &count, sizeof(int32_t));
    for (const auto& m : members_) {
        auto blob = m.save();
        const int32_t sz = static_cast<int32_t>(blob.size());
        const size_t at = out.size();
        out.resize(at + sizeof(int32_t) + blob.size());
        std::memcpy(out.data() + at, &sz, sizeof(int32_t));
        std::memcpy(out.data() + at + sizeof(int32_t), blob.data(), blob.size());
    }
    return out;
}

void EnsembleNet::load(const std::vector<uint8_t>& blob) {
    assert(blob.size() >= sizeof(int32_t));
    int32_t count = 0;
    std::memcpy(&count, blob.data(), sizeof(int32_t));
    assert(count >= 0);
    // Re-init with current base config to shape members, then overwrite weights.
    init(count, base_);
    size_t offset = sizeof(int32_t);
    for (int i = 0; i < count; ++i) {
        assert(offset + sizeof(int32_t) <= blob.size());
        int32_t sz = 0;
        std::memcpy(&sz, blob.data() + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        assert(offset + static_cast<size_t>(sz) <= blob.size());
        std::vector<uint8_t> member_blob(blob.begin() + offset, blob.begin() + offset + sz);
        members_[i].load(member_blob);
        offset += static_cast<size_t>(sz);
    }
}

bool EnsembleNet::save_file(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto blob = save();
    const size_t w = std::fwrite(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    return w == blob.size();
}

bool EnsembleNet::load_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(static_cast<size_t>(sz));
    const size_t r = std::fread(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    if (r != blob.size()) return false;
    load(blob);
    return true;
}

} // namespace brogameagent::nn
