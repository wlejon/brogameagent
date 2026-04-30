#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/ops.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#endif

#include <cassert>
#include <cstring>
#include <utility>

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

#ifdef BGA_HAS_CUDA
void PolicyValueNet::forward(const gpu::GpuTensor& x, gpu::GpuTensor& logits) {
    assert(device_ == Device::GPU);

    // Trunk: Linear → ReLU per layer. Caches owned by `this`.
    const gpu::GpuTensor* h = &x;
    for (size_t i = 0; i < trunk_.size(); ++i) {
        trunk_[i].forward(*h, trunk_raw_g_[i]);
        gpu::relu_forward_gpu(trunk_raw_g_[i], trunk_act_g_[i]);
        h = &trunk_act_g_[i];
    }

    // Value head: Linear → ReLU → Linear → tanh. Result lives in
    // v_post_tanh_g_ (accessible via value_gpu()).
    v_fc1_.forward(*h, v_h_raw_g_);
    gpu::relu_forward_gpu(v_h_raw_g_, v_h_act_g_);
    v_fc2_.forward(v_h_act_g_, v_pre_tanh_g_);
    gpu::tanh_forward_gpu(v_pre_tanh_g_, v_post_tanh_g_);

    // Policy head.
    p_fc_.forward(*h, logits);
}

void PolicyValueNet::forward_batched(const gpu::GpuTensor& X_BD,
                                     gpu::GpuTensor& logits_BL,
                                     gpu::GpuTensor& values_B1) {
    assert(device_ == Device::GPU);
    const int B = X_BD.rows;

    // Lazily size the batched scratch vector to match trunk depth.
    if (trunk_raw_bg_.size() != trunk_.size()) {
        trunk_raw_bg_.clear();
        trunk_act_bg_.clear();
        trunk_raw_bg_.resize(trunk_.size());
        trunk_act_bg_.resize(trunk_.size());
    }

    // Trunk: Linear(W, b) → ReLU per layer. Each call resizes its output to
    // (B, hidden[i]) automatically.
    const gpu::GpuTensor* h = &X_BD;
    for (size_t i = 0; i < trunk_.size(); ++i) {
        gpu::linear_forward_batched_gpu(trunk_[i].W_g(), trunk_[i].b_g(),
                                        *h, trunk_raw_bg_[i]);
        gpu::relu_forward_batched_gpu(trunk_raw_bg_[i], trunk_act_bg_[i]);
        h = &trunk_act_bg_[i];
    }

    // Value head: Linear → ReLU → Linear → Tanh, all batched.
    gpu::linear_forward_batched_gpu(v_fc1_.W_g(), v_fc1_.b_g(), *h, v_h_raw_bg_);
    gpu::relu_forward_batched_gpu(v_h_raw_bg_, v_h_act_bg_);
    gpu::linear_forward_batched_gpu(v_fc2_.W_g(), v_fc2_.b_g(),
                                    v_h_act_bg_, v_pre_tanh_bg_);
    if (values_B1.rows != B || values_B1.cols != 1) values_B1.resize(B, 1);
    gpu::tanh_forward_batched_gpu(v_pre_tanh_bg_, values_B1);

    // Policy head.
    gpu::linear_forward_batched_gpu(p_fc_.W_g(), p_fc_.b_g(), *h, logits_BL);
}

void PolicyValueNet::forward_batched_train(const gpu::GpuTensor& X_BD,
                                           gpu::GpuTensor& logits_BL,
                                           gpu::GpuTensor& values_B1) {
    assert(device_ == Device::GPU);
    const int B = X_BD.rows;

    if (trunk_raw_btr_g_.size() != trunk_.size()) {
        trunk_raw_btr_g_.clear();
        trunk_act_btr_g_.clear();
        trunk_raw_btr_g_.resize(trunk_.size());
        trunk_act_btr_g_.resize(trunk_.size());
    }

    // Trunk: Linear (caches X view) → ReLU.
    const gpu::GpuTensor* h = &X_BD;
    for (size_t i = 0; i < trunk_.size(); ++i) {
        trunk_[i].forward_batched_train(*h, trunk_raw_btr_g_[i]);
        gpu::relu_forward_batched_gpu(trunk_raw_btr_g_[i], trunk_act_btr_g_[i]);
        h = &trunk_act_btr_g_[i];
    }

    // Value head: Linear → ReLU → Linear → Tanh.
    v_fc1_.forward_batched_train(*h, v_h_raw_btr_g_);
    gpu::relu_forward_batched_gpu(v_h_raw_btr_g_, v_h_act_btr_g_);
    v_fc2_.forward_batched_train(v_h_act_btr_g_, v_pre_tanh_btr_g_);
    if (values_B1.rows != B || values_B1.cols != 1) values_B1.resize(B, 1);
    gpu::tanh_forward_batched_gpu(v_pre_tanh_btr_g_, values_B1);
    // Stash post-tanh (== values_B1) for tanh_backward via a non-owning view.
    v_post_tanh_btr_g_ = gpu::GpuTensor::view(values_B1.data,
                                              values_B1.rows, values_B1.cols);

    // Policy head: Linear (no activation).
    p_fc_.forward_batched_train(*h, logits_BL);
}

void PolicyValueNet::backward_batched(const gpu::GpuTensor& dLogits_BL,
                                      const gpu::GpuTensor& dValues_B1) {
    assert(device_ == Device::GPU);

    // ── Value head backward ───────────────────────────────────────────────
    gpu::tanh_backward_batched_gpu(v_post_tanh_btr_g_, dValues_B1, dPreTanh_btr_g_);
    v_fc2_.backward_batched(dPreTanh_btr_g_, dVAct_btr_g_);
    gpu::relu_backward_batched_gpu(v_h_raw_btr_g_, dVAct_btr_g_, dVRaw_btr_g_);
    // Write directly into dHAct so we can fold in the policy-head gradient
    // afterwards via add_inplace.
    v_fc1_.backward_batched(dVRaw_btr_g_, dHAct_btr_g_);

    // ── Policy head backward ──────────────────────────────────────────────
    p_fc_.backward_batched(dLogits_BL, dTrunkFromP_btr_g_);
    gpu::add_inplace_batched_gpu(dHAct_btr_g_, dTrunkFromP_btr_g_);

    // ── Walk trunk backwards. Ping-pong between dHAct and dPrev ──────────
    gpu::GpuTensor* cur  = &dHAct_btr_g_;
    gpu::GpuTensor* next = &dPrev_btr_g_;
    for (int li = static_cast<int>(trunk_.size()) - 1; li >= 0; --li) {
        gpu::relu_backward_batched_gpu(trunk_raw_btr_g_[li], *cur, dHRaw_btr_g_);
        if (li == 0) {
            trunk_[li].backward_batched(dHRaw_btr_g_, dXdiscard_btr_g_);
        } else {
            trunk_[li].backward_batched(dHRaw_btr_g_, *next);
            std::swap(cur, next);
        }
    }
}

void PolicyValueNet::backward(const gpu::GpuTensor& dLogits) {
    assert(device_ == Device::GPU);

    // ── Value head backward ───────────────────────────────────────────────
    // Caller wrote d(loss)/d(value) into dPostTanh_g_ via dValue_gpu().
    gpu::tanh_backward_gpu(v_post_tanh_g_, dPostTanh_g_, dPreTanh_g_);
    v_fc2_.backward(dPreTanh_g_, dVAct_g_);
    gpu::relu_backward_gpu(v_h_raw_g_, dVAct_g_, dVRaw_g_);
    // Write straight into dHAct_g_ to avoid a device→device copy.
    v_fc1_.backward(dVRaw_g_, dHAct_g_);

    // ── Policy head backward ──────────────────────────────────────────────
    p_fc_.backward(dLogits, dTrunkFromP_g_);

    // dHAct += dTrunkFromP.
    gpu::add_inplace_gpu(dHAct_g_, dTrunkFromP_g_);

    // ── Walk trunk backwards. Ping-pong between dHAct_g_ and dPrev_g_ ────
    gpu::GpuTensor* cur  = &dHAct_g_;
    gpu::GpuTensor* next = &dPrev_g_;
    for (int li = static_cast<int>(trunk_.size()) - 1; li >= 0; --li) {
        gpu::relu_backward_gpu(trunk_raw_g_[li], *cur, dHRaw_g_);
        if (li == 0) {
            // Trunk-input gradient is discarded.
            trunk_[li].backward(dHRaw_g_, dXdiscard_g_);
        } else {
            trunk_[li].backward(dHRaw_g_, *next);
            std::swap(cur, next);
        }
    }
}
#endif

void PolicyValueNet::to(Device d) {
    if (d == device_) return;
    device_require_cuda("PolicyValueNet");
#ifdef BGA_HAS_CUDA
    if (d == Device::GPU) {
        for (auto& l : trunk_) l.to(Device::GPU);
        v_fc1_.to(Device::GPU);
        v_fc2_.to(Device::GPU);
        p_fc_.to(Device::GPU);

        // Allocate forward caches.
        trunk_raw_g_.clear();
        trunk_act_g_.clear();
        trunk_raw_g_.resize(cfg_.hidden.size());
        trunk_act_g_.resize(cfg_.hidden.size());
        for (size_t i = 0; i < cfg_.hidden.size(); ++i) {
            trunk_raw_g_[i].resize(cfg_.hidden[i], 1);
            trunk_act_g_[i].resize(cfg_.hidden[i], 1);
        }
        v_h_raw_g_.resize(cfg_.value_hidden, 1);
        v_h_act_g_.resize(cfg_.value_hidden, 1);
        v_pre_tanh_g_.resize(1, 1);
        v_post_tanh_g_.resize(1, 1);
        dPostTanh_g_.resize(1, 1);
        dPreTanh_g_.resize(1, 1);
        dVAct_g_.resize(cfg_.value_hidden, 1);
        dVRaw_g_.resize(cfg_.value_hidden, 1);
        const int trunk_out = trunk_dim();
        dTrunkFromV_g_.resize(trunk_out, 1);
        dTrunkFromP_g_.resize(trunk_out, 1);
        dHAct_g_.resize(trunk_out, 1);
        // dHRaw and dPrev get re-sized inside backward as we walk layers.
        dHRaw_g_.resize(cfg_.hidden.empty() ? 1 : cfg_.hidden.back(), 1);
        dPrev_g_.resize(cfg_.in_dim, 1);
        dXdiscard_g_.resize(cfg_.in_dim, 1);
        device_ = Device::GPU;
    } else {
        for (auto& l : trunk_) l.to(Device::CPU);
        v_fc1_.to(Device::CPU);
        v_fc2_.to(Device::CPU);
        p_fc_.to(Device::CPU);
        device_ = Device::CPU;
    }
#endif
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

void PolicyValueNet::adam_step(float lr, float b1, float b2, float eps, int step) {
    for (auto& l : trunk_) l.adam_step(lr, b1, b2, eps, step);
    v_fc1_.adam_step(lr, b1, b2, eps, step);
    v_fc2_.adam_step(lr, b1, b2, eps, step);
    p_fc_.adam_step(lr, b1, b2, eps, step);
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
