#include "brogameagent/nn/factored.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace brogameagent::nn {

std::vector<int> head_strides(const std::vector<int>& head_sizes) {
    // Row-major: the last head varies fastest, so its stride is 1, the
    // second-to-last is head_sizes.back(), and so on.
    std::vector<int> s(head_sizes.size(), 1);
    if (head_sizes.empty()) return s;
    for (int i = static_cast<int>(head_sizes.size()) - 2; i >= 0; --i) {
        s[i] = s[i + 1] * head_sizes[i + 1];
    }
    return s;
}

int flat_action_count(const std::vector<int>& head_sizes) {
    int n = 1;
    for (int h : head_sizes) n *= h;
    return n;
}

void decode_flat_action(int flat,
                        const std::vector<int>& head_sizes,
                        const std::vector<int>& strides,
                        int* out) {
    const int n = static_cast<int>(head_sizes.size());
    for (int i = 0; i < n; ++i) {
        out[i] = (flat / strides[i]) % head_sizes[i];
    }
}

int encode_flat_action(const int* per_head,
                       const std::vector<int>& strides,
                       int n_heads) {
    int flat = 0;
    for (int i = 0; i < n_heads; ++i) flat += per_head[i] * strides[i];
    return flat;
}

void factored_to_flat(const float* logits,
                      const std::vector<int>& head_sizes,
                      const std::vector<int>& head_offsets,
                      float* flat_prior,
                      const float* head_masks) {
    assert(!head_sizes.empty());
    assert(head_offsets.size() == head_sizes.size() + 1);

    const int n_heads = static_cast<int>(head_sizes.size());
    const int total   = flat_action_count(head_sizes);

    // Per-head softmax into a scratch buffer of length sum(head_sizes).
    // Reused across heads via segment offsets.
    std::vector<float> probs(head_offsets.back(), 0.0f);
    std::vector<float> dummy_d(head_offsets.back(), 0.0f);
    std::vector<float> zero_target(head_offsets.back(), 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        const int off = head_offsets[h];
        const int len = head_offsets[h + 1] - off;
        const float* mask = head_masks ? head_masks + off : nullptr;
        // softmax_xent_segment is overkill (it also computes loss/grad),
        // but it's the existing stable softmax with mask we already trust.
        // The dummy target/grad buffers are written and discarded.
        softmax_xent_segment(logits + off, zero_target.data() + off,
                             probs.data() + off, dummy_d.data() + off,
                             len, mask);
    }

    // Cartesian product. Walk flat indices in row-major order; iterate
    // heads inside to multiply per-head probs. With small head counts
    // (typically ≤ 4) this is fine even for a few hundred flat actions.
    const auto strides = head_strides(head_sizes);
    for (int flat = 0; flat < total; ++flat) {
        float p = 1.0f;
        for (int h = 0; h < n_heads; ++h) {
            const int idx = (flat / strides[h]) % head_sizes[h];
            p *= probs[head_offsets[h] + idx];
        }
        flat_prior[flat] = p;
    }
}

void factored_to_flat(const Tensor& logits,
                      const std::vector<int>& head_sizes,
                      const std::vector<int>& head_offsets,
                      Tensor& flat_prior,
                      const float* head_masks) {
    assert(logits.size() == head_offsets.back());
    assert(flat_prior.size() == flat_action_count(head_sizes));
    factored_to_flat(logits.ptr(), head_sizes, head_offsets,
                     flat_prior.ptr(), head_masks);
}

} // namespace brogameagent::nn
