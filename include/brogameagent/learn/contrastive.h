#pragma once

#include <brotensor/tensor.h>

#include <vector>

namespace brogameagent::learn {

// ─── InfoNCE (cosine) ─────────────────────────────────────────────────────
//
// For each anchor i, score against every positive j with
//   s_ij = dot(a_i, p_j) / (||a_i|| * ||p_j|| * temperature)
// softmax across j, target is one-hot at j=i. Loss = mean cross-entropy
// over the batch. Produces gradients w.r.t. anchors and positives.

float infonce_loss(const std::vector<brotensor::Tensor>& anchors,
                   const std::vector<brotensor::Tensor>& positives,
                   std::vector<brotensor::Tensor>& dAnchors,
                   std::vector<brotensor::Tensor>& dPositives,
                   float temperature = 0.1f);

} // namespace brogameagent::learn
