#pragma once

#include "tensor.h"

#include <vector>

namespace brogameagent::nn {

// ─── Factored-policy helpers ──────────────────────────────────────────────
//
// PolicyValueNet emits one concatenated logit buffer; with multi-head
// configs, that buffer is the concatenation of per-head softmax inputs.
// MCTS / search code on the consumer side wants a flat prior over the
// cartesian product of per-head action choices: P(flat) = ∏_h P(head_h).
//
// These helpers do the conversion. The flat→tuple mapping is row-major:
//
//   flat = a0 * (h1 * h2 * ... * hN) + a1 * (h2 * ... * hN) + ... + aN
//
// i.e. the last head varies fastest. Use head_strides() if you need to
// decode flat indices yourself (e.g. to look up the chosen sub-action
// per head after MCTS picks a flat action).
//
// `head_offsets` follows PolicyValueNet::head_offsets() — exclusive
// prefix sums of head_sizes plus a trailing sentinel equal to the total
// width. `head_masks`, when non-null, is a buffer of length head_offsets
// .back(), with 0/1 legality flags per logit entry; the per-head softmax
// honors it.
//
// flat_prior must point at a buffer of length prod(head_sizes).

// Strides for decoding flat indices: stride[h] = product of head_sizes[h+1..].
std::vector<int> head_strides(const std::vector<int>& head_sizes);

// Total flat action space size = product of head_sizes.
int flat_action_count(const std::vector<int>& head_sizes);

// Decode a flat index into per-head action indices. `out` must hold
// head_sizes.size() entries; written in head order.
void decode_flat_action(int flat,
                        const std::vector<int>& head_sizes,
                        const std::vector<int>& strides,
                        int* out);

// Encode per-head action indices into a flat index.
int encode_flat_action(const int* per_head,
                       const std::vector<int>& strides,
                       int n_heads);

// Convert a concatenated per-head logit buffer to a flat prior over the
// cartesian product. Each head's softmax is applied independently
// (numerically stable) and the per-head probabilities are multiplied
// across heads. With a single head this is equivalent to one softmax.
void factored_to_flat(const float* logits,
                      const std::vector<int>& head_sizes,
                      const std::vector<int>& head_offsets,
                      float* flat_prior,
                      const float* head_masks = nullptr);

// Tensor-shaped wrapper for the common case (full PolicyValueNet output).
void factored_to_flat(const Tensor& logits,
                      const std::vector<int>& head_sizes,
                      const std::vector<int>& head_offsets,
                      Tensor& flat_prior,
                      const float* head_masks = nullptr);

} // namespace brogameagent::nn
