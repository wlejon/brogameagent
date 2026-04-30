#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brogameagent::nn::gpu {

// ─── GPU primitive ops (declarations only) ─────────────────────────────────
//
// One-to-one mirror of brogameagent::nn::ops over GpuTensor. Shape contracts
// match the CPU versions verbatim — see include/brogameagent/nn/ops.h for the
// authoritative semantics. These are the contracts subagents 2 and 3 will
// implement; the doc comments below are the spec.
//
// All tensors are float32, row-major, on the same CUDA device (device 0
// unless cuda_init was steered via BGA_CUDA_DEVICE). Output tensors are
// resized by the implementation if their shape doesn't match the expected
// output shape — except for accumulation outputs (dW, dB) which the caller
// must size and zero appropriately.
//
// Streams: every op is implicitly on the default (null) stream for now.
// Synchronisation is the caller's responsibility; use cuda_sync() before
// reading results back to host.

// ─── Subagent 2: dense layers + elementwise activations ────────────────────

// y = W * x + b.
//   W: (out_dim, in_dim)
//   b: (out_dim, 1)         (vector; cols == 1)
//   x: (in_dim, 1)
//   y: (out_dim, 1)         (resized if mis-shaped)
void linear_forward_gpu(const GpuTensor& W, const GpuTensor& b,
                        const GpuTensor& x, GpuTensor& y);

// Backward of linear_forward.
//   W:   (out_dim, in_dim)   (forward weights, read-only)
//   x:   (in_dim, 1)         (forward input, read-only)
//   dY:  (out_dim, 1)        (upstream gradient)
//   dX:  (in_dim, 1)         (output, *overwritten*)
//   dW:  (out_dim, in_dim)   (output, *accumulated into* — caller zeros)
//   dB:  (out_dim, 1)        (output, *accumulated into* — caller zeros)
void linear_backward_gpu(const GpuTensor& W, const GpuTensor& x,
                         const GpuTensor& dY,
                         GpuTensor& dX, GpuTensor& dW, GpuTensor& dB);

// y = max(x, 0). x and y may alias (same buffer) for in-place ReLU.
// Shapes match exactly; y resized if mis-shaped.
void relu_forward_gpu(const GpuTensor& x, GpuTensor& y);

// dX = dY * (x > 0). dX resized to match x if mis-shaped. dX may alias dY.
void relu_backward_gpu(const GpuTensor& x, const GpuTensor& dY, GpuTensor& dX);

// y = tanh(x). y resized to match x if mis-shaped.
void tanh_forward_gpu(const GpuTensor& x, GpuTensor& y);

// dX = dY * (1 - y*y). `y` is the cached forward output (NOT raw x).
void tanh_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX);

// y = 1 / (1 + exp(-x)).
void sigmoid_forward_gpu(const GpuTensor& x, GpuTensor& y);

// dX = dY * y * (1 - y). `y` is the cached forward output.
void sigmoid_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX);

// y[i] += x[i]. y and x must have identical shape.
void add_inplace_gpu(GpuTensor& y, const GpuTensor& x);

// y[i] += s for all i.
void add_scalar_inplace_gpu(GpuTensor& y, float s);

// y[i] *= s for all i.
void scale_inplace_gpu(GpuTensor& y, float s);

// ─── Subagent 3: reductions, norm, attention, optimiser ────────────────────

// Numerically stable softmax over a flat vector of length N = logits.size().
//
//   logits: (N, 1) or (1, N) — treated as flat length-N buffer.
//   probs:  same shape as logits; resized if mis-shaped.
//   d_mask: optional device pointer to N floats (1 valid, 0 invalid). May be
//           null. Invalid positions contribute 0 to the normaliser AND
//           receive 0 in `probs`. Caller guarantees at least one valid entry
//           when masking — the kernel does not check.
void softmax_forward_gpu(const GpuTensor& logits, GpuTensor& probs,
                         const float* d_mask);

// Full Jacobian softmax backward:
//   dLogits[i] = sum_j dProbs[j] * probs[j] * (delta_ij - probs[i]).
// All tensors length-N; dLogits resized to match if mis-shaped.
void softmax_backward_gpu(const GpuTensor& probs, const GpuTensor& dProbs,
                          GpuTensor& dLogits);

// LayerNorm forward (single-vector, matches CPU LayerNorm).
//   x:     (N, 1)            input vector
//   gamma: (N, 1)            learnable scale
//   beta:  (N, 1)            learnable shift
//   y:     (N, 1)            output, resized if mis-shaped
//   xhat:  (N, 1)            cached normalised x = (x - mean) * rstd, resized
//   mean_out: scalar host-side cache, written by op
//   rstd_out: scalar host-side cache (1 / sqrt(var + eps)), written by op
//   eps:   variance epsilon, typically 1e-5f
//
// The backward consumes (xhat, gamma, mean, rstd) — the signature here is
// intentionally rich so backward needs no recomputation. Subagent 3 may
// revise these caches (e.g. promote mean/rstd to a tiny GpuTensor) if it's
// cleaner; document any change here.
void layernorm_forward_gpu(const GpuTensor& x,
                           const GpuTensor& gamma, const GpuTensor& beta,
                           GpuTensor& y, GpuTensor& xhat,
                           float& mean_out, float& rstd_out,
                           float eps);

// LayerNorm backward.
//   dY:     (N, 1) upstream
//   xhat:   (N, 1) cached from forward
//   gamma:  (N, 1) forward scale
//   rstd:   scalar from forward
//   dX:     (N, 1) output, overwritten
//   dGamma: (N, 1) accumulated into — caller zeros
//   dBeta:  (N, 1) accumulated into — caller zeros
void layernorm_backward_gpu(const GpuTensor& dY, const GpuTensor& xhat,
                            const GpuTensor& gamma, float rstd,
                            GpuTensor& dX,
                            GpuTensor& dGamma, GpuTensor& dBeta);

// Single-head scaled dot-product self-attention (mirrors CPU
// ScaledDotProductAttention). All projections are square (D, D), no biases.
//
//   X:  (N, D) input
//   Wq, Wk, Wv, Wo: each (D, D)
//   d_mask: optional device pointer, length N (1 valid, 0 invalid). May be
//           null (all valid). Invalid keys are excluded from the softmax
//           denominator (additive -inf on the score row pre-softmax) and
//           invalid query rows produce zero output. Same semantics as
//           softmax mask + the CPU attention impl.
//   O:  (N, D) output, resized if mis-shaped
//
// Caches needed for backward (subagent 3 chooses representation):
//   Q, K, V: each (N, D)
//   Attn:   (N, N)  post-softmax weights
//   Y_pre_Wo: (N, D)  Attn @ V (before output projection)
// Pass these as out-parameters so backward can consume them.
void attention_forward_gpu(const GpuTensor& X,
                           const GpuTensor& Wq, const GpuTensor& Wk,
                           const GpuTensor& Wv, const GpuTensor& Wo,
                           const float* d_mask,
                           GpuTensor& Q, GpuTensor& K, GpuTensor& V,
                           GpuTensor& Attn, GpuTensor& Y_pre_Wo,
                           GpuTensor& O);

// Attention backward.
//   dO: (N, D) upstream
//   X, Q, K, V, Attn, Y_pre_Wo: forward caches
//   Wq, Wk, Wv, Wo: forward weights
//   d_mask: same mask used in forward (or null)
//   dX: (N, D) output, overwritten
//   dWq, dWk, dWv, dWo: (D, D) accumulated into — caller zeros
void attention_backward_gpu(const GpuTensor& dO,
                            const GpuTensor& X,
                            const GpuTensor& Q, const GpuTensor& K,
                            const GpuTensor& V, const GpuTensor& Attn,
                            const GpuTensor& Y_pre_Wo,
                            const GpuTensor& Wq, const GpuTensor& Wk,
                            const GpuTensor& Wv, const GpuTensor& Wo,
                            const float* d_mask,
                            GpuTensor& dX,
                            GpuTensor& dWq, GpuTensor& dWk,
                            GpuTensor& dWv, GpuTensor& dWo);

// ─── Subagent A: multi-head self-attention ─────────────────────────────────

// Multi-head scaled dot-product self-attention. Mirrors the CPU
// MultiHeadAttention class. Square (D, D) projections internally split into
// h heads of head_dim = D / h. h must divide D.
//
//   X:   (K, D) input
//   Wq, Wk, Wv, Wo: each (D, D). Each Wq/Wk/Wv is treated as h stacked
//                   per-head row slices of shape (head_dim, D).
//   d_mask: optional length-K device pointer (1 valid, 0 invalid). May be
//           null. Same semantics as single-head: invalid keys excluded from
//           softmax denom; invalid query rows produce zero output.
//   num_heads: number of attention heads.
//   O:   (K, D) output, resized if mis-shaped.
//
// Caches for backward (out-parameters; resized if mis-shaped):
//   Qh:    (h * K, head_dim) — head h occupies rows [h*K, (h+1)*K)
//   Kh:    (h * K, head_dim)
//   Vh:    (h * K, head_dim)
//   Attnh: (h * K, K)        — per-head softmax weights, same row partition
//   Yconcat: (K, D)          — pre-Wo concat of per-head outputs
void mha_forward_gpu(const GpuTensor& X,
                     const GpuTensor& Wq, const GpuTensor& Wk,
                     const GpuTensor& Wv, const GpuTensor& Wo,
                     const float* d_mask,
                     int num_heads,
                     GpuTensor& Qh, GpuTensor& Kh, GpuTensor& Vh,
                     GpuTensor& Attnh, GpuTensor& Yconcat,
                     GpuTensor& O);

// Multi-head attention backward.
//   dO: (K, D) upstream
//   X, Qh, Kh, Vh, Attnh, Yconcat: forward caches
//   Wq, Wk, Wv, Wo: forward weights (each (D, D))
//   d_mask: same mask used in forward (or null)
//   num_heads: must match forward
//   dX: (K, D) output, *overwritten*
//   dWq, dWk, dWv, dWo: (D, D) accumulated into — caller zeros
void mha_backward_gpu(const GpuTensor& dO,
                      const GpuTensor& X,
                      const GpuTensor& Qh, const GpuTensor& Kh,
                      const GpuTensor& Vh, const GpuTensor& Attnh,
                      const GpuTensor& Yconcat,
                      const GpuTensor& Wq, const GpuTensor& Wk,
                      const GpuTensor& Wv, const GpuTensor& Wo,
                      const float* d_mask,
                      int num_heads,
                      GpuTensor& dX,
                      GpuTensor& dWq, GpuTensor& dWk,
                      GpuTensor& dWv, GpuTensor& dWo);

// ─── Subagent C: pooling, losses, embedding, concat ────────────────────────

// Masked mean-pool forward over rows of a (K, D) matrix.
//   X:    (K, D)  input matrix
//   mask: device pointer to K floats (1.0 valid, 0.0 invalid). May be null —
//         null means all rows are valid.
//   y:    (D, 1)  output vector, resized if mis-shaped.
//
// Semantics: y[j] = (1 / num_valid) * sum_{k : mask[k]==1} X[k, j].
// If num_valid == 0 the output is filled with zeros (matching how
// `DeepSetsEncoder` and `SetTransformerEncoder` skip the *= inv step when
// the count is zero).
void masked_mean_pool_forward_gpu(const GpuTensor& X, const float* d_mask,
                                  GpuTensor& y);

// Masked mean-pool backward.
//   dY:   (D, 1) upstream gradient
//   mask: same mask used in forward (or null)
//   K:    number of rows in the original X (we don't carry X around)
//   dX:   (K, D) output, *overwritten* (NOT accumulated). Invalid rows are
//         set to exactly zero. Valid rows receive dY / num_valid.
//         If num_valid == 0, dX is zeroed entirely.
void masked_mean_pool_backward_gpu(const GpuTensor& dY, const float* d_mask,
                                   int K, GpuTensor& dX);

// Vector MSE forward.
//   pred, target: length-N flat tensors (any 2D shape with N elements).
// Returns: scalar loss = mean((pred - target)^2) = (1/N) * sum (p - t)^2.
// (Note: CPU `mse_scalar` is per-scalar 0.5*d^2 with grad = d. We adopt
// MEAN-of-squared-diffs for the vector form because that's the standard
// autoencoder reconstruction loss and decouples the gradient magnitude from
// N. The backward gradient is dPred = (2 / N) * (pred - target).)
float mse_vec_forward_gpu(const GpuTensor& pred, const GpuTensor& target);

// Vector MSE backward.
//   pred, target: forward inputs
//   dPred: same shape as pred, *overwritten*
//   dPred[i] = (2 / N) * (pred[i] - target[i]).
void mse_vec_backward_gpu(const GpuTensor& pred, const GpuTensor& target,
                          GpuTensor& dPred);

// Fused softmax + cross-entropy, mirroring CPU `softmax_xent_segment`.
//   logits:  length-N
//   target:  length-N (soft or one-hot; values typically in [0,1] summing to
//            1 over valid entries)
//   d_mask:  optional length-N mask (1 valid / 0 invalid). May be null.
//   probs:   length-N output (softmax over valid entries, 0 on invalid).
//            Resized if mis-shaped.
//   dLogits: length-N output (probs - target on valid; 0 on invalid).
//            Resized if mis-shaped.
// Returns: scalar loss = -sum_i (mask[i] ? target[i] * log(max(probs[i],
// 1e-12)) : 0). Caller guarantees at least one valid entry under mask.
float softmax_xent_fused_gpu(const GpuTensor& logits, const GpuTensor& target,
                             const float* d_mask,
                             GpuTensor& probs, GpuTensor& dLogits);

// Embedding lookup forward.
//   table:    (V, D) embedding matrix
//   d_idx:    device pointer to B int32 indices, each in [0, V).
//   B:        number of indices (== rows of `out`).
//   out:      (B, D), resized if mis-shaped. out[b, :] = table[d_idx[b], :].
void embedding_lookup_forward_gpu(const GpuTensor& table,
                                  const int32_t* d_idx, int B,
                                  GpuTensor& out);

// Embedding lookup backward — scatter-accumulate.
//   dOut:   (B, D) upstream
//   d_idx:  same indices used in forward (length B)
//   B:      number of indices
//   dTable: (V, D), accumulated into (caller zeros). Multiple lookups of the
//           same row sum their grads via atomicAdd.
void embedding_lookup_backward_gpu(const GpuTensor& dOut,
                                   const int32_t* d_idx, int B,
                                   GpuTensor& dTable);

// Concatenate flat tensors end-to-end.
//   parts: list of tensors, each treated as a flat buffer of size parts[i]->size().
//   out:   resized to (total, 1) where total = sum of part sizes.
// Layout: out[off_i .. off_i + size_i) = parts[i] flattened.
void concat_rows_gpu(const std::vector<const GpuTensor*>& parts,
                     GpuTensor& out);

// Inverse of concat_rows_gpu: copy disjoint segments of `in` back into the
// flat buffers of `parts`. Each parts[i] is *overwritten* (not accumulated)
// with the corresponding segment of `in`. Sizes of `parts` must be unchanged
// from the concat call. The function assumes parts[i]->size() segments laid
// end-to-end starting at offset 0 in `in`.
void split_rows_gpu(const GpuTensor& in,
                    const std::vector<GpuTensor*>& parts);

// Batched column-block concat. Each part is shape (B, d_i) for the same B;
// out becomes (B, sum_i d_i) with parts laid as column blocks per row:
//   out[b, off_i + j] = parts[i][b, j].
// Implemented via cudaMemcpy2DAsync per part — bandwidth-bound, no kernel
// launches. Use for batched per-row concat in inference.
void concat_batched_rows_gpu(const std::vector<const GpuTensor*>& parts,
                             GpuTensor& out);

// Single-stream device-to-device chunk copy. Copies `n` floats from
// src.data + src_off into dst.data + dst_off. Both tensors are treated as
// flat float buffers regardless of (rows, cols). Async on the default stream.
void copy_d2d_gpu(const GpuTensor& src, int src_off,
                  GpuTensor& dst,       int dst_off,
                  int n);

// Inference-only batched LayerNorm forward. Processes R independent rows
// of length D in a single launch (one block per row). Does not cache xhat
// or read mean/rstd back to host — no syncs. Use when backward isn't
// needed; the existing layernorm_forward_gpu remains for training.
//   X_RD:   (R, D) input
//   gamma:  (D,) scale
//   beta:   (D,) shift
//   Y_RD:   (R, D), resized if mis-shaped
void layernorm_forward_inference_batched_gpu(const GpuTensor& X_RD,
                                              const GpuTensor& gamma,
                                              const GpuTensor& beta,
                                              GpuTensor& Y_RD,
                                              float eps);

// SGD with momentum, in-place:
//   velocity = momentum * velocity + grad
//   param   -= lr * velocity
// All three tensors must have identical shape. velocity is updated in place;
// caller is responsible for grad zeroing between batches.
void sgd_step_gpu(GpuTensor& param, GpuTensor& grad, GpuTensor& velocity,
                  float lr, float momentum);

// Adam optimizer step, in-place. Mirrors adam_step_cpu in circuits.h:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   param -= lr * (m / (1 - beta1^step)) / (sqrt(v / (1 - beta2^step)) + eps)
// `step` is a 1-based step counter for bias correction. All four tensors must
// have identical shape.
void adam_step_gpu(GpuTensor& param, const GpuTensor& grad,
                   GpuTensor& m, GpuTensor& v,
                   float lr, float beta1, float beta2, float eps, int step);

// ─── Batched (inference-only) variants ─────────────────────────────────────
//
// These run B independent forward passes in a single kernel launch. They are
// forward-only; backward is not provided. Used by the BatchedInferenceServer
// to amortise per-kernel-launch latency across many concurrent requests.
//
// Layout convention: tensors carrying B rows are shaped (B, D) row-major, so
// row b at columns 0..D-1 holds the b'th sample. This is the natural shape
// for staging samples end-to-end into a flat host buffer and uploading once.

// Y[b, :] = W * X[b, :] + b   for b in [0, B).
//   W:    (out_dim, in_dim)
//   bias: (out_dim, 1)
//   X_BD: (B, in_dim)
//   Y_BD: (B, out_dim) — resized if mis-shaped.
void linear_forward_batched_gpu(const GpuTensor& W, const GpuTensor& bias,
                                const GpuTensor& X_BD, GpuTensor& Y_BD);

// Elementwise ReLU/Tanh over (B, D). Y resized to match X if mis-shaped.
// X and Y may alias.
void relu_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD);
void tanh_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD);

// Y[i] += X[i] over (B, D). Identical shape required.
void add_inplace_batched_gpu(GpuTensor& Y_BD, const GpuTensor& X_BD);

// ─── Batched (training) backward variants ──────────────────────────────────
//
// Backward partners for the batched-train path used by GenericExItTrainer.
// Match the math of the single-sample versions summed across B.

// Linear backward over a B-row minibatch.
//   W:    (out_dim, in_dim) — read-only forward weights
//   X_BD: (B, in_dim)       — forward input (cached by caller)
//   dY_BD:(B, out_dim)      — upstream gradient
//   dX_BD:(B, in_dim)       — output, *overwritten* (resized if mis-shaped)
//   dW:   (out_dim, in_dim) — *accumulated*; caller zeros before the step
//   dB:   (out_dim, 1)      — *accumulated*; caller zeros before the step
// Math: dX[b] = W^T * dY[b], dW += sum_b dY[b] * X[b]^T, dB += sum_b dY[b].
void linear_backward_batched_gpu(const GpuTensor& W, const GpuTensor& X_BD,
                                 const GpuTensor& dY_BD,
                                 GpuTensor& dX_BD,
                                 GpuTensor& dW, GpuTensor& dB);

// Elementwise ReLU/Tanh backward over (B, D). Same shapes throughout.
//   relu:  dX = dY * (X > 0); reads X_BD (the forward input).
//   tanh:  dX = dY * (1 - Y*Y); reads Y_BD (the forward output).
void relu_backward_batched_gpu(const GpuTensor& X_BD, const GpuTensor& dY_BD,
                               GpuTensor& dX_BD);
void tanh_backward_batched_gpu(const GpuTensor& Y_BD, const GpuTensor& dY_BD,
                               GpuTensor& dX_BD);

// ─── Batched per-sample loss kernels (training) ────────────────────────────
//
// Used by GenericExItTrainer to fuse loss + grad across the whole minibatch
// in a single launch, with no per-sample host roundtrips.

// Per-sample MSE matching CPU `mse_scalar` (loss = 0.5 * d², dPred = d).
//   pred:   (B, 1)
//   target: (B, 1)
//   dPred:  (B, 1) — overwritten with (pred - target)
//   loss_per_sample: (B, 1) — overwritten with 0.5 * (pred - target)^2
void mse_vec_per_sample_gpu(const GpuTensor& pred, const GpuTensor& target,
                            GpuTensor& dPred, GpuTensor& loss_per_sample);

// Batched fused softmax + cross-entropy across (sample, head) tiles.
//
// For each row b in [0, B) and head h in [0, n_heads), runs a numerically
// stable softmax-xent over the slice
//     [d_head_offsets[h], d_head_offsets[h+1])
// of the b'th row of `logits_BL`. Writes:
//     probs_BL[b, slice]   = softmax(logits_BL[b, slice]) (0 on masked)
//     dLogits_BL[b, slice] = probs - target on valid; 0 on masked
//     loss_per_sample[b]  += sum_h (-sum_{i in slice, valid} target * log p)
// loss_per_sample is *overwritten* before the per-head accumulation begins.
//
//   logits_BL, target_BL, probs_BL, dLogits_BL: (B, n_act_total)
//   d_mask_BL: optional (B, n_act_total) device pointer (1 valid / 0 invalid)
//   d_head_offsets: device int* of length n_heads + 1 (cumulative per spec).
//   loss_per_sample: (B, 1) — overwritten with the sum-over-heads loss.
//
// The caller is responsible for any mean-over-heads reduction on the loss
// (CPU formulation divides by n_heads). The kernel does not scale dLogits
// by 1/n_heads either — the caller applies that with scale_inplace_gpu.
void softmax_xent_fused_batched_gpu(const GpuTensor& logits_BL,
                                    const GpuTensor& target_BL,
                                    const float* d_mask_BL,
                                    const int* d_head_offsets,
                                    int n_heads,
                                    GpuTensor& probs_BL,
                                    GpuTensor& dLogits_BL,
                                    GpuTensor& loss_per_sample);

} // namespace brogameagent::nn::gpu
