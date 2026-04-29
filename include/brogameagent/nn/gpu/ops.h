#pragma once

#include "tensor.h"

#include <cstdint>

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

// SGD with momentum, in-place:
//   velocity = momentum * velocity + grad
//   param   -= lr * velocity
// All three tensors must have identical shape. velocity is updated in place;
// caller is responsible for grad zeroing between batches.
void sgd_step_gpu(GpuTensor& param, GpuTensor& grad, GpuTensor& velocity,
                  float lr, float momentum);

} // namespace brogameagent::nn::gpu
