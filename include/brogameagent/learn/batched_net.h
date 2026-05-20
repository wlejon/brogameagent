#pragma once

// BatchedNet
// ──────────
// Thin abstract interface for nets that can serve a batched inference forward
// pass. BatchedInferenceServer and ServerBackend depend only on this
// interface, so any net that implements it (PolicyValueNet, SingleHeroNetTX,
// future variants) can be plugged in.
//
// Device-neutral: brotensor::Tensor carries its own Device tag and ops
// dispatch at runtime, so a BatchedNet runs wherever its parameters live.
// `device()` reports that so the server can stage inputs on the same backend.

#include <brotensor/tensor.h>

namespace brogameagent::learn {

class BatchedNet {
public:
    virtual ~BatchedNet() = default;

    // Width of one observation row. The server uses this to validate inputs.
    virtual int input_dim() const = 0;

    // Width of one logits row. The server uses this to size per-call results.
    virtual int logits_dim() const = 0;

    // Device the net's parameters currently live on. The server stages its
    // (B, *) input/output tensors on this device before calling forward.
    virtual brotensor::Device device() const = 0;

    // Run a batched forward.
    //   X_BD       : (B, input_dim())                observations, row-major
    //   logits_BL  : (B, logits_dim())  — resized to fit if mis-shaped
    //   values_B1  : (B, 1)             — resized to fit if mis-shaped
    //
    // Implementations must produce results equivalent (within floating-point
    // tolerance) to running B independent single-sample forward passes on the
    // same inputs.
    virtual void forward_batched(const brotensor::Tensor& X_BD,
                                 brotensor::Tensor& logits_BL,
                                 brotensor::Tensor& values_B1) = 0;
};

} // namespace brogameagent::learn
