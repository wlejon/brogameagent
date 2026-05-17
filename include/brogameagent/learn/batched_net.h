#pragma once

// BatchedNet
// ──────────
// Thin abstract interface for GPU-resident nets that can serve a batched
// inference forward pass. The BatchedInferenceServer and ServerBackend depend
// only on this interface, so any net that implements it (PolicyValueNet,
// SingleHeroNetTX, future variants) can be plugged in.
//
// GPU-only: the interface is gated behind BROTENSOR_HAS_GPU because GpuTensor is
// only defined in CUDA builds. Direct CPU dispatch goes through the
// per-net forward(...) APIs as before.

#ifdef BROTENSOR_HAS_GPU

#include <brotensor/tensor.h>

namespace brogameagent::learn {

class BatchedNet {
public:
    virtual ~BatchedNet() = default;

    // Width of one observation row. The server uses this to validate inputs.
    virtual int input_dim() const = 0;

    // Width of one logits row. The server uses this to size per-call results.
    virtual int logits_dim() const = 0;

    // Run a batched forward.
    //   X_BD       : (B, input_dim())                observations, row-major
    //   logits_BL  : (B, logits_dim())  — resized to fit if mis-shaped
    //   values_B1  : (B, 1)             — resized to fit if mis-shaped
    //
    // Implementations must produce results equivalent (within floating-point
    // tolerance) to running B independent single-sample forward passes on the
    // same inputs.
    virtual void forward_batched(const brotensor::GpuTensor& X_BD,
                                 brotensor::GpuTensor& logits_BL,
                                 brotensor::GpuTensor& values_B1) = 0;
};

} // namespace brogameagent::learn

#endif // BROTENSOR_HAS_GPU
