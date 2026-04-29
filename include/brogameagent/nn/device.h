#pragma once

// ─── Device tag for GPU-aware layers ───────────────────────────────────────
//
// Lightweight enum + helpers so each parameter-bearing layer can answer
// "where do my weights currently live?" without touching ICircuit.
//
// Layers retrofitted with GPU dispatch hold a `Device device_` field plus
// optional GpuTensor mirrors of every host Tensor they own. `to(Device)`
// migrates parameters/grads/velocities/caches host↔device. The CPU code
// path is byte-identical to before this retrofit; the GPU path is gated
// by BGA_HAS_CUDA.
//
// Calling to(Device::GPU) on a CPU-only build (BGA_HAS_CUDA undefined) is
// a runtime error: see device_require_cuda() below.

#include <stdexcept>

namespace brogameagent::nn {

enum class Device { CPU, GPU };

// Throws std::runtime_error with a readable message when GPU is requested
// but CUDA support wasn't compiled in. Layers call this from `to(GPU)`.
inline void device_require_cuda(const char* layer_name) {
#ifndef BGA_HAS_CUDA
    throw std::runtime_error(
        std::string("brogameagent: cannot move ") + layer_name +
        " to Device::GPU — built without BGA_HAS_CUDA (CUDA support disabled)");
#else
    (void)layer_name;
#endif
}

} // namespace brogameagent::nn
