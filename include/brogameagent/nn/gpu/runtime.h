#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brogameagent::nn::gpu {

// ─── CUDA Runtime helpers ──────────────────────────────────────────────────
//
// This header is safe to include from non-CUDA translation units. It does
// NOT include <cuda_runtime.h>. The check macro forwards to a thin wrapper
// implemented in runtime.cu — that's where the real cudaGetErrorString call
// happens. We want public headers to be plain C++ so that libraries
// consuming the foundation tensor type don't get pulled into nvcc.

// Idempotent. Selects device 0 unless the env var BGA_CUDA_DEVICE overrides
// it with a decimal device index. Safe to call multiple times.
void cuda_init();

// Wraps cudaDeviceSynchronize(). Throws std::runtime_error on failure.
void cuda_sync();

// Implementation hook for BGA_CUDA_CHECK — translates a cudaError_t (passed
// in as int because we don't include cuda_runtime.h here) into a human
// readable error message and throws std::runtime_error if non-zero.
//
// `expr_text` is the stringified expression for diagnostic context;
// `file`/`line` describe the call site.
void cuda_check_throw(int err, const char* expr_text, const char* file, int line);

} // namespace brogameagent::nn::gpu

// Wrap any CUDA call. Safe to use from .cpp files (only forward-declared
// helpers are referenced; the int conversion is implicit from cudaError_t).
#define BGA_CUDA_CHECK(expr)                                                    \
    do {                                                                        \
        int _bga_err = static_cast<int>(expr);                                  \
        if (_bga_err != 0) {                                                    \
            ::brogameagent::nn::gpu::cuda_check_throw(                          \
                _bga_err, #expr, __FILE__, __LINE__);                           \
        }                                                                       \
    } while (0)
