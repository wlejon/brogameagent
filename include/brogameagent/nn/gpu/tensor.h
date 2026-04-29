#pragma once

#include "../tensor.h"

#include <cstddef>

namespace brogameagent::nn::gpu {

// ─── GpuTensor ─────────────────────────────────────────────────────────────
//
// Device-resident counterpart to brogameagent::nn::Tensor. Same row-major
// (rows, cols) shape semantics, same float32 dtype. Storage is a raw device
// pointer obtained via cudaMalloc; freed by cudaFree on destruction when
// owning. Move-only — copying device buffers must be explicit (clone()).
//
// This header is safe to include from non-CUDA TUs (no <cuda_runtime.h>);
// the actual CUDA API calls live in tensor.cu.

struct GpuTensor {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;

    GpuTensor() = default;
    GpuTensor(int r, int c);                   // cudaMalloc(r*c*sizeof(float))
    ~GpuTensor();

    // Move-only.
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    int  size() const { return rows * cols; }
    void zero();                                // cudaMemset to 0
    // Reallocates if shape differs from current; leaves contents undefined
    // (caller should zero() if needed). Existing storage is freed.
    void resize(int r, int c);

    // Device→device copy producing an owning duplicate.
    GpuTensor clone() const;

    // Non-owning view over an existing device pointer. The returned tensor's
    // destructor will NOT free `data`. Caller is responsible for lifetime.
    static GpuTensor view(float* data, int rows, int cols);

private:
    bool owns_ = false;
    void release_();
};

// Host→device. dst is resized to match src.shape if shape differs.
void upload(const Tensor& src, GpuTensor& dst);

// Device→host. dst is resized to match src.shape if shape differs.
void download(const GpuTensor& src, Tensor& dst);

} // namespace brogameagent::nn::gpu
