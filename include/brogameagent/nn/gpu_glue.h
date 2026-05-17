#pragma once

// Glue between brogameagent's host Tensor type and brotensor's GpuTensor.
// brotensor does not depend on brogameagent — its upload/download primitives
// take raw float pointers. These thin inlines adapt at brogameagent's
// boundary so dispatch sites can write `nn::upload_to(W_, W_g_)` instead of
// pulling out the (ptr, rows, cols) by hand each time.
//
// Only available when a GPU backend is enabled. Including this header
// without BROTENSOR_HAS_GPU is a no-op.

#ifdef BROTENSOR_HAS_GPU

#include <brogameagent/nn/tensor.h>
#include <brotensor/tensor.h>

namespace brogameagent::nn {

// Host→device. Resizes dst to match (src.rows, src.cols).
inline void upload_to(const Tensor& src, brotensor::GpuTensor& dst) {
    brotensor::upload(src.data.data(), src.rows, src.cols, dst);
}

// Device→host. Resizes dst to match (src.rows, src.cols).
inline void download_to(const brotensor::GpuTensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        dst.resize(src.rows, src.cols);
    }
    brotensor::download(src, dst.data.data());
}

} // namespace brogameagent::nn

#endif // BROTENSOR_HAS_GPU
