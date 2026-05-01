#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

#include <cstring>

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;

namespace {

// Unified-memory: just memcpy on the host pointer. Simplest, fastest.
inline void copy_floats(float* dst, const float* src, size_t n) {
    if (n == 0) return;
    std::memcpy(dst, src, n * sizeof(float));
}

} // namespace

void concat_rows_gpu(const std::vector<const GpuTensor*>& parts,
                     GpuTensor& out) {
    int total = 0;
    for (const auto* p : parts) total += p ? p->size() : 0;
    if (out.rows != total || out.cols != 1) out.resize(total, 1);
    if (total == 0) return;
    int off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        copy_floats(out.data + off, p->data, static_cast<size_t>(n));
        off += n;
    }
}

void split_rows_gpu(const GpuTensor& in,
                    const std::vector<GpuTensor*>& parts) {
    int off = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        copy_floats(p->data, in.data + off, static_cast<size_t>(n));
        off += n;
    }
}

void concat_batched_rows_gpu(const std::vector<const GpuTensor*>& parts,
                             GpuTensor& out) {
    if (parts.empty()) { out.resize(0, 0); return; }
    int B = 0;
    int total_cols = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        if (B == 0) B = p->rows;
        total_cols += p->cols;
    }
    if (out.rows != B || out.cols != total_cols) out.resize(B, total_cols);
    if (B == 0 || total_cols == 0) return;
    int col_off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        for (int b = 0; b < B; ++b) {
            copy_floats(out.data + static_cast<size_t>(b) * total_cols + col_off,
                        p->data + static_cast<size_t>(b) * d,
                        static_cast<size_t>(d));
        }
        col_off += d;
    }
}

void copy_d2d_gpu(const GpuTensor& src, int src_off,
                  GpuTensor& dst,       int dst_off,
                  int n) {
    if (n <= 0) return;
    copy_floats(dst.data + dst_off, src.data + src_off, static_cast<size_t>(n));
}

} // namespace brogameagent::nn::gpu
