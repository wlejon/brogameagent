#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

// Concatenate flat tensors end-to-end. We use cudaMemcpyAsync per part on
// the default stream — this is a single-pass copy with no kernel launches,
// trivially correct and matches what an "optimised" kernel would do at the
// PCIe-bandwidth boundary anyway.
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
        BGA_CUDA_CHECK(cudaMemcpyAsync(out.data + off, p->data,
                                       sizeof(float) * n,
                                       cudaMemcpyDeviceToDevice));
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
        BGA_CUDA_CHECK(cudaMemcpyAsync(p->data, in.data + off,
                                       sizeof(float) * n,
                                       cudaMemcpyDeviceToDevice));
        off += n;
    }
    (void)in;
}

// Batched column-block concat. Each part is (B, d_i) for the same B; out
// becomes (B, sum_i d_i) with parts laid out as column blocks per row.
// out[b, off_i + j] = parts[i][b, j]. Implemented via cudaMemcpy2DAsync per
// part — pure bandwidth, no kernel launches.
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

    const size_t dst_pitch = sizeof(float) * total_cols;
    int col_off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        BGA_CUDA_CHECK(cudaMemcpy2DAsync(
            out.data + col_off, dst_pitch,
            p->data,             sizeof(float) * d,
            sizeof(float) * d,   B,
            cudaMemcpyDeviceToDevice));
        col_off += d;
    }
}

// Single-stream device-to-device chunk copy. Copies `n` floats from
// src.data + src_off into dst.data + dst_off. No bounds checking.
void copy_d2d_gpu(const GpuTensor& src, int src_off,
                  GpuTensor& dst,       int dst_off,
                  int n) {
    if (n <= 0) return;
    BGA_CUDA_CHECK(cudaMemcpyAsync(dst.data + dst_off,
                                   src.data + src_off,
                                   sizeof(float) * n,
                                   cudaMemcpyDeviceToDevice));
}

} // namespace brogameagent::nn::gpu
