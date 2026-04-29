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

} // namespace brogameagent::nn::gpu
