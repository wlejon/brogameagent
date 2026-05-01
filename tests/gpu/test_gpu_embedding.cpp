// CPU↔GPU parity tests for embedding_lookup_forward / _backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/tensor.h>

#include <cstdint>
#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_embedding(int V, int D, const std::vector<int32_t>& idx,
                   uint64_t seed) {
    const int B = static_cast<int>(idx.size());
    SplitMix64 rng(seed);

    Tensor table(V, D), dOut(B, D);
    fill_random(table, rng);
    fill_random(dOut, rng);

    // CPU forward: gather rows.
    Tensor out_cpu(B, D);
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) out_cpu(b, j) = table(row, j);
    }
    // CPU backward: scatter-accumulate. Pre-fill with non-zero to verify +=.
    Tensor dTable_init(V, D);
    fill_random(dTable_init, rng, 0.25f);
    Tensor dTable_cpu = dTable_init;
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) dTable_cpu(row, j) += dOut(b, j);
    }

    // GPU.
    GpuTensor gtable, gdOut, gout, gdTable;
    upload(table, gtable);
    upload(dOut, gdOut);
    upload(dTable_init, gdTable);

    auto d_idx_buf = upload_indices(idx);
    int32_t* d_idx = d_idx_buf.device_ptr();

    brogameagent::nn::gpu::embedding_lookup_forward_gpu(gtable, d_idx, B, gout);
    brogameagent::nn::gpu::embedding_lookup_backward_gpu(gdOut, d_idx, B, gdTable);

    Tensor out_gpu = download_to_host(gout);
    Tensor dTable_gpu = download_to_host(gdTable);

    compare_tensors(out_cpu, out_gpu, "emb.out");
    compare_tensors(dTable_cpu, dTable_gpu, "emb.dTable");
}

} // namespace

BGA_PARITY_TEST(emb_V8_D4_distinct) {
    run_embedding(8, 4, {0, 1, 2, 3, 4, 5, 6, 7}, 0x500ull);
}
BGA_PARITY_TEST(emb_V16_D32_repeats) {
    // Repeated indices to exercise scatter accumulation.
    run_embedding(16, 32, {3, 3, 3, 7, 7, 1, 0, 15, 15, 15}, 0x501ull);
}
BGA_PARITY_TEST(emb_V64_D8_random) {
    std::vector<int32_t> idx;
    SplitMix64 rng(0x502ull);
    for (int i = 0; i < 32; ++i) {
        idx.push_back(static_cast<int32_t>(rng.next_u64() % 64));
    }
    run_embedding(64, 8, idx, 0x502ull);
}
BGA_PARITY_TEST(emb_V4_D16_all_same) {
    run_embedding(4, 16, {2, 2, 2, 2, 2, 2}, 0x503ull);
}

int main() { return run_all("gpu embedding parity"); }
