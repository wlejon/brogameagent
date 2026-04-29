// CPU↔GPU parity tests for concat_rows_gpu / split_rows_gpu (round-trip).

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/tensor.h>

#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_concat(const std::vector<int>& sizes, uint64_t seed) {
    SplitMix64 rng(seed);

    // Make CPU parts and concat reference.
    std::vector<Tensor> parts_cpu;
    int total = 0;
    for (int s : sizes) {
        Tensor t(s, 1);
        fill_random(t, rng);
        total += s;
        parts_cpu.push_back(std::move(t));
    }
    Tensor cat_cpu(total, 1);
    int off = 0;
    for (const auto& p : parts_cpu) {
        for (int i = 0; i < p.size(); ++i) cat_cpu.data[off + i] = p.data[i];
        off += p.size();
    }

    // Upload parts.
    std::vector<GpuTensor> g_parts(sizes.size());
    std::vector<const GpuTensor*> g_parts_ptr;
    for (size_t i = 0; i < sizes.size(); ++i) {
        upload(parts_cpu[i], g_parts[i]);
        g_parts_ptr.push_back(&g_parts[i]);
    }
    GpuTensor gcat;
    brogameagent::nn::gpu::concat_rows_gpu(g_parts_ptr, gcat);
    cuda_sync();

    Tensor cat_gpu = download_to_host(gcat);
    compare_tensors(cat_cpu, cat_gpu, "concat");

    // Round-trip: split into fresh tensors of the right shapes and check
    // each segment recovers exactly.
    std::vector<GpuTensor> g_split(sizes.size());
    std::vector<GpuTensor*> g_split_ptr;
    for (size_t i = 0; i < sizes.size(); ++i) {
        g_split[i].resize(sizes[i], 1);
        g_split_ptr.push_back(&g_split[i]);
    }
    brogameagent::nn::gpu::split_rows_gpu(gcat, g_split_ptr);
    cuda_sync();

    for (size_t i = 0; i < sizes.size(); ++i) {
        Tensor seg = download_to_host(g_split[i]);
        compare_tensors(parts_cpu[i], seg, "split.seg");
    }
}

} // namespace

BGA_PARITY_TEST(concat_three_equal)   { run_concat({16, 16, 16}, 0x600ull); }
BGA_PARITY_TEST(concat_varying_sizes) { run_concat({4, 17, 33, 8}, 0x601ull); }
BGA_PARITY_TEST(concat_two_segments)  { run_concat({64, 32}, 0x602ull); }
BGA_PARITY_TEST(concat_single_seg)    { run_concat({128}, 0x603ull); }
BGA_PARITY_TEST(concat_many_small)    { run_concat({1, 2, 3, 5, 7, 11, 13}, 0x604ull); }

int main() { return run_all("gpu concat/split parity"); }
