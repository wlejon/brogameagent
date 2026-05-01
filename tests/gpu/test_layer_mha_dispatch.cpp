// End-to-end GPU dispatch parity test for MultiHeadAttention.

#include "parity_helpers.h"

#include <brogameagent/nn/multi_head_attention.h>


using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::MultiHeadAttention;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void seed_layer(MultiHeadAttention& m, int K, int D, int H, uint64_t seed) {
    uint64_t s = seed;
    m.init(K, D, H, s);
    SplitMix64 rng(seed ^ 0xF00Dull);
    for (int i = 0; i < D * D; ++i) {
        m.Wq().data[i] = rng.next_unit() * 0.5f;
        m.Wk().data[i] = rng.next_unit() * 0.5f;
        m.Wv().data[i] = rng.next_unit() * 0.5f;
        m.Wo().data[i] = rng.next_unit() * 0.5f;
    }
}

void run_dispatch(int K, int D, int H, uint64_t seed) {
    MultiHeadAttention cpu, gpu_a;
    seed_layer(cpu,   K, D, H, seed);
    seed_layer(gpu_a, K, D, H, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X(K, D), dO(K, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    Tensor O_cpu(K, D), dX_cpu(K, D);
    cpu.zero_grad();
    cpu.forward(X, nullptr, O_cpu);
    cpu.backward(dO, dX_cpu);

    gpu_a.to(Device::GPU);
    BGA_CHECK(gpu_a.device() == Device::GPU);
    GpuTensor gX, gO, gdO, gdX;
    upload(X, gX); upload(dO, gdO);
    gO.resize(K, D); gdX.resize(K, D);
    gpu_a.zero_grad();
    gpu_a.forward(gX, nullptr, gO);
    gpu_a.backward(gdO, gdX);

    compare_tensors(O_cpu, download_to_host(gO),  "mha.dispatch.O");
    compare_tensors(dX_cpu, download_to_host(gdX), "mha.dispatch.dX");

    cpu.sgd_step(0.01f, 0.9f);
    gpu_a.sgd_step(0.01f, 0.9f);
    gpu_a.to(Device::CPU);
    compare_tensors(cpu.Wq(), gpu_a.Wq(), "mha.dispatch.Wq_after_sgd");
    compare_tensors(cpu.Wk(), gpu_a.Wk(), "mha.dispatch.Wk_after_sgd");
    compare_tensors(cpu.Wv(), gpu_a.Wv(), "mha.dispatch.Wv_after_sgd");
    compare_tensors(cpu.Wo(), gpu_a.Wo(), "mha.dispatch.Wo_after_sgd");

    // Save/load round-trip after GPU migration.
    gpu_a.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_a.save_to(blob);
    MultiHeadAttention restored;
    uint64_t s = seed;
    restored.init(K, D, H, s);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.Wq(), restored.Wq(), "mha.dispatch.save_load_Wq");
    compare_tensors(cpu.Wo(), restored.Wo(), "mha.dispatch.save_load_Wo");
}

} // namespace

BGA_PARITY_TEST(mha_dispatch_K8_D16_H2)  { run_dispatch(8,  16, 2, 0x500ull); }
BGA_PARITY_TEST(mha_dispatch_K12_D32_H4) { run_dispatch(12, 32, 4, 0x501ull); }

int main() { return run_all("gpu mha dispatch parity"); }
