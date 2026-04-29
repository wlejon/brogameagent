// End-to-end GPU dispatch parity test for ScaledDotProductAttention.

#include "parity_helpers.h"

#include <brogameagent/nn/attention.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::ScaledDotProductAttention;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void seed_layer(ScaledDotProductAttention& a, int N, int D, uint64_t seed) {
    uint64_t s = seed;
    a.init(N, D, s);
    // Replace xavier weights with deterministic random for clean parity.
    SplitMix64 rng(seed ^ 0xF00Dull);
    for (int i = 0; i < D * D; ++i) {
        a.Wq().data[i] = rng.next_unit() * 0.5f;
        a.Wk().data[i] = rng.next_unit() * 0.5f;
        a.Wv().data[i] = rng.next_unit() * 0.5f;
        a.Wo().data[i] = rng.next_unit() * 0.5f;
    }
}

void run_dispatch(int N, int D, uint64_t seed) {
    ScaledDotProductAttention cpu, gpu_a;
    seed_layer(cpu,   N, D, seed);
    seed_layer(gpu_a, N, D, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X(N, D), dO(N, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    // CPU.
    Tensor O_cpu(N, D), dX_cpu(N, D);
    cpu.zero_grad();
    cpu.forward(X, nullptr, O_cpu);
    cpu.backward(dO, dX_cpu);
    Tensor dWq_cpu = cpu.dWq(), dWk_cpu = cpu.dWk();
    Tensor dWv_cpu = cpu.dWv(), dWo_cpu = cpu.dWo();

    // GPU.
    gpu_a.to(Device::GPU);
    BGA_CHECK(gpu_a.device() == Device::GPU);
    GpuTensor gX, gO, gdO, gdX;
    upload(X, gX); upload(dO, gdO);
    gO.resize(N, D); gdX.resize(N, D);
    gpu_a.zero_grad();
    gpu_a.forward(gX, nullptr, gO);
    gpu_a.backward(gdO, gdX);

    compare_tensors(O_cpu, download_to_host(gO),  "att.dispatch.O");
    compare_tensors(dX_cpu, download_to_host(gdX), "att.dispatch.dX");

    cpu.sgd_step(0.01f, 0.9f);
    gpu_a.sgd_step(0.01f, 0.9f);
    gpu_a.to(Device::CPU);
    compare_tensors(cpu.Wq(), gpu_a.Wq(), "att.dispatch.Wq_after_sgd");
    compare_tensors(cpu.Wk(), gpu_a.Wk(), "att.dispatch.Wk_after_sgd");
    compare_tensors(cpu.Wv(), gpu_a.Wv(), "att.dispatch.Wv_after_sgd");
    compare_tensors(cpu.Wo(), gpu_a.Wo(), "att.dispatch.Wo_after_sgd");

    // Save/load round-trip after GPU migration.
    gpu_a.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_a.save_to(blob);
    ScaledDotProductAttention restored;
    uint64_t s = seed;
    restored.init(N, D, s);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.Wq(), restored.Wq(), "att.dispatch.save_load_Wq");
    compare_tensors(cpu.Wo(), restored.Wo(), "att.dispatch.save_load_Wo");
}

} // namespace

BGA_PARITY_TEST(attention_dispatch_n8_d16)  { run_dispatch(8, 16,  0x400ull); }
BGA_PARITY_TEST(attention_dispatch_n16_d32) { run_dispatch(16, 32, 0x401ull); }

int main() { return run_all("gpu attention dispatch parity"); }
