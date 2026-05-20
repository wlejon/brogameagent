// End-to-end GPU dispatch parity test for MultiHeadAttention.

#include "parity_helpers.h"

#include <brogameagent/nn/multi_head_attention.h>


using namespace bga_parity;
using brotensor::Device;
using brogameagent::nn::MultiHeadAttention;
using brotensor::Tensor;

namespace {

void seed_layer(MultiHeadAttention& m, int K, int D, int H, uint64_t seed) {
    uint64_t s = seed;
    m.init(K, D, H, s);
    SplitMix64 rng(seed ^ 0xF00Dull);
    for (int i = 0; i < D * D; ++i) {
        m.Wq()[i] = rng.next_unit() * 0.5f;
        m.Wk()[i] = rng.next_unit() * 0.5f;
        m.Wv()[i] = rng.next_unit() * 0.5f;
        m.Wo()[i] = rng.next_unit() * 0.5f;
    }
}

void run_dispatch(int K, int D, int H, uint64_t seed) {
    MultiHeadAttention cpu, gpu_a;
    seed_layer(cpu,   K, D, H, seed);
    seed_layer(gpu_a, K, D, H, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X = Tensor::mat(K, D), dO = Tensor::mat(K, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    Tensor O_cpu = Tensor::mat(K, D), dX_cpu = Tensor::mat(K, D);
    cpu.zero_grad();
    cpu.forward(X, nullptr, O_cpu);
    cpu.backward(dO, dX_cpu);

    gpu_a.to(Device::CUDA);
    BGA_CHECK(gpu_a.device() == Device::CUDA);
    Tensor gX = X.to(Device::CUDA), gdO = dO.to(Device::CUDA);
    Tensor gO = Tensor::zeros_on(Device::CUDA, K, D);
    Tensor gdX = Tensor::zeros_on(Device::CUDA, K, D);
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
    gpu_a.to(Device::CUDA);
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
