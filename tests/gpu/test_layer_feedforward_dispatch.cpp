// End-to-end GPU dispatch parity test for FeedForward.

#include "parity_helpers.h"

#include <brogameagent/nn/feedforward.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::FeedForward;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void seed_layer(FeedForward& f, int D, int Df, uint64_t seed) {
    uint64_t s = seed;
    f.init(D, Df, s);
    SplitMix64 rng(seed ^ 0xF00Dull);
    for (int i = 0; i < f.W1().size(); ++i) f.W1().data[i] = rng.next_unit() * 0.5f;
    for (int i = 0; i < f.b1().size(); ++i) f.b1().data[i] = rng.next_unit() * 0.1f;
    for (int i = 0; i < f.W2().size(); ++i) f.W2().data[i] = rng.next_unit() * 0.5f;
    for (int i = 0; i < f.b2().size(); ++i) f.b2().data[i] = rng.next_unit() * 0.1f;
}

void run_dispatch(int K, int D, int Df, uint64_t seed) {
    FeedForward cpu, gpu_f;
    seed_layer(cpu,   D, Df, seed);
    seed_layer(gpu_f, D, Df, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X(K, D), dY(K, D);
    fill_random(X, rng);
    fill_random(dY, rng);

    Tensor Y_cpu(K, D), dX_cpu(K, D);
    cpu.zero_grad();
    cpu.forward(X, Y_cpu);
    cpu.backward(dY, dX_cpu);

    gpu_f.to(Device::GPU);
    BGA_CHECK(gpu_f.device() == Device::GPU);
    GpuTensor gX, gY, gdY, gdX;
    upload(X, gX); upload(dY, gdY);
    gY.resize(K, D); gdX.resize(K, D);
    gpu_f.zero_grad();
    gpu_f.forward(gX, gY);
    gpu_f.backward(gdY, gdX);

    compare_tensors(Y_cpu,  download_to_host(gY),  "ff.dispatch.Y");
    compare_tensors(dX_cpu, download_to_host(gdX), "ff.dispatch.dX");

    cpu.sgd_step(0.01f, 0.9f);
    gpu_f.sgd_step(0.01f, 0.9f);
    gpu_f.to(Device::CPU);
    compare_tensors(cpu.W1(), gpu_f.W1(), "ff.dispatch.W1_after_sgd");
    compare_tensors(cpu.b1(), gpu_f.b1(), "ff.dispatch.b1_after_sgd");
    compare_tensors(cpu.W2(), gpu_f.W2(), "ff.dispatch.W2_after_sgd");
    compare_tensors(cpu.b2(), gpu_f.b2(), "ff.dispatch.b2_after_sgd");

    // Save/load round-trip after GPU migration.
    gpu_f.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_f.save_to(blob);
    FeedForward restored;
    uint64_t s = seed;
    restored.init(D, Df, s);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.W1(), restored.W1(), "ff.dispatch.save_load_W1");
    compare_tensors(cpu.W2(), restored.W2(), "ff.dispatch.save_load_W2");
}

} // namespace

BGA_PARITY_TEST(ff_dispatch_K6_D8_Df16)   { run_dispatch(6,  8,  16, 0x600ull); }
BGA_PARITY_TEST(ff_dispatch_K10_D16_Df32) { run_dispatch(10, 16, 32, 0x601ull); }

int main() { return run_all("gpu feedforward dispatch parity"); }
