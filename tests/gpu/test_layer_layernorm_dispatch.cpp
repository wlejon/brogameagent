// End-to-end GPU dispatch parity test for LayerNorm.
//
// Builds two LayerNorm instances seeded identically; migrates one to
// Device::GPU; runs forward/backward and sgd_step on both; verifies that
// outputs, parameter grads and post-step parameters match within tolerance.
// Also exercises save/load round-trip after a host↔device migration.

#include "parity_helpers.h"

#include <brogameagent/nn/layernorm.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::LayerNorm;
using brogameagent::nn::Tensor;

namespace {

void seed_layer(LayerNorm& ln, int n, uint64_t seed) {
    ln.init(n);
    SplitMix64 rng(seed);
    for (int i = 0; i < n; ++i) {
        ln.gamma()[i] = 0.5f + 0.5f * rng.next_f01();
        ln.beta()[i]  = rng.next_unit() * 0.25f;
    }
}

void run_dispatch(int n, uint64_t seed) {
    LayerNorm cpu, gpu_ln;
    seed_layer(cpu,    n, seed);
    seed_layer(gpu_ln, n, seed);

    SplitMix64 rng(seed ^ 0xABCDull);
    Tensor x(n, 1), dY(n, 1);
    fill_random(x, rng);
    fill_random(dY, rng);

    // CPU path.
    Tensor y_cpu(n, 1), dX_cpu(n, 1);
    cpu.zero_grad();
    cpu.forward(x, y_cpu);
    cpu.backward(dY, dX_cpu);
    Tensor dGamma_cpu = cpu.dGamma();
    Tensor dBeta_cpu  = cpu.dBeta();

    // GPU path: migrate, then run via the GpuTensor overloads.
    gpu_ln.to(Device::GPU);
    BGA_CHECK(gpu_ln.device() == Device::GPU);
    brogameagent::nn::gpu::GpuTensor gx, gy, gdY, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(n, 1); gdX.resize(n, 1);
    gpu_ln.zero_grad();
    gpu_ln.forward(gx, gy);
    gpu_ln.backward(gdY, gdX);

    Tensor y_gpu  = download_to_host(gy);
    Tensor dX_gpu = download_to_host(gdX);
    compare_tensors(y_cpu, y_gpu, "ln.dispatch.y");
    compare_tensors(dX_cpu, dX_gpu, "ln.dispatch.dX");

    // sgd_step on both, then bring GPU params back to host and compare.
    cpu.sgd_step(0.01f, 0.9f);
    gpu_ln.sgd_step(0.01f, 0.9f);
    gpu_ln.to(Device::CPU);
    BGA_CHECK(gpu_ln.device() == Device::CPU);
    compare_tensors(cpu.gamma(), gpu_ln.gamma(), "ln.dispatch.gamma_after_sgd");
    compare_tensors(cpu.beta(),  gpu_ln.beta(),  "ln.dispatch.beta_after_sgd");

    // Save/load round-trip after migrating to GPU and back.
    gpu_ln.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_ln.save_to(blob);
    LayerNorm restored;
    restored.init(n);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.gamma(), restored.gamma(), "ln.dispatch.save_load_gamma");
    compare_tensors(cpu.beta(),  restored.beta(),  "ln.dispatch.save_load_beta");
}

} // namespace

BGA_PARITY_TEST(layernorm_dispatch_n16)  { run_dispatch(16,  0x300ull); }
BGA_PARITY_TEST(layernorm_dispatch_n128) { run_dispatch(128, 0x301ull); }

int main() { return run_all("gpu layernorm dispatch parity"); }
