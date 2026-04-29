// End-to-end GPU dispatch parity test for LayerNorm + Adam.
//
// Mirrors test_layer_layernorm_dispatch.cpp but exercises adam_step instead
// of sgd_step. Two LayerNorm instances are seeded identically; one is
// migrated to GPU. We run several cycles of forward/backward/adam_step on
// both, then bring the GPU params back to host and verify they match.

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

void run_dispatch_adam(int n, uint64_t seed, int n_steps) {
    LayerNorm cpu, gpu_ln;
    seed_layer(cpu,    n, seed);
    seed_layer(gpu_ln, n, seed);

    gpu_ln.to(Device::GPU);
    BGA_CHECK(gpu_ln.device() == Device::GPU);

    const float lr = 1e-2f;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

    for (int step = 1; step <= n_steps; ++step) {
        SplitMix64 rng(seed ^ (0xABCDull * static_cast<uint64_t>(step)));
        Tensor x(n, 1), dY(n, 1);
        fill_random(x, rng);
        fill_random(dY, rng);

        // CPU path.
        Tensor y_cpu(n, 1), dX_cpu(n, 1);
        cpu.zero_grad();
        cpu.forward(x, y_cpu);
        cpu.backward(dY, dX_cpu);
        cpu.adam_step(lr, b1, b2, eps, step);

        // GPU path.
        brogameagent::nn::gpu::GpuTensor gx, gy, gdY, gdX;
        upload(x, gx); upload(dY, gdY);
        gy.resize(n, 1); gdX.resize(n, 1);
        gpu_ln.zero_grad();
        gpu_ln.forward(gx, gy);
        gpu_ln.backward(gdY, gdX);
        gpu_ln.adam_step(lr, b1, b2, eps, step);
    }

    // Migrate back and compare params.
    gpu_ln.to(Device::CPU);
    compare_tensors(cpu.gamma(), gpu_ln.gamma(), "ln.adam.dispatch.gamma");
    compare_tensors(cpu.beta(),  gpu_ln.beta(),  "ln.adam.dispatch.beta");
}

} // namespace

BGA_PARITY_TEST(layernorm_adam_dispatch_n16_3steps)   { run_dispatch_adam(16,  0x600ull, 3); }
BGA_PARITY_TEST(layernorm_adam_dispatch_n128_5steps)  { run_dispatch_adam(128, 0x601ull, 5); }

int main() { return run_all("gpu layernorm adam dispatch parity"); }
