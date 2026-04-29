// CPU↔GPU parity tests for layernorm_forward / layernorm_backward.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/layernorm.h>

using namespace bga_parity;
using brogameagent::nn::LayerNorm;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void run_layernorm(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x(n, 1), dY(n, 1);
    fill_random(x, rng);
    fill_random(dY, rng);

    // Random gamma/beta (non-trivial).
    Tensor gamma(n, 1), beta(n, 1);
    for (int i = 0; i < n; ++i) {
        gamma.data[i] = 0.5f + 0.5f * rng.next_f01();   // [0.5, 1.0)
        beta.data[i]  = rng.next_unit() * 0.25f;
    }

    // Pre-fill dGamma/dBeta to validate accumulation.
    Tensor dGamma_init(n, 1), dBeta_init(n, 1);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init, rng, 0.25f);

    // CPU.
    LayerNorm ln;
    ln.init(n);
    ln.gamma() = gamma;
    ln.beta()  = beta;
    ln.dGamma() = dGamma_init;
    ln.dBeta()  = dBeta_init;
    Tensor y_cpu(n, 1), dX_cpu(n, 1);
    ln.forward(x, y_cpu);
    ln.backward(dY, dX_cpu);
    Tensor dGamma_cpu = ln.dGamma();
    Tensor dBeta_cpu  = ln.dBeta();

    // GPU.
    GpuTensor gx, ggamma, gbeta, gy, gxhat, gdY, gdX, gdGamma, gdBeta;
    upload(x, gx);
    upload(gamma, ggamma);
    upload(beta, gbeta);
    upload(dY, gdY);
    gy.resize(n, 1); gxhat.resize(n, 1); gdX.resize(n, 1);
    upload(dGamma_init, gdGamma);
    upload(dBeta_init,  gdBeta);
    float mean_out = 0.0f, rstd_out = 0.0f;
    brogameagent::nn::gpu::layernorm_forward_gpu(
        gx, ggamma, gbeta, gy, gxhat, mean_out, rstd_out, 1e-5f);
    Tensor y_gpu = download_to_host(gy);
    brogameagent::nn::gpu::layernorm_backward_gpu(
        gdY, gxhat, ggamma, rstd_out, gdX, gdGamma, gdBeta);

    compare_tensors(y_cpu, y_gpu, "layernorm.y");
    compare_tensors(dX_cpu, download_to_host(gdX), "layernorm.dX");
    compare_tensors(dGamma_cpu, download_to_host(gdGamma), "layernorm.dGamma");
    compare_tensors(dBeta_cpu, download_to_host(gdBeta), "layernorm.dBeta");
}

} // namespace

BGA_PARITY_TEST(layernorm_n16)  { run_layernorm(16,  0x200ull); }
BGA_PARITY_TEST(layernorm_n64)  { run_layernorm(64,  0x201ull); }
BGA_PARITY_TEST(layernorm_n256) { run_layernorm(256, 0x202ull); }

int main() { return run_all("gpu layernorm parity"); }
