// CPU↔GPU parity tests for layernorm_forward / layernorm_backward.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brogameagent/nn/layernorm.h>

using namespace bga_parity;
using brogameagent::nn::LayerNorm;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_layernorm(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n), dY = Tensor::vec(n);
    fill_random(x, rng);
    fill_random(dY, rng);

    // Random gamma/beta (non-trivial).
    Tensor gamma = Tensor::vec(n), beta = Tensor::vec(n);
    for (int i = 0; i < n; ++i) {
        gamma[i] = 0.5f + 0.5f * rng.next_f01();   // [0.5, 1.0)
        beta[i]  = rng.next_unit() * 0.25f;
    }

    // Pre-fill dGamma/dBeta to validate accumulation.
    Tensor dGamma_init = Tensor::vec(n), dBeta_init = Tensor::vec(n);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init, rng, 0.25f);

    // CPU.
    LayerNorm ln;
    ln.init(n);
    ln.gamma() = gamma;
    ln.beta()  = beta;
    ln.dGamma() = dGamma_init;
    ln.dBeta()  = dBeta_init;
    Tensor y_cpu = Tensor::vec(n), dX_cpu = Tensor::vec(n);
    ln.forward(x, y_cpu);
    ln.backward(dY, dX_cpu);
    Tensor dGamma_cpu = ln.dGamma();
    Tensor dBeta_cpu  = ln.dBeta();

    // GPU — run the same ops directly on CUDA-resident tensors.
    Tensor gx = x.to(Device::CUDA);
    Tensor ggamma = gamma.to(Device::CUDA);
    Tensor gbeta = beta.to(Device::CUDA);
    Tensor gdY = dY.to(Device::CUDA);
    Tensor gy = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gxhat = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdX = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdGamma = dGamma_init.to(Device::CUDA);
    Tensor gdBeta = dBeta_init.to(Device::CUDA);
    float mean_out = 0.0f, rstd_out = 0.0f;
    brotensor::layernorm_forward(
        gx, ggamma, gbeta, gy, gxhat, mean_out, rstd_out, 1e-5f);
    Tensor y_gpu = download_to_host(gy);
    brotensor::layernorm_backward(
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
