// CPU↔GPU parity tests for sgd_step.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/tensor.h>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

// Mirror of static sgd_mat in src/nn/attention.cpp.
void cpu_sgd_step(Tensor& W, Tensor& vW, const Tensor& dW, float lr, float momentum) {
    const int n = W.size();
    for (int i = 0; i < n; ++i) {
        vW.data[i] = momentum * vW.data[i] + dW.data[i];
        W.data[i] -= lr * vW.data[i];
    }
}

void run_sgd(int n, uint64_t seed, float lr, float momentum) {
    SplitMix64 rng(seed);
    Tensor param(n, 1), grad(n, 1), vel(n, 1);
    fill_random(param, rng);
    fill_random(grad, rng);
    fill_random(vel, rng, 0.25f);

    Tensor param_cpu = param;
    Tensor vel_cpu   = vel;
    cpu_sgd_step(param_cpu, vel_cpu, grad, lr, momentum);

    GpuTensor gparam, ggrad, gvel;
    upload(param, gparam);
    upload(grad, ggrad);
    upload(vel, gvel);
    brogameagent::nn::gpu::sgd_step_gpu(gparam, ggrad, gvel, lr, momentum);

    compare_tensors(param_cpu, download_to_host(gparam), "sgd.param");
    compare_tensors(vel_cpu,   download_to_host(gvel),   "sgd.velocity");
}

} // namespace

BGA_PARITY_TEST(sgd_n1)        { run_sgd(1,    0x400ull, 1e-2f, 0.9f); }
BGA_PARITY_TEST(sgd_n64)       { run_sgd(64,   0x401ull, 1e-2f, 0.9f); }
BGA_PARITY_TEST(sgd_n1024)     { run_sgd(1024, 0x402ull, 1e-2f, 0.9f); }
BGA_PARITY_TEST(sgd_zero_mom)  { run_sgd(64,   0x403ull, 5e-2f, 0.0f); }
BGA_PARITY_TEST(sgd_high_mom)  { run_sgd(64,   0x404ull, 1e-3f, 0.99f); }

int main() { return run_all("gpu optim parity"); }
