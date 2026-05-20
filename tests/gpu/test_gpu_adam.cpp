// CPU↔GPU parity tests for adam_step.

#include "parity_helpers.h"

#include <brogameagent/nn/circuits.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bga_parity;
using brotensor::Tensor;
using brotensor::Device;
using brogameagent::nn::adam_step_cpu;

namespace {

void run_adam(int n, uint64_t seed, float lr, float beta1, float beta2,
              float eps, int n_steps) {
    SplitMix64 rng(seed);
    Tensor param = Tensor::vec(n), grad = Tensor::vec(n),
           m = Tensor::vec(n), v = Tensor::vec(n);
    fill_random(param, rng);
    fill_random(grad, rng);
    m.zero();
    v.zero();

    Tensor param_cpu = param;
    Tensor m_cpu = m;
    Tensor v_cpu = v;

    Tensor gparam = param.to(Device::CUDA);
    Tensor ggrad = grad.to(Device::CUDA);
    Tensor gm = m.to(Device::CUDA);
    Tensor gv = v.to(Device::CUDA);

    // Run K steps with the same gradient each step (sufficient to surface
    // any bias-correction or moment-update mismatch).
    for (int s = 1; s <= n_steps; ++s) {
        adam_step_cpu(param_cpu, grad, m_cpu, v_cpu, lr, beta1, beta2, eps, s);
        brotensor::adam_step(gparam, ggrad, gm, gv, lr, beta1, beta2, eps, s);
    }

    compare_tensors(param_cpu, download_to_host(gparam), "adam.param");
    compare_tensors(m_cpu,     download_to_host(gm),     "adam.m");
    compare_tensors(v_cpu,     download_to_host(gv),     "adam.v");
}

} // namespace

BGA_PARITY_TEST(adam_n1_step1)    { run_adam(1,    0x500ull, 1e-3f, 0.9f,  0.999f, 1e-8f, 1); }
BGA_PARITY_TEST(adam_n64_step1)   { run_adam(64,   0x501ull, 1e-3f, 0.9f,  0.999f, 1e-8f, 1); }
BGA_PARITY_TEST(adam_n64_step10)  { run_adam(64,   0x502ull, 1e-3f, 0.9f,  0.999f, 1e-8f, 10); }
BGA_PARITY_TEST(adam_n1024_step5) { run_adam(1024, 0x503ull, 1e-2f, 0.9f,  0.999f, 1e-8f, 5); }
BGA_PARITY_TEST(adam_high_betas)  { run_adam(64,   0x504ull, 1e-3f, 0.95f, 0.9999f, 1e-7f, 5); }
BGA_PARITY_TEST(adam_zero_betas)  { run_adam(64,   0x505ull, 1e-2f, 0.0f,  0.0f,    1e-8f, 3); }

int main() { return run_all("gpu adam parity"); }
