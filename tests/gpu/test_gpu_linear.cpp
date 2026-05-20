// CPU↔GPU parity tests for linear_forward / linear_backward.

#include "parity_helpers.h"

#include <brotensor/ops.h>

using namespace bga_parity;
using brotensor::Tensor;
using brotensor::Device;

static void run_linear_forward(int in_dim, int out_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim), b = Tensor::vec(out_dim),
           x = Tensor::vec(in_dim);
    fill_random(W, rng);
    fill_random(b, rng);
    fill_random(x, rng);

    Tensor y_cpu = Tensor::vec(out_dim);
    brotensor::linear_forward(W, b, x, y_cpu);

    Tensor gW = W.to(Device::CUDA), gb = b.to(Device::CUDA),
           gx = x.to(Device::CUDA);
    Tensor gy = Tensor::zeros_on(Device::CUDA, out_dim, 1);
    brotensor::linear_forward(gW, gb, gx, gy);
    Tensor y_gpu = download_to_host(gy);

    compare_tensors(y_cpu, y_gpu, "linear_forward");
}

BGA_PARITY_TEST(linear_forward_64x32) { run_linear_forward(64, 32, 0xA1ull); }
BGA_PARITY_TEST(linear_forward_1x1)   { run_linear_forward(1, 1, 0xA2ull); }
BGA_PARITY_TEST(linear_forward_128x128) { run_linear_forward(128, 128, 0xA3ull); }

static void run_linear_backward(int in_dim, int out_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim), x = Tensor::vec(in_dim),
           dY = Tensor::vec(out_dim);
    fill_random(W, rng);
    fill_random(x, rng);
    fill_random(dY, rng);

    // Pre-fill dW, dB with non-zero starting values to verify accumulation.
    Tensor dW_init = Tensor::mat(out_dim, in_dim), dB_init = Tensor::vec(out_dim);
    fill_random(dW_init, rng, 0.25f);
    fill_random(dB_init, rng, 0.25f);

    // CPU path.
    Tensor dX_cpu = Tensor::vec(in_dim);
    Tensor dW_cpu = dW_init;
    Tensor dB_cpu = dB_init;
    brotensor::linear_backward(W, x, dY, dX_cpu, dW_cpu, dB_cpu);

    // GPU path with the same starting accumulators.
    Tensor gW = W.to(Device::CUDA), gx = x.to(Device::CUDA),
           gdY = dY.to(Device::CUDA);
    Tensor gdX = Tensor::zeros_on(Device::CUDA, in_dim, 1);
    Tensor gdW = dW_init.to(Device::CUDA);
    Tensor gdB = dB_init.to(Device::CUDA);
    brotensor::linear_backward(gW, gx, gdY, gdX, gdW, gdB);

    Tensor dX_gpu = download_to_host(gdX);
    Tensor dW_gpu = download_to_host(gdW);
    Tensor dB_gpu = download_to_host(gdB);

    compare_tensors(dX_cpu, dX_gpu, "linear_backward.dX");
    compare_tensors(dW_cpu, dW_gpu, "linear_backward.dW");
    compare_tensors(dB_cpu, dB_gpu, "linear_backward.dB");
}

BGA_PARITY_TEST(linear_backward_64x32)   { run_linear_backward(64, 32, 0xB1ull); }
BGA_PARITY_TEST(linear_backward_1x1)     { run_linear_backward(1, 1, 0xB2ull); }
BGA_PARITY_TEST(linear_backward_128x128) { run_linear_backward(128, 128, 0xB3ull); }

int main() { return run_all("gpu linear parity"); }
