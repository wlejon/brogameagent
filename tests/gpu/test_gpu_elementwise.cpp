// CPU↔GPU parity tests for elementwise activations and adds.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/ops.h>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void test_relu(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x(n, 1), dY(n, 1);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu(n, 1), dX_cpu(n, 1);
    brogameagent::nn::relu_forward(x, y_cpu);
    brogameagent::nn::relu_backward(x, dY, dX_cpu);

    GpuTensor gx, gdY, gy, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(n, 1); gdX.resize(n, 1);
    brogameagent::nn::gpu::relu_forward_gpu(gx, gy);
    brogameagent::nn::gpu::relu_backward_gpu(gx, gdY, gdX);

    compare_tensors(y_cpu, download_to_host(gy), "relu_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "relu_backward");
}

void test_tanh(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x(n, 1), dY(n, 1);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu(n, 1), dX_cpu(n, 1);
    brogameagent::nn::tanh_forward(x, y_cpu);
    brogameagent::nn::tanh_backward(y_cpu, dY, dX_cpu);

    GpuTensor gx, gy, gdY, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(n, 1); gdX.resize(n, 1);
    brogameagent::nn::gpu::tanh_forward_gpu(gx, gy);
    Tensor y_gpu = download_to_host(gy);
    brogameagent::nn::gpu::tanh_backward_gpu(gy, gdY, gdX);

    compare_tensors(y_cpu, y_gpu, "tanh_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "tanh_backward");
}

void test_sigmoid(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x(n, 1), dY(n, 1);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu(n, 1), dX_cpu(n, 1);
    brogameagent::nn::sigmoid_forward(x, y_cpu);
    brogameagent::nn::sigmoid_backward(y_cpu, dY, dX_cpu);

    GpuTensor gx, gy, gdY, gdX;
    upload(x, gx); upload(dY, gdY);
    gy.resize(n, 1); gdX.resize(n, 1);
    brogameagent::nn::gpu::sigmoid_forward_gpu(gx, gy);
    Tensor y_gpu = download_to_host(gy);
    brogameagent::nn::gpu::sigmoid_backward_gpu(gy, gdY, gdX);

    compare_tensors(y_cpu, y_gpu, "sigmoid_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "sigmoid_backward");
}

void test_add_inplace(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y(n, 1), x(n, 1);
    fill_random(y, rng);
    fill_random(x, rng);

    Tensor y_cpu = y;
    brogameagent::nn::add_inplace(y_cpu, x);

    GpuTensor gy, gx;
    upload(y, gy); upload(x, gx);
    brogameagent::nn::gpu::add_inplace_gpu(gy, gx);

    compare_tensors(y_cpu, download_to_host(gy), "add_inplace");
}

void test_add_scalar_inplace(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y(n, 1);
    fill_random(y, rng);
    const float s = 0.375f;

    Tensor y_cpu = y;
    brogameagent::nn::add_scalar_inplace(y_cpu, s);

    GpuTensor gy;
    upload(y, gy);
    brogameagent::nn::gpu::add_scalar_inplace_gpu(gy, s);

    compare_tensors(y_cpu, download_to_host(gy), "add_scalar_inplace");
}

} // namespace

BGA_PARITY_TEST(relu_n1)    { test_relu(1, 0x10ull); }
BGA_PARITY_TEST(relu_n7)    { test_relu(7, 0x11ull); }
BGA_PARITY_TEST(relu_n256)  { test_relu(256, 0x12ull); }
BGA_PARITY_TEST(relu_n1024) { test_relu(1024, 0x13ull); }

BGA_PARITY_TEST(tanh_n1)    { test_tanh(1, 0x20ull); }
BGA_PARITY_TEST(tanh_n7)    { test_tanh(7, 0x21ull); }
BGA_PARITY_TEST(tanh_n256)  { test_tanh(256, 0x22ull); }
BGA_PARITY_TEST(tanh_n1024) { test_tanh(1024, 0x23ull); }

BGA_PARITY_TEST(sigmoid_n1)    { test_sigmoid(1, 0x30ull); }
BGA_PARITY_TEST(sigmoid_n7)    { test_sigmoid(7, 0x31ull); }
BGA_PARITY_TEST(sigmoid_n256)  { test_sigmoid(256, 0x32ull); }
BGA_PARITY_TEST(sigmoid_n1024) { test_sigmoid(1024, 0x33ull); }

BGA_PARITY_TEST(add_inplace_n1)    { test_add_inplace(1, 0x40ull); }
BGA_PARITY_TEST(add_inplace_n7)    { test_add_inplace(7, 0x41ull); }
BGA_PARITY_TEST(add_inplace_n256)  { test_add_inplace(256, 0x42ull); }
BGA_PARITY_TEST(add_inplace_n1024) { test_add_inplace(1024, 0x43ull); }

BGA_PARITY_TEST(add_scalar_n1)    { test_add_scalar_inplace(1, 0x50ull); }
BGA_PARITY_TEST(add_scalar_n7)    { test_add_scalar_inplace(7, 0x51ull); }
BGA_PARITY_TEST(add_scalar_n256)  { test_add_scalar_inplace(256, 0x52ull); }
BGA_PARITY_TEST(add_scalar_n1024) { test_add_scalar_inplace(1024, 0x53ull); }

int main() { return run_all("gpu elementwise parity"); }
