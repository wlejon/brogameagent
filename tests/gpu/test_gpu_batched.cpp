// Parity tests for the batched (inference-only) GPU kernels.
//
// For each batched op we compare its B-row output against B independent
// calls to the corresponding single-sample kernel.

#include "parity_helpers.h"

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/ops.h>

#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

// ─── linear_forward_batched ────────────────────────────────────────────────

static void run_linear_batched(int B, int in_dim, int out_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W(out_dim, in_dim), b(out_dim, 1);
    fill_random(W, rng);
    fill_random(b, rng);

    // Build (B, in_dim) input matrix.
    Tensor X_BD(B, in_dim);
    fill_random(X_BD, rng);

    // Reference: B sequential single-sample GPU calls into a (B, out_dim) buffer.
    GpuTensor gW, gb, gX_BD, gY_BD;
    upload(W, gW); upload(b, gb); upload(X_BD, gX_BD);
    Tensor Y_ref(B, out_dim);
    for (int i = 0; i < B; ++i) {
        Tensor xi(in_dim, 1);
        for (int j = 0; j < in_dim; ++j)
            xi.data[j] = X_BD.data[static_cast<size_t>(i) * in_dim + j];
        GpuTensor gxi, gyi;
        upload(xi, gxi);
        gyi.resize(out_dim, 1);
        brogameagent::nn::gpu::linear_forward_gpu(gW, gb, gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < out_dim; ++j)
            Y_ref.data[static_cast<size_t>(i) * out_dim + j] = yi.data[j];
    }

    // Batched call.
    brogameagent::nn::gpu::linear_forward_batched_gpu(gW, gb, gX_BD, gY_BD);
    Tensor Y_batched = download_to_host(gY_BD);
    BGA_CHECK(Y_batched.rows == B);
    BGA_CHECK(Y_batched.cols == out_dim);
    compare_tensors(Y_ref, Y_batched, "linear_forward_batched");
}

BGA_PARITY_TEST(linear_batched_B1)   { run_linear_batched(1,  16, 32, 0xC1ull); }
BGA_PARITY_TEST(linear_batched_B4)   { run_linear_batched(4,  64, 32, 0xC2ull); }
BGA_PARITY_TEST(linear_batched_B64)  { run_linear_batched(64, 128, 96, 0xC3ull); }
BGA_PARITY_TEST(linear_batched_skinny) { run_linear_batched(8, 1, 7, 0xC4ull); }

// ─── relu_forward_batched ──────────────────────────────────────────────────

static void run_relu_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD(B, D);
    fill_random(X_BD, rng);

    // Reference: per-row single-sample kernel.
    Tensor Y_ref(B, D);
    for (int i = 0; i < B; ++i) {
        Tensor xi(D, 1);
        for (int j = 0; j < D; ++j)
            xi.data[j] = X_BD.data[static_cast<size_t>(i) * D + j];
        GpuTensor gxi, gyi;
        upload(xi, gxi);
        gyi.resize(D, 1);
        brogameagent::nn::gpu::relu_forward_gpu(gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref.data[static_cast<size_t>(i) * D + j] = yi.data[j];
    }

    GpuTensor gX, gY;
    upload(X_BD, gX);
    brogameagent::nn::gpu::relu_forward_batched_gpu(gX, gY);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "relu_forward_batched");
}

BGA_PARITY_TEST(relu_batched_B1)  { run_relu_batched(1, 64, 0xD1ull); }
BGA_PARITY_TEST(relu_batched_B4)  { run_relu_batched(4, 64, 0xD2ull); }
BGA_PARITY_TEST(relu_batched_B64) { run_relu_batched(64, 32, 0xD3ull); }

// ─── tanh_forward_batched ──────────────────────────────────────────────────

static void run_tanh_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD(B, D);
    fill_random(X_BD, rng);

    Tensor Y_ref(B, D);
    for (int i = 0; i < B; ++i) {
        Tensor xi(D, 1);
        for (int j = 0; j < D; ++j)
            xi.data[j] = X_BD.data[static_cast<size_t>(i) * D + j];
        GpuTensor gxi, gyi;
        upload(xi, gxi);
        gyi.resize(D, 1);
        brogameagent::nn::gpu::tanh_forward_gpu(gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref.data[static_cast<size_t>(i) * D + j] = yi.data[j];
    }

    GpuTensor gX, gY;
    upload(X_BD, gX);
    brogameagent::nn::gpu::tanh_forward_batched_gpu(gX, gY);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "tanh_forward_batched");
}

BGA_PARITY_TEST(tanh_batched_B1)  { run_tanh_batched(1, 8, 0xE1ull); }
BGA_PARITY_TEST(tanh_batched_B4)  { run_tanh_batched(4, 16, 0xE2ull); }
BGA_PARITY_TEST(tanh_batched_B64) { run_tanh_batched(64, 1, 0xE3ull); }

// ─── add_inplace_batched ───────────────────────────────────────────────────

static void run_add_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Y_init(B, D), X(B, D);
    fill_random(Y_init, rng);
    fill_random(X, rng);

    // Reference: per-row single-sample kernel.
    Tensor Y_ref = Y_init;
    for (int i = 0; i < B; ++i) {
        Tensor yi(D, 1), xi(D, 1);
        for (int j = 0; j < D; ++j) {
            yi.data[j] = Y_ref.data[static_cast<size_t>(i) * D + j];
            xi.data[j] = X.data[static_cast<size_t>(i) * D + j];
        }
        GpuTensor gyi, gxi;
        upload(yi, gyi); upload(xi, gxi);
        brogameagent::nn::gpu::add_inplace_gpu(gyi, gxi);
        Tensor out = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref.data[static_cast<size_t>(i) * D + j] = out.data[j];
    }

    GpuTensor gY, gX;
    upload(Y_init, gY); upload(X, gX);
    brogameagent::nn::gpu::add_inplace_batched_gpu(gY, gX);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "add_inplace_batched");
}

BGA_PARITY_TEST(add_batched_B1)  { run_add_batched(1, 8, 0xF1ull); }
BGA_PARITY_TEST(add_batched_B4)  { run_add_batched(4, 16, 0xF2ull); }
BGA_PARITY_TEST(add_batched_B64) { run_add_batched(64, 32, 0xF3ull); }

int main() { return run_all("gpu batched ops parity"); }
