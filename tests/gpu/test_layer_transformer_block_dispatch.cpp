// End-to-end GPU dispatch parity test for TransformerBlock.

#include "parity_helpers.h"

#include <brogameagent/nn/transformer_block.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::TransformerBlock;
using brogameagent::nn::NormPlacement;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void seed_block(TransformerBlock& b, int K, int D, int H, int Df,
                NormPlacement np, uint64_t seed) {
    TransformerBlock::Config cfg{};
    cfg.dim = D; cfg.num_heads = H; cfg.d_ff = Df; cfg.n_slots = K;
    cfg.norm = np;
    uint64_t s = seed;
    b.init(cfg, s);
    SplitMix64 rng(seed ^ 0xF00Dull);
    auto& mha = b.mha();
    for (int i = 0; i < mha.Wq().size(); ++i) mha.Wq().data[i] = rng.next_unit() * 0.4f;
    for (int i = 0; i < mha.Wk().size(); ++i) mha.Wk().data[i] = rng.next_unit() * 0.4f;
    for (int i = 0; i < mha.Wv().size(); ++i) mha.Wv().data[i] = rng.next_unit() * 0.4f;
    for (int i = 0; i < mha.Wo().size(); ++i) mha.Wo().data[i] = rng.next_unit() * 0.4f;
    auto& ff = b.ff();
    for (int i = 0; i < ff.W1().size(); ++i) ff.W1().data[i] = rng.next_unit() * 0.4f;
    for (int i = 0; i < ff.W2().size(); ++i) ff.W2().data[i] = rng.next_unit() * 0.4f;
}

void run_dispatch(int K, int D, int H, int Df, NormPlacement np, uint64_t seed) {
    TransformerBlock cpu, gpu_b;
    seed_block(cpu,   K, D, H, Df, np, seed);
    seed_block(gpu_b, K, D, H, Df, np, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X(K, D), dY(K, D);
    fill_random(X, rng);
    fill_random(dY, rng);

    Tensor Y_cpu(K, D), dX_cpu(K, D);
    cpu.zero_grad();
    cpu.forward(X, nullptr, Y_cpu);
    cpu.backward(dY, dX_cpu);

    gpu_b.to(Device::GPU);
    BGA_CHECK(gpu_b.device() == Device::GPU);
    GpuTensor gX, gY, gdY, gdX;
    upload(X, gX); upload(dY, gdY);
    gY.resize(K, D); gdX.resize(K, D);
    gpu_b.zero_grad();
    gpu_b.forward(gX, nullptr, gY);
    gpu_b.backward(gdY, gdX);

    // Slightly looser tolerance — composite layer with several stacked sources
    // of fp accumulation order differences (per-row LN loop, per-row linear).
    compare_tensors(Y_cpu,  download_to_host(gY),  "tb.dispatch.Y",
                    1e-4f, 1e-3f);
    compare_tensors(dX_cpu, download_to_host(gdX), "tb.dispatch.dX",
                    1e-4f, 1e-3f);

    cpu.sgd_step(0.01f, 0.9f);
    gpu_b.sgd_step(0.01f, 0.9f);
    gpu_b.to(Device::CPU);
    compare_tensors(cpu.mha().Wq(), gpu_b.mha().Wq(), "tb.dispatch.Wq_after_sgd",
                    1e-4f, 1e-3f);
    compare_tensors(cpu.ff().W1(),  gpu_b.ff().W1(),  "tb.dispatch.W1_after_sgd",
                    1e-4f, 1e-3f);

    // Save/load round-trip after GPU migration.
    gpu_b.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_b.save_to(blob);
    TransformerBlock restored;
    TransformerBlock::Config rcfg{};
    rcfg.dim = D; rcfg.num_heads = H; rcfg.d_ff = Df; rcfg.n_slots = K;
    rcfg.norm = np;
    uint64_t s = seed;
    restored.init(rcfg, s);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.mha().Wq(), restored.mha().Wq(),
                    "tb.dispatch.save_load_Wq", 1e-4f, 1e-3f);
    compare_tensors(cpu.ff().W2(),  restored.ff().W2(),
                    "tb.dispatch.save_load_W2", 1e-4f, 1e-3f);
}

} // namespace

BGA_PARITY_TEST(tb_dispatch_pre_K6_D8_H2_Df16)  {
    run_dispatch(6, 8, 2, 16, NormPlacement::PreNorm,  0x700ull);
}
BGA_PARITY_TEST(tb_dispatch_post_K6_D8_H2_Df16) {
    run_dispatch(6, 8, 2, 16, NormPlacement::PostNorm, 0x701ull);
}

int main() { return run_all("gpu transformer_block dispatch parity"); }
