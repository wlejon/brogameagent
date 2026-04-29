// End-to-end GPU dispatch parity test for TransformerEncoder.

#include "parity_helpers.h"

#include <brogameagent/nn/transformer_encoder.h>

#include <cuda_runtime.h>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::TransformerEncoder;
using brogameagent::nn::NormPlacement;
using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;

namespace {

void seed_encoder(TransformerEncoder& e, int K, int D, int H, int Df,
                  int n_layers, NormPlacement np, uint64_t seed) {
    TransformerEncoder::Config cfg{};
    cfg.n_layers = n_layers;
    cfg.dim = D; cfg.num_heads = H; cfg.d_ff = Df; cfg.n_slots = K;
    cfg.norm = np;
    uint64_t s = seed;
    e.init(cfg, s);
    SplitMix64 rng(seed ^ 0xF00Dull);
    for (int li = 0; li < n_layers; ++li) {
        auto& blk = e.block(li);
        for (int i = 0; i < blk.mha().Wq().size(); ++i) blk.mha().Wq().data[i] = rng.next_unit() * 0.3f;
        for (int i = 0; i < blk.mha().Wk().size(); ++i) blk.mha().Wk().data[i] = rng.next_unit() * 0.3f;
        for (int i = 0; i < blk.mha().Wv().size(); ++i) blk.mha().Wv().data[i] = rng.next_unit() * 0.3f;
        for (int i = 0; i < blk.mha().Wo().size(); ++i) blk.mha().Wo().data[i] = rng.next_unit() * 0.3f;
        for (int i = 0; i < blk.ff().W1().size(); ++i)  blk.ff().W1().data[i]  = rng.next_unit() * 0.3f;
        for (int i = 0; i < blk.ff().W2().size(); ++i)  blk.ff().W2().data[i]  = rng.next_unit() * 0.3f;
    }
}

void run_dispatch(int K, int D, int H, int Df, int n_layers,
                  NormPlacement np, uint64_t seed) {
    TransformerEncoder cpu, gpu_e;
    seed_encoder(cpu,   K, D, H, Df, n_layers, np, seed);
    seed_encoder(gpu_e, K, D, H, Df, n_layers, np, seed);

    SplitMix64 rng(seed ^ 0xBEEFull);
    Tensor X(K, D), dY(K, D);
    fill_random(X, rng);
    fill_random(dY, rng);

    Tensor Y_cpu(K, D), dX_cpu(K, D);
    cpu.zero_grad();
    cpu.forward(X, nullptr, Y_cpu);
    cpu.backward(dY, dX_cpu);

    gpu_e.to(Device::GPU);
    BGA_CHECK(gpu_e.device() == Device::GPU);
    GpuTensor gX, gY, gdY, gdX;
    upload(X, gX); upload(dY, gdY);
    gY.resize(K, D); gdX.resize(K, D);
    gpu_e.zero_grad();
    gpu_e.forward(gX, nullptr, gY);
    gpu_e.backward(gdY, gdX);

    // Looser tolerance for stacked composite — fp accumulation order varies
    // across CPU vs GPU per-row LN/per-row linear loops compounded over
    // n_layers blocks.
    compare_tensors(Y_cpu,  download_to_host(gY),  "te.dispatch.Y",
                    1e-4f, 2e-3f);
    compare_tensors(dX_cpu, download_to_host(gdX), "te.dispatch.dX",
                    1e-4f, 2e-3f);

    cpu.sgd_step(0.01f, 0.9f);
    gpu_e.sgd_step(0.01f, 0.9f);
    gpu_e.to(Device::CPU);
    compare_tensors(cpu.block(0).mha().Wq(),
                    gpu_e.block(0).mha().Wq(),
                    "te.dispatch.Wq0_after_sgd", 1e-4f, 2e-3f);

    // Save/load round-trip after GPU migration.
    gpu_e.to(Device::GPU);
    std::vector<uint8_t> blob;
    gpu_e.save_to(blob);
    TransformerEncoder restored;
    TransformerEncoder::Config rcfg{};
    rcfg.n_layers = n_layers;
    rcfg.dim = D; rcfg.num_heads = H; rcfg.d_ff = Df; rcfg.n_slots = K;
    rcfg.norm = np;
    uint64_t s = seed;
    restored.init(rcfg, s);
    size_t off = 0;
    restored.load_from(blob.data(), off, blob.size());
    compare_tensors(cpu.block(0).mha().Wq(),
                    restored.block(0).mha().Wq(),
                    "te.dispatch.save_load_Wq0", 1e-4f, 2e-3f);
}

} // namespace

BGA_PARITY_TEST(te_dispatch_pre_K6_D8_H2_L2)  {
    run_dispatch(6, 8, 2, 16, 2, NormPlacement::PreNorm,  0x800ull);
}
BGA_PARITY_TEST(te_dispatch_post_K6_D8_H2_L2) {
    run_dispatch(6, 8, 2, 16, 2, NormPlacement::PostNorm, 0x801ull);
}

int main() { return run_all("gpu transformer_encoder dispatch parity"); }
