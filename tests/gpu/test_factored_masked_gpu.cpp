// Masked softmax / cross-entropy on a GPU device vs the CPU reference.
//
// brotensor's softmax_forward / softmax_xent take the legality mask as a raw
// pointer on the operands' device. factored_softmax / factored_xent used to
// walk host pointers only, and the bro.ai.game.nn binding handed a host
// Float32Array mask straight to the GPU kernel — which Metal refused ("mask is
// not a Metal device pointer") or ignored, and CUDA read as a host address.
// These run the masked ops on the first GPU device with the mask uploaded,
// and check probs / dLogits / loss against the CPU path, masked entries 0.
// Exits 0 with a SKIP line when no GPU backend is registered.

#include "parity_helpers.h"

#include <brogameagent/nn/heads.h>
#include <brotensor/ops.h>

#include <stdexcept>

using namespace bga_parity;
using brogameagent::nn::FactoredPolicyHead;
using brotensor::Device;
using brotensor::Tensor;

namespace {

Device g_gpu = Device::CPU;

constexpr int N_MOVE = FactoredPolicyHead::N_MOVE;
constexpr int N_ATK  = FactoredPolicyHead::N_ATTACK;
constexpr int N_AB   = FactoredPolicyHead::N_ABILITY;
constexpr int TOTAL  = N_MOVE + N_ATK + N_AB;

// One-hot-ish soft target over n entries, all mass on legal entries.
Tensor soft_target(int n, const std::vector<float>& legal) {
    Tensor t = Tensor::vec(n);
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        const bool ok = legal.empty() || legal[static_cast<size_t>(i)] >= 0.5f;
        t[i] = ok ? static_cast<float>(i + 1) : 0.0f;
        s += t[i];
    }
    for (int i = 0; i < n; ++i) t[i] /= s;
    return t;
}

void check_masked_zero(const Tensor& probs, int off, const std::vector<float>& mask) {
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] < 0.5f) BGA_CHECK(probs[off + static_cast<int>(i)] == 0.0f);
    }
}

} // namespace

BGA_PARITY_TEST(softmax_forward_device_mask) {
    const int n = 7;
    SplitMix64 rng(0x5EED01);
    Tensor logits = Tensor::vec(n);
    fill_random(logits, rng, 2.0f);
    const std::vector<float> mask = {1, 0, 1, 1, 0, 1, 0};

    Tensor p_cpu = Tensor::vec(n);
    brotensor::softmax_forward(logits, p_cpu, mask.data());

    Tensor gl = logits.to(g_gpu);
    Tensor gp = Tensor::zeros_on(g_gpu, n, 1);
    Tensor gm = Tensor::from_host_on(g_gpu, mask.data(), n, 1);
    brotensor::softmax_forward(gl, gp, static_cast<const float*>(gm.data));
    Tensor p_gpu = download_to_host(gp);
    compare_tensors(p_cpu, p_gpu, "softmax_forward.masked");
    check_masked_zero(p_gpu, 0, mask);
}

BGA_PARITY_TEST(softmax_xent_device_mask) {
    const int n = 7;
    SplitMix64 rng(0x5EED02);
    Tensor logits = Tensor::vec(n);
    fill_random(logits, rng, 2.0f);
    const std::vector<float> mask = {0, 1, 1, 0, 1, 1, 1};
    Tensor target = soft_target(n, mask);

    Tensor p_cpu = Tensor::vec(n), d_cpu = Tensor::vec(n);
    const float loss_cpu = brotensor::softmax_xent(logits, target, p_cpu, d_cpu, mask.data());

    Tensor gl = logits.to(g_gpu), gt = target.to(g_gpu);
    Tensor gp = Tensor::zeros_on(g_gpu, n, 1), gd = Tensor::zeros_on(g_gpu, n, 1);
    Tensor gm = Tensor::from_host_on(g_gpu, mask.data(), n, 1);
    const float loss_gpu = brotensor::softmax_xent(gl, gt, gp, gd,
                                                   static_cast<const float*>(gm.data));
    BGA_CHECK(std::fabs(loss_cpu - loss_gpu) <= 1e-4f * std::fmax(1.0f, std::fabs(loss_cpu)));
    compare_tensors(p_cpu, download_to_host(gp), "softmax_xent.probs");
    compare_tensors(d_cpu, download_to_host(gd), "softmax_xent.dLogits");
}

BGA_PARITY_TEST(factored_softmax_masked) {
    SplitMix64 rng(0x5EED03);
    Tensor logits = Tensor::vec(TOTAL);
    fill_random(logits, rng, 2.0f);
    std::vector<float> amask(N_ATK - 1, 1.0f), bmask(N_AB - 1, 1.0f);
    amask[0] = 0.0f;
    if (N_ATK - 1 > 2) amask[2] = 0.0f;
    bmask[N_AB - 2] = 0.0f;

    for (int variant = 0; variant < 2; ++variant) {
        const float* am = variant == 0 ? amask.data() : nullptr;
        const float* bm = variant == 0 ? bmask.data() : nullptr;
        Tensor p_cpu = Tensor::vec(TOTAL);
        brogameagent::nn::factored_softmax(logits, p_cpu, am, bm);

        Tensor gl = logits.to(g_gpu);
        Tensor gp = Tensor::zeros_on(g_gpu, TOTAL, 1);
        brogameagent::nn::factored_softmax(gl, gp, am, bm);
        Tensor p_gpu = download_to_host(gp);
        compare_tensors(p_cpu, p_gpu, variant == 0 ? "factored_softmax.masked"
                                                   : "factored_softmax.unmasked");
        if (variant == 0) {
            check_masked_zero(p_gpu, N_MOVE, amask);
            check_masked_zero(p_gpu, N_MOVE + N_ATK, bmask);
        }
    }
}

BGA_PARITY_TEST(factored_xent_masked) {
    SplitMix64 rng(0x5EED04);
    Tensor logits = Tensor::vec(TOTAL);
    fill_random(logits, rng, 2.0f);
    std::vector<float> amask(N_ATK - 1, 1.0f), bmask(N_AB - 1, 1.0f);
    amask[1] = 0.0f;
    bmask[0] = 0.0f;
    std::vector<float> amask_full(amask), bmask_full(bmask);
    amask_full.push_back(1.0f);
    bmask_full.push_back(1.0f);
    Tensor mt = soft_target(N_MOVE, {});
    Tensor at = soft_target(N_ATK, amask_full);
    Tensor bt = soft_target(N_AB, bmask_full);

    Tensor p_cpu = Tensor::vec(TOTAL), d_cpu = Tensor::vec(TOTAL);
    const float loss_cpu = brogameagent::nn::factored_xent(logits, mt, at, bt, p_cpu, d_cpu,
                                                           amask.data(), bmask.data());

    Tensor gl = logits.to(g_gpu);
    Tensor gmt = mt.to(g_gpu), gat = at.to(g_gpu), gbt = bt.to(g_gpu);
    Tensor gp = Tensor::zeros_on(g_gpu, TOTAL, 1), gd = Tensor::zeros_on(g_gpu, TOTAL, 1);
    const float loss_gpu = brogameagent::nn::factored_xent(gl, gmt, gat, gbt, gp, gd,
                                                           amask.data(), bmask.data());
    BGA_CHECK(std::fabs(loss_cpu - loss_gpu) <= 1e-4f * std::fmax(1.0f, std::fabs(loss_cpu)));
    Tensor p_gpu = download_to_host(gp);
    compare_tensors(p_cpu, p_gpu, "factored_xent.probs");
    compare_tensors(d_cpu, download_to_host(gd), "factored_xent.dLogits");
    check_masked_zero(p_gpu, N_MOVE, amask);
    check_masked_zero(p_gpu, N_MOVE + N_ATK, bmask);
}

BGA_PARITY_TEST(factored_rejects_mixed_devices) {
    Tensor gl = Tensor::zeros_on(g_gpu, TOTAL, 1);
    Tensor p_cpu = Tensor::vec(TOTAL);
    bool threw = false;
    try {
        brogameagent::nn::factored_softmax(gl, p_cpu);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    BGA_CHECK(threw);
}

int main() {
    brotensor::init();
    for (const Device& d : brotensor::available_devices()) {
        if (d.is_gpu()) { g_gpu = d; break; }
    }
    if (!g_gpu.is_gpu()) {
        std::printf("SKIP: no GPU backend registered\n");
        return 0;
    }
    std::printf("GPU device: %s\n", brotensor::to_string(g_gpu).c_str());
    return run_all("Masked softmax / xent GPU parity");
}
