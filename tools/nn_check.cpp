// nn_check — finite-difference gradient checks for every NN circuit.
//
// Each test compares the analytic gradient produced by backward() against
// the numerical gradient from (f(x+h) - f(x-h)) / (2h). Agreement to ~1e-3
// relative error is expected for fp32.
//
// Usage: nn_check [--verbose]
//   Exits 0 on all-pass, 1 on any failure. Prints TSV of per-test results
//   so it composes into sweep logging.

#include "brogameagent/nn/autoencoder.h"
#include "brogameagent/nn/circuits.h"
#include "brogameagent/nn/decoder.h"
#include "brogameagent/nn/encoder.h"
#include "brogameagent/nn/heads.h"
#include "brogameagent/nn/net.h"
#include "brogameagent/nn/ops.h"
#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/tensor.h"
#include "brogameagent/nn/layernorm.h"
#include "brogameagent/nn/embedding.h"
#include "brogameagent/nn/attention.h"
#include "brogameagent/nn/gru.h"
#include "brogameagent/nn/set_transformer.h"
#include "brogameagent/nn/net_st.h"
#include "brogameagent/nn/heads_dist.h"
#include "brogameagent/nn/ensemble.h"
#include "brogameagent/nn/forward_model.h"
#include "brogameagent/learn/contrastive.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace brogameagent::nn;

static int g_fail = 0;
static int g_pass = 0;
static bool g_verbose = false;

static float rel_err(float a, float b) {
    const float denom = std::max(1e-6f, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) / denom;
}

// Numerical gradient dL/dp[i] via central difference on parameter/input p[i].
// loss_fn takes no args (reads current param state).
template <typename Loss>
static void numerical_grad(float* p, int n, Loss loss_fn, std::vector<float>& out, float h = 1e-3f) {
    out.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float saved = p[i];
        p[i] = saved + h;
        const float lp = loss_fn();
        p[i] = saved - h;
        const float lm = loss_fn();
        p[i] = saved;
        out[i] = (lp - lm) / (2.0f * h);
    }
}

static bool check_grad(const std::vector<float>& analytic,
                       const std::vector<float>& numeric,
                       const char* label, float tol_rel = 5e-3f,
                       float tol_abs = 1e-4f) {
    float max_err = 0.0f;
    int arg = -1;
    // Use the worse of (relative error) unless both values are tiny, in
    // which case only absolute error matters. An entry passes if it's
    // within tol_rel relatively OR tol_abs absolutely.
    for (size_t i = 0; i < analytic.size(); ++i) {
        const float abs_diff = std::fabs(analytic[i] - numeric[i]);
        if (abs_diff < tol_abs) continue;
        const float e = rel_err(analytic[i], numeric[i]);
        if (e > max_err) { max_err = e; arg = static_cast<int>(i); }
    }
    const bool ok = max_err < tol_rel;
    std::printf("%s\t%s\tmax_rel_err=%.4e\tat=%d\n",
                label, ok ? "PASS" : "FAIL", max_err, arg);
    if (ok) ++g_pass; else ++g_fail;
    if (!ok && g_verbose) {
        std::printf("  analytic[%d] = %.6f   numeric[%d] = %.6f\n",
                    arg, analytic[arg], arg, numeric[arg]);
    }
    return ok;
}

// ───────────────────── Linear ─────────────────────
static void test_linear() {
    uint64_t seed = 42;
    const int in_dim = 5, out_dim = 3;
    Linear L(in_dim, out_dim, seed);

    Tensor x = Tensor::vec(in_dim);
    for (int i = 0; i < in_dim; ++i) x[i] = 0.1f * (i + 1) - 0.3f;
    Tensor y = Tensor::vec(out_dim);
    Tensor target = Tensor::vec(out_dim);
    for (int i = 0; i < out_dim; ++i) target[i] = 0.2f * i - 0.1f;

    auto loss_fn = [&]() {
        L.forward(x, y);
        float l = 0;
        for (int i = 0; i < out_dim; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    // Analytic path.
    L.zero_grad();
    L.forward(x, y);
    Tensor dY = Tensor::vec(out_dim);
    for (int i = 0; i < out_dim; ++i) dY[i] = y[i] - target[i];
    Tensor dX_analytic = Tensor::vec(in_dim);
    L.backward(dY, dX_analytic);

    // Numerical dX.
    std::vector<float> dX_num;
    numerical_grad(x.ptr(), in_dim, loss_fn, dX_num);
    std::vector<float> dX_a(in_dim);
    for (int i = 0; i < in_dim; ++i) dX_a[i] = dX_analytic[i];
    check_grad(dX_a, dX_num, "Linear/dX");

    // Numerical dW.
    std::vector<float> dW_num;
    numerical_grad(L.W().ptr(), L.W().size(), loss_fn, dW_num);
    std::vector<float> dW_a(L.W().size());
    for (int i = 0; i < L.W().size(); ++i) dW_a[i] = L.dW()[i];
    check_grad(dW_a, dW_num, "Linear/dW");

    // Numerical dB.
    std::vector<float> dB_num;
    numerical_grad(L.b().ptr(), L.b().size(), loss_fn, dB_num);
    std::vector<float> dB_a(L.b().size());
    for (int i = 0; i < L.b().size(); ++i) dB_a[i] = L.dB()[i];
    check_grad(dB_a, dB_num, "Linear/dB");
}

// ───────────────────── Relu / Tanh ─────────────────────
static void test_relu() {
    const int n = 6;
    Tensor x = Tensor::vec(n);
    for (int i = 0; i < n; ++i) x[i] = 0.3f * i - 0.8f;
    Tensor y = Tensor::vec(n);
    Tensor target = Tensor::vec(n);
    for (int i = 0; i < n; ++i) target[i] = 0.1f * i;

    auto loss_fn = [&]() {
        Relu R;
        R.forward(x, y);
        float l = 0;
        for (int i = 0; i < n; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    Relu R;
    R.forward(x, y);
    Tensor dY = Tensor::vec(n);
    for (int i = 0; i < n; ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(n);
    R.backward(dY, dX);

    std::vector<float> dX_num;
    numerical_grad(x.ptr(), n, loss_fn, dX_num);
    std::vector<float> dX_a(n);
    for (int i = 0; i < n; ++i) dX_a[i] = dX[i];
    check_grad(dX_a, dX_num, "ReLU/dX");
}

static void test_tanh() {
    const int n = 5;
    Tensor x = Tensor::vec(n);
    for (int i = 0; i < n; ++i) x[i] = 0.2f * i - 0.5f;
    Tensor y = Tensor::vec(n);
    Tensor target = Tensor::vec(n);
    for (int i = 0; i < n; ++i) target[i] = 0.1f * i - 0.2f;

    auto loss_fn = [&]() {
        Tanh T;
        T.forward(x, y);
        float l = 0;
        for (int i = 0; i < n; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    Tanh T;
    T.forward(x, y);
    Tensor dY = Tensor::vec(n);
    for (int i = 0; i < n; ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(n);
    T.backward(dY, dX);

    std::vector<float> dX_num;
    numerical_grad(x.ptr(), n, loss_fn, dX_num);
    std::vector<float> dX_a(n);
    for (int i = 0; i < n; ++i) dX_a[i] = dX[i];
    check_grad(dX_a, dX_num, "Tanh/dX");
}

// ───────────────────── Softmax / Xent ─────────────────────
static void test_softmax_xent() {
    const int n = 4;
    Tensor logits = Tensor::vec(n);
    for (int i = 0; i < n; ++i) logits[i] = 0.1f * i - 0.15f;
    Tensor target = Tensor::vec(n);
    target[0] = 0.1f; target[1] = 0.2f; target[2] = 0.3f; target[3] = 0.4f;
    Tensor probs = Tensor::vec(n);
    Tensor dLogits = Tensor::vec(n);

    auto loss_fn = [&]() {
        Tensor p = Tensor::vec(n);
        Tensor dz = Tensor::vec(n);
        return softmax_xent(logits, target, p, dz, nullptr);
    };

    softmax_xent(logits, target, probs, dLogits, nullptr);

    std::vector<float> num;
    numerical_grad(logits.ptr(), n, loss_fn, num);
    std::vector<float> ana(n);
    for (int i = 0; i < n; ++i) ana[i] = dLogits[i];
    check_grad(ana, num, "SoftmaxXent/dLogits");
}

// ───────────────────── DeepSetsEncoder ─────────────────────
static void test_encoder() {
    uint64_t seed = 7;
    DeepSetsEncoder::Config cfg;
    cfg.hidden = 8;
    cfg.embed_dim = 6;
    DeepSetsEncoder enc;
    enc.init(cfg, seed);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.01f * (i - 60);
    // Mark some slots valid.
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f; // enemy slot 0 valid
    x[brogameagent::observation::SELF_FEATURES + brogameagent::observation::ENEMY_FEATURES] = 1.0f; // slot 1
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES * brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;  // ally slot 0 valid

    Tensor y = Tensor::vec(enc.out_dim());
    Tensor target = Tensor::vec(enc.out_dim());
    for (int i = 0; i < enc.out_dim(); ++i) target[i] = 0.01f * i;

    auto loss_fn = [&]() {
        Tensor yy = Tensor::vec(enc.out_dim());
        enc.forward(x, yy);
        float l = 0;
        for (int i = 0; i < enc.out_dim(); ++i) { float d = yy[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    enc.forward(x, y);
    Tensor dY = Tensor::vec(enc.out_dim());
    for (int i = 0; i < enc.out_dim(); ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(x.size());
    enc.backward(dY, dX);

    std::vector<float> num;
    numerical_grad(x.ptr(), x.size(), loss_fn, num, 1e-3f);
    std::vector<float> ana(x.size());
    for (int i = 0; i < x.size(); ++i) ana[i] = dX[i];
    // Use a looser tol because fp32 through a multi-layer net accumulates noise.
    check_grad(ana, num, "DeepSetsEncoder/dX", 2e-2f, 1e-3f);
}

// ───────────────────── SingleHeroNet (e2e param grad spot-check) ─────────────────────
static void test_full_net() {
    SingleHeroNet net;
    SingleHeroNet::Config cfg;
    cfg.enc.hidden = 8;
    cfg.enc.embed_dim = 8;
    cfg.trunk_hidden = 12;
    cfg.value_hidden = 8;
    cfg.seed = 12345;
    net.init(cfg);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.02f * i - 0.5f;
    // Valid slots.
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f;
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES * brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;

    const float v_target = 0.25f;

    auto loss_fn = [&]() {
        float v = 0;
        Tensor lg = Tensor::vec(net.policy_logits());
        net.forward(x, v, lg);
        float dv = 0;
        return mse_scalar(v, v_target, dv);
    };

    // Simple param grad check: perturb a single linear weight and compare.
    // Forward + backward to populate grads.
    net.zero_grad();
    float v = 0;
    Tensor logits = Tensor::vec(net.policy_logits());
    net.forward(x, v, logits);
    float dv = 0;
    mse_scalar(v, v_target, dv);
    Tensor dLog = Tensor::vec(net.policy_logits());
    dLog.zero(); // policy is not part of this loss
    net.backward(dv, dLog);

    // We can't easily get at internal grads from here — do it via save/load
    // round-trip sanity instead.
    auto blob = net.save();
    SingleHeroNet net2;
    net2.init(cfg);
    net2.load(blob);
    float v2 = 0;
    Tensor lg2 = Tensor::vec(net2.policy_logits());
    net2.forward(x, v2, lg2);
    const float err = std::fabs(v - v2);
    std::printf("SingleHeroNet/save-load\t%s\tvalue_err=%.4e\n",
                err < 1e-6f ? "PASS" : "FAIL", err);
    if (err < 1e-6f) ++g_pass; else ++g_fail;

    // Numerical sanity: loss is sensitive to x (perturb an obs entry).
    std::vector<float> num;
    numerical_grad(x.ptr(), 3, loss_fn, num);  // check first 3 entries
    // Compare against forward-only Jacobian via finite diff (analytic path not exposed).
    // Just verify at least one entry moves non-trivially.
    bool any_nonzero = false;
    for (float g : num) if (std::fabs(g) > 1e-5f) any_nonzero = true;
    std::printf("SingleHeroNet/sensitivity\t%s\n", any_nonzero ? "PASS" : "FAIL");
    if (any_nonzero) ++g_pass; else ++g_fail;
    (void)loss_fn;
}

// ───────────────────── PolicyValueNet (gradient + save/load) ─────────────────────
static void test_policy_value_net() {
    PolicyValueNet net;
    PolicyValueNet::Config cfg;
    cfg.in_dim       = 12;
    cfg.hidden       = {16, 12};
    cfg.value_hidden = 8;
    cfg.num_actions  = 6;
    cfg.seed         = 0xA11CE;
    net.init(cfg);

    Tensor x = Tensor::vec(cfg.in_dim);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.07f * i - 0.4f;

    // Soft target distribution (sums to 1) and a value target in [-1, 1].
    Tensor tgt = Tensor::vec(cfg.num_actions);
    {
        float s = 0.0f;
        for (int i = 0; i < cfg.num_actions; ++i) { tgt[i] = 0.1f + 0.05f * i; s += tgt[i]; }
        for (int i = 0; i < cfg.num_actions; ++i) tgt[i] /= s;
    }
    const float v_target = 0.3f;

    auto loss_fn = [&]() -> float {
        float v = 0.0f;
        Tensor lg = Tensor::vec(cfg.num_actions);
        net.forward(x, v, lg);
        float dv = 0.0f;
        const float lv = mse_scalar(v, v_target, dv);
        Tensor pr = Tensor::vec(cfg.num_actions);
        Tensor dL = Tensor::vec(cfg.num_actions);
        const float lp = softmax_xent(lg, tgt, pr, dL, nullptr);
        return lv + lp;
    };

    // Sensitivity: numeric grad over a few obs entries should be non-trivially
    // nonzero (loss responds to inputs at all).
    std::vector<float> num;
    numerical_grad(x.ptr(), 4, loss_fn, num, 1e-3f);
    bool any_nonzero = false;
    for (float g : num) if (std::fabs(g) > 1e-5f) any_nonzero = true;
    std::printf("PolicyValueNet/sensitivity\t%s\n", any_nonzero ? "PASS" : "FAIL");
    if (any_nonzero) ++g_pass; else ++g_fail;

    // Save/load round-trip preserves outputs bit-exactly (same fp ops).
    float v1 = 0.0f;
    Tensor lg1 = Tensor::vec(cfg.num_actions);
    net.forward(x, v1, lg1);

    auto blob = net.save();
    PolicyValueNet net2;
    net2.init(cfg);
    net2.load(blob);
    float v2 = 0.0f;
    Tensor lg2 = Tensor::vec(cfg.num_actions);
    net2.forward(x, v2, lg2);

    float max_err = std::fabs(v1 - v2);
    for (int i = 0; i < lg1.size(); ++i)
        max_err = std::max(max_err, std::fabs(lg1[i] - lg2[i]));
    const bool ok = max_err < 1e-6f;
    std::printf("PolicyValueNet/save-load\t%s\tmax_err=%.4e\n", ok ? "PASS" : "FAIL", max_err);
    if (ok) ++g_pass; else ++g_fail;

    // SGD reduces loss on this single example after a few steps.
    const float l0 = loss_fn();
    for (int i = 0; i < 50; ++i) {
        net.zero_grad();
        float v = 0.0f;
        Tensor lg = Tensor::vec(cfg.num_actions);
        net.forward(x, v, lg);
        float dv = 0.0f;
        mse_scalar(v, v_target, dv);
        Tensor pr = Tensor::vec(cfg.num_actions);
        Tensor dL = Tensor::vec(cfg.num_actions);
        softmax_xent(lg, tgt, pr, dL, nullptr);
        net.backward(dv, dL);
        net.sgd_step(0.05f, 0.9f);
    }
    const float l1 = loss_fn();
    const bool sgd_ok = l1 < l0 * 0.95f;
    std::printf("PolicyValueNet/sgd-decreases\t%s\tl0=%.4e\tl1=%.4e\n",
                sgd_ok ? "PASS" : "FAIL", l0, l1);
    if (sgd_ok) ++g_pass; else ++g_fail;
}

// ───────────────────── DeepSetsDecoder ─────────────────────
static void test_decoder() {
    uint64_t seed = 13;
    DeepSetsDecoder::Config cfg;
    cfg.embed_dim = 4;
    cfg.hidden    = 4;
    DeepSetsDecoder dec;
    dec.init(cfg, seed);

    Tensor x = Tensor::vec(dec.in_dim());
    for (int i = 0; i < x.size(); ++i) x[i] = 0.05f * i - 0.15f;
    Tensor y = Tensor::vec(dec.out_dim());
    Tensor target = Tensor::vec(dec.out_dim());
    for (int i = 0; i < dec.out_dim(); ++i) target[i] = 0.01f * i - 0.2f;

    auto loss_fn = [&]() {
        Tensor yy = Tensor::vec(dec.out_dim());
        dec.forward(x, yy);
        float l = 0;
        for (int i = 0; i < dec.out_dim(); ++i) { float d = yy[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    dec.forward(x, y);
    Tensor dY = Tensor::vec(dec.out_dim());
    for (int i = 0; i < dec.out_dim(); ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(dec.in_dim());
    dec.backward(dY, dX);

    std::vector<float> num;
    numerical_grad(x.ptr(), x.size(), loss_fn, num, 1e-3f);
    std::vector<float> ana(x.size());
    for (int i = 0; i < x.size(); ++i) ana[i] = dX[i];
    check_grad(ana, num, "DeepSetsDecoder/dX", 2e-2f, 1e-3f);
}

// ───────────────────── DeepSetsAutoencoder ─────────────────────
static void test_autoencoder() {
    DeepSetsAutoencoder ae;
    DeepSetsAutoencoder::Config cfg;
    cfg.enc.embed_dim = 4;
    cfg.enc.hidden    = 4;
    cfg.dec_hidden    = 4;
    cfg.seed          = 0xABCDEFULL;
    ae.init(cfg);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.01f * (i - 50);
    // Mark a couple slots valid so masking doesn't zero everything.
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f;
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES *
                      brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;

    Tensor x_hat = Tensor::vec(x.size());
    Tensor dXh   = Tensor::vec(x.size());

    auto loss_fn = [&]() {
        Tensor xh = Tensor::vec(x.size());
        Tensor dd = Tensor::vec(x.size());
        ae.forward(x, xh);
        return reconstruction_loss(x, xh, dd);
    };

    ae.forward(x, x_hat);
    reconstruction_loss(x, x_hat, dXh);
    // The loss above returns mean-per-element; its gradient is (x_hat - x)
    // but the analytic gradient we pass to backward is the raw (x_hat - x)
    // without the 1/N factor. For grad-check, build numerical grad of the
    // *same* loss_fn to match. dX (input to AE) is what we're checking.
    // Note: reconstruction_loss produces dXh = (x_hat - x) on valid entries.
    // numerical_grad observes loss = sum_{valid} 0.5*d^2 / N; so analytic
    // dx needs scaling by 1/N to match the numerical. We instead check a
    // *sum-loss* variant here to keep the analytic grad unchanged.
    //
    // Simpler: use a custom sum-loss that matches the dXh = (x_hat - x)
    // convention directly.
    auto sum_loss_fn = [&]() {
        Tensor xh = Tensor::vec(x.size());
        ae.forward(x, xh);
        float l = 0.0f;
        for (int i = 0; i < brogameagent::observation::SELF_FEATURES; ++i) {
            float d = xh[i] - x[i]; l += 0.5f * d * d;
        }
        const int off_e = brogameagent::observation::SELF_FEATURES;
        for (int k = 0; k < brogameagent::observation::K_ENEMIES; ++k) {
            const int base = off_e + k * brogameagent::observation::ENEMY_FEATURES;
            if (x[base] <= 0.5f) continue;
            for (int j = 0; j < brogameagent::observation::ENEMY_FEATURES; ++j) {
                float d = xh[base + j] - x[base + j]; l += 0.5f * d * d;
            }
        }
        const int off_al = off_e + brogameagent::observation::K_ENEMIES *
                           brogameagent::observation::ENEMY_FEATURES;
        for (int k = 0; k < brogameagent::observation::K_ALLIES; ++k) {
            const int base = off_al + k * brogameagent::observation::ALLY_FEATURES;
            if (x[base] <= 0.5f) continue;
            for (int j = 0; j < brogameagent::observation::ALLY_FEATURES; ++j) {
                float d = xh[base + j] - x[base + j]; l += 0.5f * d * d;
            }
        }
        return l;
    };

    // Re-run forward to refresh caches, then backward with dXh = (xhat - x).
    ae.forward(x, x_hat);
    reconstruction_loss(x, x_hat, dXh);
    ae.zero_grad();
    ae.backward(dXh);

    // Now verify against sum_loss_fn by perturbing x. AE::backward does not
    // expose dX_observation, but numerical grad wrt the obs is still
    // computable externally and should equal the encoder's backward product.
    // Instead we check the decoder-side loss-vs-reconstruction path by
    // grad-checking the *sum_loss* wrt x (should be zero for invalid entries
    // that do not appear in the loss and do not influence xhat — except
    // they *do* influence xhat through the valid-flag masking in the
    // encoder's pool. We limit the check to the self block which has no
    // masking interaction.)
    const int Nself = brogameagent::observation::SELF_FEATURES;
    std::vector<float> num;
    numerical_grad(x.ptr(), Nself, sum_loss_fn, num, 1e-3f);
    // Analytic: numerical derivative of sum_loss_fn wrt x[i] is
    //   sum_{valid j} (xhat_j - x_j) * (d xhat_j / d x_i  -  [i==j ? 1 : 0])
    // For the self block, the first SELF_FEATURES entries of dXh are
    // (xhat - x), but we also need the contribution through the encoder.
    // That's exactly what enc.backward produces via dX_obs, so we recompute
    // it here by running forward+backward and asking the encoder for dX.
    //
    // Because AE.backward() discards dX_obs, we drive the encoder directly.
    Tensor dEmbed = Tensor::vec(ae.encoder().out_dim());
    Tensor embed  = Tensor::vec(ae.encoder().out_dim());
    ae.encoder().forward(x, embed);
    Tensor xh2 = Tensor::vec(x.size());
    ae.decoder().forward(embed, xh2);
    Tensor dXhat2 = Tensor::vec(x.size());
    reconstruction_loss(x, xh2, dXhat2);
    ae.decoder().backward(dXhat2, dEmbed);
    Tensor dX_obs = Tensor::vec(x.size());
    ae.encoder().backward(dEmbed, dX_obs);
    // Total dL/dx_i = -dXhat[i] + dX_obs[i]  (the -1 from d/dx_i of (xhat_i - x_i))
    std::vector<float> ana(Nself);
    for (int i = 0; i < Nself; ++i) ana[i] = dX_obs[i] - dXhat2[i];
    check_grad(ana, num, "DeepSetsAutoencoder/dX_self", 2e-2f, 2e-3f);
}

// ───────────────────── Sigmoid ─────────────────────
static void test_sigmoid() {
    const int n = 5;
    Tensor x = Tensor::vec(n);
    for (int i = 0; i < n; ++i) x[i] = 0.3f * i - 0.7f;
    Tensor y = Tensor::vec(n);
    Tensor target = Tensor::vec(n);
    for (int i = 0; i < n; ++i) target[i] = 0.1f * i;

    auto loss_fn = [&]() {
        Sigmoid S;
        S.forward(x, y);
        float l = 0;
        for (int i = 0; i < n; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    Sigmoid S;
    S.forward(x, y);
    Tensor dY = Tensor::vec(n);
    for (int i = 0; i < n; ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(n);
    S.backward(dY, dX);

    std::vector<float> num;
    numerical_grad(x.ptr(), n, loss_fn, num);
    std::vector<float> ana(n);
    for (int i = 0; i < n; ++i) ana[i] = dX[i];
    check_grad(ana, num, "Sigmoid/dX");
}

// ───────────────────── LayerNorm ─────────────────────
static void test_layernorm() {
    const int n = 7;
    LayerNorm ln(n);
    // Perturb gamma/beta off defaults so grads are non-trivial.
    for (int i = 0; i < n; ++i) { ln.gamma()[i] = 0.8f + 0.05f * i; ln.beta()[i] = 0.1f * i - 0.2f; }

    Tensor x = Tensor::vec(n);
    for (int i = 0; i < n; ++i) x[i] = 0.2f * i - 0.5f;
    Tensor y = Tensor::vec(n);
    Tensor target = Tensor::vec(n);
    for (int i = 0; i < n; ++i) target[i] = 0.05f * i + 0.1f;

    auto loss_fn = [&]() {
        ln.forward(x, y);
        float l = 0;
        for (int i = 0; i < n; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    ln.zero_grad();
    ln.forward(x, y);
    Tensor dY = Tensor::vec(n);
    for (int i = 0; i < n; ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(n);
    ln.backward(dY, dX);

    std::vector<float> dX_num;
    numerical_grad(x.ptr(), n, loss_fn, dX_num, 5e-3f);
    std::vector<float> dX_a(n);
    for (int i = 0; i < n; ++i) dX_a[i] = dX[i];
    check_grad(dX_a, dX_num, "LayerNorm/dX", 1e-2f);

    std::vector<float> dG_num;
    numerical_grad(ln.gamma().ptr(), n, loss_fn, dG_num);
    std::vector<float> dG_a(n);
    for (int i = 0; i < n; ++i) dG_a[i] = ln.dGamma()[i];
    check_grad(dG_a, dG_num, "LayerNorm/dGamma");

    std::vector<float> dB_num;
    numerical_grad(ln.beta().ptr(), n, loss_fn, dB_num);
    std::vector<float> dB_a(n);
    for (int i = 0; i < n; ++i) dB_a[i] = ln.dBeta()[i];
    check_grad(dB_a, dB_num, "LayerNorm/dBeta");
}

// ───────────────────── Embedding ─────────────────────
static void test_embedding() {
    uint64_t seed = 123;
    const int vocab = 6, dim = 4;
    Embedding E(vocab, dim, seed);

    const int idx = 3;
    Tensor y = Tensor::vec(dim);
    Tensor target = Tensor::vec(dim);
    for (int i = 0; i < dim; ++i) target[i] = 0.1f * i - 0.2f;

    auto loss_fn = [&]() {
        E.forward(idx, y);
        float l = 0;
        for (int i = 0; i < dim; ++i) { float d = y[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    E.zero_grad();
    E.forward(idx, y);
    Tensor dY = Tensor::vec(dim);
    for (int i = 0; i < dim; ++i) dY[i] = y[i] - target[i];
    E.backward(idx, dY);

    std::vector<float> dW_num;
    numerical_grad(E.W().ptr(), E.W().size(), loss_fn, dW_num);
    std::vector<float> dW_a(E.W().size());
    for (int i = 0; i < E.W().size(); ++i) dW_a[i] = E.dW()[i];
    check_grad(dW_a, dW_num, "Embedding/dW");
}

// ───────────────────── ScaledDotProductAttention ─────────────────────
static void test_attention() {
    uint64_t seed = 321;
    const int N = 4, D = 5;
    ScaledDotProductAttention A;
    A.init(N, D, seed);

    Tensor X = Tensor::mat(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            X(i, j) = 0.05f * (i + 1) - 0.1f * j + 0.02f * i * j;

    // Mask: last slot invalid.
    std::vector<float> mask = {1.0f, 1.0f, 1.0f, 0.0f};

    Tensor O = Tensor::mat(N, D);
    Tensor target = Tensor::mat(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            target(i, j) = 0.02f * i - 0.03f * j;

    auto loss_fn = [&]() {
        Tensor OO = Tensor::mat(N, D);
        A.forward(X, mask.data(), OO);
        float l = 0;
        for (int k = 0; k < OO.size(); ++k) { float d = OO[k] - target[k]; l += 0.5f * d * d; }
        return l;
    };

    A.zero_grad();
    A.forward(X, mask.data(), O);
    Tensor dO = Tensor::mat(N, D);
    for (int k = 0; k < O.size(); ++k) dO[k] = O[k] - target[k];
    Tensor dX = Tensor::mat(N, D);
    A.backward(dO, dX);

    const float tol = 2e-2f;

    std::vector<float> dX_num;
    numerical_grad(X.ptr(), X.size(), loss_fn, dX_num);
    std::vector<float> dX_a(X.size());
    for (int i = 0; i < X.size(); ++i) dX_a[i] = dX[i];
    check_grad(dX_a, dX_num, "Attention/dX", tol);

    std::vector<float> dWq_num;
    numerical_grad(A.Wq().ptr(), A.Wq().size(), loss_fn, dWq_num);
    std::vector<float> dWq_a(A.Wq().size());
    for (int i = 0; i < A.Wq().size(); ++i) dWq_a[i] = A.dWq()[i];
    check_grad(dWq_a, dWq_num, "Attention/dWq", tol);

    std::vector<float> dWk_num;
    numerical_grad(A.Wk().ptr(), A.Wk().size(), loss_fn, dWk_num);
    std::vector<float> dWk_a(A.Wk().size());
    for (int i = 0; i < A.Wk().size(); ++i) dWk_a[i] = A.dWk()[i];
    check_grad(dWk_a, dWk_num, "Attention/dWk", tol);

    std::vector<float> dWv_num;
    numerical_grad(A.Wv().ptr(), A.Wv().size(), loss_fn, dWv_num);
    std::vector<float> dWv_a(A.Wv().size());
    for (int i = 0; i < A.Wv().size(); ++i) dWv_a[i] = A.dWv()[i];
    check_grad(dWv_a, dWv_num, "Attention/dWv", tol);

    std::vector<float> dWo_num;
    numerical_grad(A.Wo().ptr(), A.Wo().size(), loss_fn, dWo_num);
    std::vector<float> dWo_a(A.Wo().size());
    for (int i = 0; i < A.Wo().size(); ++i) dWo_a[i] = A.dWo()[i];
    check_grad(dWo_a, dWo_num, "Attention/dWo", tol);
}

// ───────────────────── GRUCell ─────────────────────
static void test_gru() {
    uint64_t seed = 999;
    const int I = 4, H = 3;
    GRUCell G(I, H, seed);

    Tensor x = Tensor::vec(I);
    for (int i = 0; i < I; ++i) x[i] = 0.1f * i - 0.2f;
    Tensor h_prev = Tensor::vec(H);
    for (int i = 0; i < H; ++i) h_prev[i] = 0.15f * i - 0.1f;
    Tensor h = Tensor::vec(H);
    Tensor target = Tensor::vec(H);
    for (int i = 0; i < H; ++i) target[i] = 0.05f * i + 0.1f;

    auto loss_fn = [&]() {
        G.forward(x, h_prev, h);
        float l = 0;
        for (int i = 0; i < H; ++i) { float d = h[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    G.zero_grad();
    G.forward(x, h_prev, h);
    Tensor dH = Tensor::vec(H);
    for (int i = 0; i < H; ++i) dH[i] = h[i] - target[i];
    Tensor dX = Tensor::vec(I), dHp = Tensor::vec(H);
    G.backward(dH, dX, dHp);

    std::vector<float> dX_num;
    numerical_grad(x.ptr(), I, loss_fn, dX_num);
    std::vector<float> dX_a(I);
    for (int i = 0; i < I; ++i) dX_a[i] = dX[i];
    check_grad(dX_a, dX_num, "GRU/dX");

    std::vector<float> dHp_num;
    numerical_grad(h_prev.ptr(), H, loss_fn, dHp_num);
    std::vector<float> dHp_a(H);
    for (int i = 0; i < H; ++i) dHp_a[i] = dHp[i];
    check_grad(dHp_a, dHp_num, "GRU/dH_prev");

    std::vector<float> dWir_num;
    numerical_grad(G.W_ir().ptr(), G.W_ir().size(), loss_fn, dWir_num);
    std::vector<float> dWir_a(G.W_ir().size());
    for (int i = 0; i < G.W_ir().size(); ++i) dWir_a[i] = G.dW_ir()[i];
    check_grad(dWir_a, dWir_num, "GRU/dW_ir");

    std::vector<float> dWhn_num;
    numerical_grad(G.W_hn().ptr(), G.W_hn().size(), loss_fn, dWhn_num);
    std::vector<float> dWhn_a(G.W_hn().size());
    for (int i = 0; i < G.W_hn().size(); ++i) dWhn_a[i] = G.dW_hn()[i];
    check_grad(dWhn_a, dWhn_num, "GRU/dW_hn");
}

// ───────────────────── SetTransformerEncoder ─────────────────────
static void test_set_transformer() {
    uint64_t seed = 77;
    SetTransformerEncoder::Config cfg;
    cfg.hidden = 6;
    cfg.embed_dim = 6;
    SetTransformerEncoder enc;
    enc.init(cfg, seed);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.01f * (i - 55);
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f;
    x[brogameagent::observation::SELF_FEATURES + brogameagent::observation::ENEMY_FEATURES] = 1.0f;
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES * brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;

    Tensor y = Tensor::vec(enc.out_dim());
    Tensor target = Tensor::vec(enc.out_dim());
    for (int i = 0; i < enc.out_dim(); ++i) target[i] = 0.01f * i;

    auto loss_fn = [&]() {
        Tensor yy = Tensor::vec(enc.out_dim());
        enc.forward(x, yy);
        float l = 0;
        for (int i = 0; i < enc.out_dim(); ++i) { float d = yy[i] - target[i]; l += 0.5f * d * d; }
        return l;
    };

    enc.zero_grad();
    enc.forward(x, y);
    Tensor dY = Tensor::vec(enc.out_dim());
    for (int i = 0; i < enc.out_dim(); ++i) dY[i] = y[i] - target[i];
    Tensor dX = Tensor::vec(x.size());
    enc.backward(dY, dX);

    std::vector<float> num;
    numerical_grad(x.ptr(), x.size(), loss_fn, num, 1e-3f);
    std::vector<float> ana(x.size());
    for (int i = 0; i < x.size(); ++i) ana[i] = dX[i];
    check_grad(ana, num, "SetTransformerEncoder/dX", 2e-2f, 2e-3f);
}

// ───────────────────── SingleHeroNetST (save/load round-trip) ─────────────────────
static void test_single_hero_net_st() {
    SingleHeroNetST net;
    SingleHeroNetST::Config cfg;
    cfg.enc.hidden = 6;
    cfg.enc.embed_dim = 6;
    cfg.trunk_hidden = 10;
    cfg.value_hidden = 6;
    cfg.seed = 0xBEEF;
    net.init(cfg);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.01f * i - 0.3f;
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f;
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES * brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;

    float v1 = 0; Tensor lg1 = Tensor::vec(net.policy_logits());
    net.forward(x, v1, lg1);

    auto blob = net.save();
    SingleHeroNetST net2;
    net2.init(cfg);
    net2.load(blob);
    float v2 = 0; Tensor lg2 = Tensor::vec(net2.policy_logits());
    net2.forward(x, v2, lg2);

    float max_err = std::fabs(v1 - v2);
    for (int i = 0; i < lg1.size(); ++i) max_err = std::max(max_err, std::fabs(lg1[i] - lg2[i]));
    const bool ok = max_err < 1e-6f;
    std::printf("SingleHeroNetST/save-load\t%s\tmax_err=%.4e\n", ok ? "PASS" : "FAIL", max_err);
    if (ok) ++g_pass; else ++g_fail;
}

// ───────────────────── DistributionalValueHead ─────────────────────
static void test_distributional_value() {
    uint64_t seed = 111;
    DistributionalValueHead dv;
    const int E = 5, H = 6, K = 11;
    dv.init(E, H, K, seed);

    Tensor embed = Tensor::vec(E);
    for (int i = 0; i < E; ++i) embed[i] = 0.1f * i - 0.2f;

    Tensor p_target = Tensor::vec(K);
    dv.project_target(0.3f, p_target);
    // Sanity: sums to 1 and non-negative.
    float s = 0.0f; bool neg = false;
    for (int i = 0; i < K; ++i) { s += p_target[i]; if (p_target[i] < -1e-6f) neg = true; }
    const bool proj_ok = std::fabs(s - 1.0f) < 1e-5f && !neg;
    std::printf("DistributionalValueHead/project\t%s\n", proj_ok ? "PASS" : "FAIL");
    if (proj_ok) ++g_pass; else ++g_fail;

    auto loss_fn = [&]() {
        Tensor probs = Tensor::vec(K);
        float v = 0;
        dv.forward(embed, probs, v);
        float l = 0.0f;
        for (int i = 0; i < K; ++i) {
            if (p_target[i] > 0.0f) {
                const float p = probs[i] > 1e-12f ? probs[i] : 1e-12f;
                l -= p_target[i] * std::log(p);
            }
        }
        return l;
    };

    dv.zero_grad();
    Tensor probs = Tensor::vec(K);
    float v = 0;
    dv.forward(embed, probs, v);
    Tensor dEmbed = Tensor::vec(E);
    dv.xent_backward(probs, p_target, dEmbed);

    std::vector<float> num;
    numerical_grad(embed.ptr(), E, loss_fn, num);
    std::vector<float> ana(E);
    for (int i = 0; i < E; ++i) ana[i] = dEmbed[i];
    check_grad(ana, num, "DistributionalValueHead/dEmbed", 2e-2f, 1e-3f);
}

// ───────────────────── OpponentPolicyHead (xent retarget) ─────────────────────
static void test_opponent_head() {
    uint64_t seed = 222;
    OpponentPolicyHead oph;
    const int E = 6;
    oph.init(E, seed);

    Tensor embed = Tensor::vec(E);
    for (int i = 0; i < E; ++i) embed[i] = 0.05f * i - 0.1f;

    const int N = oph.total_logits();
    Tensor logits = Tensor::vec(N);

    // Synthetic opponent target: one-hot per factor.
    Tensor mt = Tensor::vec(FactoredPolicyHead::N_MOVE);    mt.zero(); mt[2] = 1.0f;
    Tensor at = Tensor::vec(FactoredPolicyHead::N_ATTACK);  at.zero(); at[1] = 1.0f;
    Tensor bt = Tensor::vec(FactoredPolicyHead::N_ABILITY); bt.zero(); bt[0] = 1.0f;

    auto loss_fn = [&]() {
        Tensor lg = Tensor::vec(N);
        oph.forward(embed, lg);
        Tensor p = Tensor::vec(N), dL = Tensor::vec(N);
        return factored_xent(lg, mt, at, bt, p, dL, nullptr, nullptr);
    };

    oph.forward(embed, logits);
    Tensor probs = Tensor::vec(N), dLogits = Tensor::vec(N);
    factored_xent(logits, mt, at, bt, probs, dLogits, nullptr, nullptr);

    // Check dLogits via finite-diff on logits.
    std::vector<float> num;
    numerical_grad(logits.ptr(), N, [&]() {
        Tensor p = Tensor::vec(N), dL = Tensor::vec(N);
        return factored_xent(logits, mt, at, bt, p, dL, nullptr, nullptr);
    }, num);
    std::vector<float> ana(N);
    for (int i = 0; i < N; ++i) ana[i] = dLogits[i];
    check_grad(ana, num, "OpponentPolicyHead/dLogits");
    (void)loss_fn;
}

// ───────────────────── EnsembleNet (save/load round-trip) ─────────────────────
static void test_ensemble_roundtrip() {
    EnsembleNet ens;
    SingleHeroNet::Config base;
    base.enc.hidden = 6;
    base.enc.embed_dim = 6;
    base.trunk_hidden = 8;
    base.value_hidden = 6;
    base.seed = 0xA5A5ULL;
    ens.init(3, base);

    Tensor x = Tensor::vec(brogameagent::observation::TOTAL);
    for (int i = 0; i < x.size(); ++i) x[i] = 0.01f * i - 0.2f;
    x[brogameagent::observation::SELF_FEATURES + 0] = 1.0f;
    const int off_a = brogameagent::observation::SELF_FEATURES +
                      brogameagent::observation::K_ENEMIES * brogameagent::observation::ENEMY_FEATURES;
    x[off_a + 0] = 1.0f;

    float vm1 = 0, vs1 = 0;
    Tensor lg1 = Tensor::vec(ens.member(0).policy_logits());
    ens.forward_mean(x, vm1, vs1, lg1);

    auto blob = ens.save();
    EnsembleNet ens2;
    ens2.init(3, base);
    ens2.load(blob);
    float vm2 = 0, vs2 = 0;
    Tensor lg2 = Tensor::vec(ens2.member(0).policy_logits());
    ens2.forward_mean(x, vm2, vs2, lg2);

    float max_err = std::fabs(vm1 - vm2) + std::fabs(vs1 - vs2);
    for (int i = 0; i < lg1.size(); ++i) max_err = std::max(max_err, std::fabs(lg1[i] - lg2[i]));
    const bool ok = max_err < 1e-6f && ens2.num_members() == 3;
    std::printf("EnsembleNet/save-load\t%s\tmax_err=%.4e\tmembers=%d\n",
                ok ? "PASS" : "FAIL", max_err, ens2.num_members());
    if (ok) ++g_pass; else ++g_fail;
}

// ───────────────────── ForwardModelHead ─────────────────────
static void test_forward_model() {
    uint64_t seed = 444;
    ForwardModelHead fm;
    const int E = 6, H = 8;
    fm.init(E, H, seed);

    Tensor embed = Tensor::vec(E);
    for (int i = 0; i < E; ++i) embed[i] = 0.1f * i - 0.2f;
    Tensor action = Tensor::vec(ForwardModelHead::ACTION_DIM);
    build_action_onehot(3, 2, 1, action);

    Tensor target = Tensor::vec(E);
    for (int i = 0; i < E; ++i) target[i] = 0.05f * i - 0.1f;

    auto loss_fn = [&]() {
        Tensor pred = Tensor::vec(E);
        fm.forward(embed, action, pred);
        Tensor dp = Tensor::vec(E);
        return spr_loss(pred, target, dp);
    };

    fm.zero_grad();
    Tensor pred = Tensor::vec(E);
    fm.forward(embed, action, pred);
    Tensor dPred = Tensor::vec(E);
    spr_loss(pred, target, dPred);
    Tensor dEmbed = Tensor::vec(E);
    fm.backward(dPred, dEmbed);

    std::vector<float> num;
    numerical_grad(embed.ptr(), E, loss_fn, num);
    std::vector<float> ana(E);
    for (int i = 0; i < E; ++i) ana[i] = dEmbed[i];
    check_grad(ana, num, "ForwardModelHead/dEmbed");
}

// ───────────────────── InfoNCE ─────────────────────
static void test_infonce() {
    using brogameagent::learn::infonce_loss;
    const int B = 4, D = 8;
    std::vector<Tensor> anchors(B, Tensor::vec(D));
    std::vector<Tensor> positives(B, Tensor::vec(D));
    for (int i = 0; i < B; ++i) {
        for (int k = 0; k < D; ++k) {
            anchors[i][k]   = 0.05f * (i + 1) + 0.02f * k - 0.1f;
            positives[i][k] = 0.04f * (i + 1) - 0.03f * k + 0.1f;
        }
    }

    auto loss_fn = [&]() {
        std::vector<Tensor> da, dp;
        return infonce_loss(anchors, positives, da, dp, 0.1f);
    };

    std::vector<Tensor> dA, dP;
    /*float loss =*/ infonce_loss(anchors, positives, dA, dP, 0.1f);

    // Check dAnchors[0] via finite diff.
    std::vector<float> num;
    numerical_grad(anchors[0].ptr(), D, loss_fn, num, 1e-3f);
    std::vector<float> ana(D);
    for (int k = 0; k < D; ++k) ana[k] = dA[0][k];
    check_grad(ana, num, "InfoNCE/dAnchor[0]", 1e-2f, 1e-3f);

    // And dPositives[1].
    std::vector<float> num2;
    numerical_grad(positives[1].ptr(), D, loss_fn, num2, 1e-3f);
    std::vector<float> ana2(D);
    for (int k = 0; k < D; ++k) ana2[k] = dP[1][k];
    check_grad(ana2, num2, "InfoNCE/dPositive[1]", 1e-2f, 1e-3f);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0) g_verbose = true;
    }
    std::printf("test\tstatus\tdetail\n");
    test_linear();
    test_relu();
    test_tanh();
    test_softmax_xent();
    test_encoder();
    test_decoder();
    test_autoencoder();
    test_sigmoid();
    test_layernorm();
    test_embedding();
    test_attention();
    test_gru();
    test_full_net();
    test_policy_value_net();
    test_set_transformer();
    test_single_hero_net_st();
    test_distributional_value();
    test_opponent_head();
    test_ensemble_roundtrip();
    test_forward_model();
    test_infonce();
    std::printf("# summary\tpass=%d\tfail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
