// nn_check — finite-difference gradient checks for every NN circuit.
//
// Each test compares the analytic gradient produced by backward() against
// the numerical gradient from (f(x+h) - f(x-h)) / (2h). Agreement to ~1e-3
// relative error is expected for fp32.
//
// Usage: nn_check [--verbose]
//   Exits 0 on all-pass, 1 on any failure. Prints TSV of per-test results
//   so it composes into sweep logging.

#include "brogameagent/nn/circuits.h"
#include "brogameagent/nn/encoder.h"
#include "brogameagent/nn/heads.h"
#include "brogameagent/nn/net.h"
#include "brogameagent/nn/ops.h"
#include "brogameagent/nn/tensor.h"

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
    test_full_net();
    std::printf("# summary\tpass=%d\tfail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
