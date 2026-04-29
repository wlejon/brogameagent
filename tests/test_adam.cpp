// CPU correctness tests for the Adam optimizer.
//
// Two checks:
//   1. Hand-computed reference for the first 3 Adam steps on a 1-element
//      problem with a fixed gradient. This pins down the bias-correction
//      math against a Python-style reference computation done inline.
//   2. End-to-end convergence: minimize f(w) = (w - 3)^2 from w = 0 using
//      Adam(lr=0.1) for 100 steps; check w is within 0.1 of 3.0.
//
// Also exercises Linear::adam_step over a larger param space to make sure
// the per-Linear plumbing wires through correctly.

#include <brogameagent/nn/circuits.h>
#include <brogameagent/nn/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using brogameagent::nn::Tensor;
using brogameagent::nn::Linear;
using brogameagent::nn::adam_step_cpu;

namespace {

int passed_count = 0;
int total_count = 0;

void check(bool cond, const char* msg, const char* file, int line) {
    ++total_count;
    if (cond) {
        ++passed_count;
    } else {
        std::printf("    FAIL %s:%d: %s\n", file, line, msg);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) do {                                          \
    const float _a = static_cast<float>(a);                                 \
    const float _b = static_cast<float>(b);                                 \
    const float _t = static_cast<float>(tol);                               \
    const bool _ok = std::fabs(_a - _b) <= _t;                              \
    ++total_count;                                                          \
    if (_ok) ++passed_count;                                                \
    else std::printf("    FAIL %s:%d: %s (got %g, want %g, tol %g)\n",      \
                     __FILE__, __LINE__, #a " ≈ " #b, _a, _b, _t);          \
} while (0)

// Reference: hand-computed three Adam steps on a scalar with constant grad.
//   Init: w=1.0, m=0.0, v=0.0
//   Constant g = 0.5 every step.
//   beta1=0.9, beta2=0.999, eps=1e-8, lr=0.1
//
// step=1:
//   m1 = 0.9*0 + 0.1*0.5 = 0.05
//   v1 = 0.999*0 + 0.001*0.25 = 0.00025
//   m_hat = 0.05 / (1 - 0.9) = 0.5
//   v_hat = 0.00025 / (1 - 0.999) = 0.25
//   w1 = 1.0 - 0.1 * 0.5 / (sqrt(0.25) + 1e-8) ≈ 1.0 - 0.1 = 0.9
// step=2:
//   m2 = 0.9*0.05 + 0.1*0.5 = 0.095
//   v2 = 0.999*0.00025 + 0.001*0.25 = 0.00049975
//   bc1 = 1 - 0.9^2 = 0.19; m_hat = 0.5
//   bc2 = 1 - 0.999^2 = 0.001999; v_hat ≈ 0.25
//   w2 = 0.9 - 0.1 * 0.5 / (sqrt(0.25) + 1e-8) ≈ 0.8
// step=3:
//   m3 = 0.9*0.095 + 0.1*0.5 = 0.1355
//   v3 = 0.999*0.00049975 + 0.001*0.25 = 0.000749250...
//   bc1 = 1 - 0.9^3 = 0.271; m_hat ≈ 0.5
//   bc2 = 1 - 0.999^3 ≈ 0.002997; v_hat ≈ 0.25
//   w3 ≈ 0.7
//
// In short: with a constant gradient, Adam moves at almost exactly lr per
// step early on (bias correction kills the slow start that vanilla EMA would
// produce). Use this as the spec-pin reference.
void test_adam_bias_correction_reference() {
    Tensor w(1, 1), g(1, 1), m(1, 1), v(1, 1);
    w.ptr()[0] = 1.0f;
    g.ptr()[0] = 0.5f;
    m.zero(); v.zero();

    const float lr = 0.1f;
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1e-8f;

    adam_step_cpu(w, g, m, v, lr, b1, b2, eps, 1);
    CHECK_NEAR(w.ptr()[0], 0.9f, 1e-6f);
    CHECK_NEAR(m.ptr()[0], 0.05f, 1e-6f);
    CHECK_NEAR(v.ptr()[0], 0.00025f, 1e-7f);

    adam_step_cpu(w, g, m, v, lr, b1, b2, eps, 2);
    CHECK_NEAR(w.ptr()[0], 0.8f, 1e-4f);
    CHECK_NEAR(m.ptr()[0], 0.095f, 1e-6f);
    CHECK_NEAR(v.ptr()[0], 0.00049975f, 1e-7f);

    adam_step_cpu(w, g, m, v, lr, b1, b2, eps, 3);
    CHECK_NEAR(w.ptr()[0], 0.7f, 1e-3f);
    CHECK_NEAR(m.ptr()[0], 0.1355f, 1e-6f);
}

// Convergence: minimize (w - 3)^2 starting at w=0, gradient = 2*(w - 3).
void test_adam_convergence_quadratic() {
    Tensor w(1, 1), g(1, 1), m(1, 1), v(1, 1);
    w.ptr()[0] = 0.0f;
    m.zero(); v.zero();

    const float lr = 0.1f;
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1e-8f;

    for (int step = 1; step <= 100; ++step) {
        g.ptr()[0] = 2.0f * (w.ptr()[0] - 3.0f);
        adam_step_cpu(w, g, m, v, lr, b1, b2, eps, step);
    }
    const float final_w = w.ptr()[0];
    std::printf("    convergence: w=%.4f after 100 steps (target 3.0)\n", final_w);
    CHECK(std::fabs(final_w - 3.0f) < 0.1f);
}

// Linear::adam_step exercise: drive a 4×3 Linear toward a fixed target with
// a constant input + MSE-style gradient and verify the loss shrinks.
void test_linear_adam_step() {
    uint64_t rng = 0xC0FFEE12345ULL;
    Linear L;
    L.init(3, 4, rng);

    Tensor x(3, 1), y(4, 1), target(4, 1), dY(4, 1), dX(3, 1);
    for (int i = 0; i < 3; ++i) x.ptr()[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 4; ++i) target.ptr()[i] = 0.5f * static_cast<float>(i);

    auto compute_loss = [&]() {
        L.forward(x, y);
        float loss = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float d = y.ptr()[i] - target.ptr()[i];
            loss += 0.5f * d * d;
            dY.ptr()[i] = d;
        }
        return loss;
    };

    const float loss0 = compute_loss();
    for (int step = 1; step <= 200; ++step) {
        L.zero_grad();
        compute_loss();
        L.backward(dY, dX);
        L.adam_step(0.05f, 0.9f, 0.999f, 1e-8f, step);
    }
    const float lossN = compute_loss();
    std::printf("    Linear loss: %.4f -> %.4f after 200 Adam steps\n", loss0, lossN);
    CHECK(lossN < 1e-3f);
    CHECK(lossN < loss0);
}

} // namespace

int main() {
    std::printf("Adam optimizer (CPU) tests\n");
    std::printf("==========================\n");

    test_adam_bias_correction_reference();
    test_adam_convergence_quadratic();
    test_linear_adam_step();

    std::printf("\n%d/%d checks passed\n", passed_count, total_count);
    return (passed_count == total_count) ? 0 : 1;
}
