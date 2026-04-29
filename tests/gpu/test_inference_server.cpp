// BatchedInferenceServer functional, stress, and (smoke) bench tests.

#include "parity_helpers.h"

#include <brogameagent/learn/inference_server.h>
#include <brogameagent/nn/policy_value_net.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

using namespace bga_parity;
using brogameagent::nn::Tensor;
using brogameagent::nn::Device;
using brogameagent::nn::PolicyValueNet;
using brogameagent::nn::gpu::GpuTensor;
using brogameagent::learn::BatchedInferenceServer;

namespace {

PolicyValueNet make_net() {
    PolicyValueNet::Config cfg;
    cfg.in_dim       = 24;
    cfg.hidden       = {32, 32};
    cfg.value_hidden = 16;
    cfg.num_actions  = 7;
    cfg.seed         = 0xFEEDBEEFull;
    PolicyValueNet net;
    net.init(cfg);
    net.to(Device::GPU);
    return net;
}

std::vector<float> random_obs(int n, SplitMix64& rng) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = rng.next_unit();
    return v;
}

// Run a direct (single-sample) GPU forward and return result for parity.
BatchedInferenceServer::EvalResult direct_forward(PolicyValueNet& net,
                                                  const std::vector<float>& obs) {
    const int in_dim = net.in_dim();
    const int A = net.num_actions();
    Tensor x(in_dim, 1);
    for (int i = 0; i < in_dim; ++i) x.data[i] = obs[i];
    GpuTensor gx, glogits;
    upload(x, gx);
    net.forward(gx, glogits);
    Tensor h_logits = download_to_host(glogits);
    Tensor h_value  = download_to_host(net.value_gpu());
    BatchedInferenceServer::EvalResult r;
    r.logits.assign(A, 0.0f);
    for (int i = 0; i < A; ++i) r.logits[i] = h_logits.data[i];
    r.value = h_value.data[0];
    return r;
}

} // namespace

// ─── Functional: 64 threads, parity vs. direct forward ─────────────────────

BGA_PARITY_TEST(server_functional_64_threads) {
    PolicyValueNet net = make_net();
    BatchedInferenceServer::Config cfg;
    cfg.max_batch_size = 32;
    cfg.max_wait_micros = 2000;
    BatchedInferenceServer server(&net, cfg);

    constexpr int kThreads = 64;

    // Generate one obs per thread up front so we can also compute the direct
    // reference deterministically.
    SplitMix64 rng(0xABCDEFull);
    std::vector<std::vector<float>> all_obs(kThreads);
    for (int i = 0; i < kThreads; ++i) all_obs[i] = random_obs(net.in_dim(), rng);

    std::vector<BatchedInferenceServer::EvalResult> results(kThreads);
    std::vector<std::thread> ts;
    ts.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        ts.emplace_back([&, i] {
            results[i] = server.evaluate(all_obs[i]);
        });
    }
    for (auto& t : ts) t.join();

    // Parity vs direct forward (each obs run individually after the server
    // is destructed-or-quiescent — server is still alive but idle here).
    for (int i = 0; i < kThreads; ++i) {
        const auto& r = results[i];
        BGA_CHECK(static_cast<int>(r.logits.size()) == net.num_actions());
        for (float v : r.logits) BGA_CHECK(std::isfinite(v));
        BGA_CHECK(std::isfinite(r.value));

        auto ref = direct_forward(net, all_obs[i]);
        for (int j = 0; j < net.num_actions(); ++j) {
            const float a = ref.logits[j];
            const float b = r.logits[j];
            const float tol = 1e-4f + 1e-3f * std::fabs(a);
            BGA_CHECK(std::fabs(a - b) < tol);
        }
        BGA_CHECK(std::fabs(ref.value - r.value) < 1e-4f);
    }

    BGA_CHECK(server.batches_run() >= 1);
}

// ─── Stress: 1000 sequential calls ─────────────────────────────────────────

BGA_PARITY_TEST(server_stress_1000_sequential) {
    PolicyValueNet net = make_net();
    BatchedInferenceServer::Config cfg;
    cfg.max_batch_size = 64;
    cfg.max_wait_micros = 100;
    BatchedInferenceServer server(&net, cfg);

    SplitMix64 rng(0x12340000ull);
    for (int i = 0; i < 1000; ++i) {
        auto obs = random_obs(net.in_dim(), rng);
        auto r = server.evaluate(obs);
        BGA_CHECK(static_cast<int>(r.logits.size()) == net.num_actions());
        BGA_CHECK(std::isfinite(r.value));
        for (float v : r.logits) BGA_CHECK(std::isfinite(v));
    }
}

// ─── Smoke bench: 256 evals, server vs. direct sequential ──────────────────

BGA_PARITY_TEST(server_bench_smoke) {
    PolicyValueNet net = make_net();

    constexpr int kCalls = 256;
    SplitMix64 rng(0xBEEF55ull);
    std::vector<std::vector<float>> obs;
    obs.reserve(kCalls);
    for (int i = 0; i < kCalls; ++i) obs.push_back(random_obs(net.in_dim(), rng));

    // Direct sequential.
    auto t0 = std::chrono::steady_clock::now();
    for (const auto& o : obs) {
        auto r = direct_forward(net, o);
        (void)r;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double direct_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Server: 64 threads concurrently issue calls.
    BatchedInferenceServer::Config cfg;
    cfg.max_batch_size = 32;
    cfg.max_wait_micros = 1000;
    BatchedInferenceServer server(&net, cfg);

    constexpr int kThreads = 64;
    const int per_thread = kCalls / kThreads;

    auto s0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t] {
            for (int j = 0; j < per_thread; ++j) {
                auto r = server.evaluate(obs[t * per_thread + j]);
                (void)r;
            }
        });
    }
    for (auto& th : ts) th.join();
    auto s1 = std::chrono::steady_clock::now();
    const double server_ms =
        std::chrono::duration<double, std::milli>(s1 - s0).count();

    std::printf("    [bench] %d evals  direct=%.2f ms  server=%.2f ms  "
                "speedup=%.2fx  batches=%d\n",
                kCalls, direct_ms, server_ms, direct_ms / server_ms,
                server.batches_run());
    // Don't fail on a specific ratio — print only.
}

int main() { return run_all("BatchedInferenceServer"); }
