// MCTS + BatchedInferenceServer integration test.
//
// Wires GenericMcts up to two interchangeable IInferenceBackend impls
// (DirectBackend = synchronous net.forward, ServerBackend = submit to
// BatchedInferenceServer) and asserts:
//   1. Search-result equivalence: identical root visit distributions and
//      best action between backends on the same problem with the same
//      number of rollouts (deterministic eval — the server is just a
//      batching layer).
//   2. A smoke benchmark printing rollouts/sec under each backend.

#include "parity_helpers.h"

#include <brogameagent/generic_mcts.h>
#include <brogameagent/learn/inference_backend.h>
#include <brogameagent/learn/inference_server.h>
#include <brogameagent/nn/policy_value_net.h>

#include <any>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::PolicyValueNet;
using brogameagent::learn::DirectBackend;
using brogameagent::learn::ServerBackend;
using brogameagent::learn::BatchedInferenceServer;
using brogameagent::learn::IInferenceBackend;
using brogameagent::learn::EvalResult;
namespace mcts = brogameagent::mcts;

namespace {

// ─── Corridor env (small deterministic 1D grid) ────────────────────────────

struct CorridorState {
    int pos = 0;
    int goal = 10;
    bool done = false;
};

constexpr int kInDim = 4;        // [behind, here, ahead, dist_norm]
constexpr int kNumActions = 3;   // 0=left, 1=right, 2=stay

std::vector<float> build_obs(const CorridorState& st) {
    std::vector<float> v(kInDim, 0.0f);
    v[0] = (st.pos > 0) ? 0.0f : 1.0f;        // behind wall indicator
    v[1] = 0.0f;                               // here always empty
    v[2] = (st.pos + 1 >= st.goal) ? 1.0f : 0.0f;
    v[3] = static_cast<float>(st.goal - st.pos) /
           static_cast<float>(st.goal);
    return v;
}

mcts::GenericEnv make_env(CorridorState& st) {
    mcts::GenericEnv env;
    env.num_actions = kNumActions;
    env.snapshot_fn = [&]() -> std::any { return st; };
    env.restore_fn  = [&](const std::any& s) { st = std::any_cast<CorridorState>(s); };
    env.observe_fn  = [&]() -> std::vector<float> { return build_obs(st); };
    env.legal_actions_fn = [&]() -> std::vector<int> {
        if (st.done) return {};
        std::vector<int> a;
        if (st.pos > 0) a.push_back(0);
        a.push_back(1);
        a.push_back(2);
        return a;
    };
    env.step_fn = [&](int action) -> mcts::GenericStepResult {
        mcts::GenericStepResult r;
        if (st.done) { r.done = true; return r; }
        if      (action == 0 && st.pos > 0) st.pos--;
        else if (action == 1)               st.pos++;
        if (st.pos >= st.goal) { st.done = true; r.reward = 1.0f; r.done = true; }
        return r;
    };
    return env;
}

// Build prior/value functions that wrap an IInferenceBackend. The MCTS
// expansion code calls prior_fn(obs, legal) for the prior and value_fn(obs)
// for the leaf value separately; we evaluate once and serve both via tiny
// per-thread caches (simplest: just evaluate twice — DirectBackend is fast
// and ServerBackend's batching coalesces both calls into the same batch on
// the same iteration anyway). For the equivalence check we want the two
// backends to receive the same calls in the same order.
mcts::GenericPriorFn make_prior_fn(IInferenceBackend* backend) {
    return [backend](const std::vector<float>& obs,
                     const std::vector<int>& legal) -> std::vector<float> {
        const auto r = backend->evaluate(obs);
        const int A = backend->num_actions();
        std::vector<float> probs(static_cast<size_t>(A), 0.0f);
        // Masked softmax over legal entries.
        std::vector<uint8_t> mask(static_cast<size_t>(A), 0);
        for (int a : legal) if (a >= 0 && a < A) mask[a] = 1;
        float m = -1e30f;
        for (int a = 0; a < A; ++a) if (mask[a] && r.logits[a] > m) m = r.logits[a];
        float s = 0.0f;
        for (int a = 0; a < A; ++a) {
            if (!mask[a]) { probs[a] = 0.0f; continue; }
            probs[a] = std::exp(r.logits[a] - m);
            s += probs[a];
        }
        if (s > 0.0f) for (int a = 0; a < A; ++a) probs[a] /= s;
        return probs;
    };
}

mcts::GenericValueFn make_value_fn(IInferenceBackend* backend) {
    return [backend](const std::vector<float>& obs) -> float {
        return backend->evaluate(obs).value;
    };
}

// Run search with `iterations` rollouts and return (root visits, best action).
struct SearchOut {
    std::vector<float> visits;
    int best = -1;
};

SearchOut run_search(IInferenceBackend* backend, int iterations,
                     uint64_t seed, int start_pos = 0) {
    CorridorState st;
    st.pos = start_pos;
    auto env = make_env(st);

    mcts::GenericMcts m(env);
    mcts::GenericMctsConfig cfg;
    cfg.iterations    = iterations;
    cfg.c_puct        = 1.5f;
    cfg.gamma         = 0.99f;
    cfg.rollout_depth = 0;          // value_fn fully replaces rollouts
    cfg.seed          = seed;
    m.set_config(cfg);
    m.set_prior_fn(make_prior_fn(backend));
    m.set_value_fn(make_value_fn(backend));

    SearchOut out;
    out.best = m.search();
    out.visits = m.root_visits();
    return out;
}

PolicyValueNet make_net() {
    PolicyValueNet::Config cfg;
    cfg.in_dim       = kInDim;
    cfg.hidden       = {16, 16};
    cfg.value_hidden = 8;
    cfg.num_actions  = kNumActions;
    cfg.seed         = 0xCAFEDEEDull;
    PolicyValueNet net;
    net.init(cfg);
    net.to(Device::GPU);
    return net;
}

} // namespace

// ─── Equivalence: direct vs server, same N rollouts ───────────────────────

BGA_PARITY_TEST(mcts_server_search_equivalence) {
    PolicyValueNet net = make_net();
    DirectBackend direct(&net);

    BatchedInferenceServer::Config scfg;
    scfg.max_batch_size  = 8;
    scfg.max_wait_micros = 200;
    BatchedInferenceServer server(&net, scfg);
    ServerBackend server_be(&server, net.num_actions(), net.in_dim());

    constexpr int kIter = 64;

    // Two independent searchers — same env state, same seed, same iter count.
    auto a = run_search(&direct,   kIter, 0xDEADBEEFull, /*start_pos=*/3);
    auto b = run_search(&server_be, kIter, 0xDEADBEEFull, /*start_pos=*/3);

    BGA_CHECK(a.visits.size() == b.visits.size());
    BGA_CHECK(a.best == b.best);
    // Visit counts: server may differ slightly if PUCT ties break under tiny
    // floating-point drift, but on this deterministic problem the two should
    // match exactly. Allow a small per-action delta as a safety margin.
    for (size_t i = 0; i < a.visits.size(); ++i) {
        const float d = std::fabs(a.visits[i] - b.visits[i]);
        if (d > 1e-3f) {
            std::fprintf(stderr,
                "visit dist mismatch at action %zu: direct=%.5f server=%.5f\n",
                i, a.visits[i], b.visits[i]);
        }
        BGA_CHECK(d <= 1e-3f);
    }
}

// ─── Bench: rollouts/sec under each backend ───────────────────────────────

BGA_PARITY_TEST(mcts_server_bench) {
    PolicyValueNet net = make_net();
    DirectBackend direct(&net);

    BatchedInferenceServer::Config scfg;
    scfg.max_batch_size  = 32;
    scfg.max_wait_micros = 50;
    BatchedInferenceServer server(&net, scfg);
    ServerBackend server_be(&server, net.num_actions(), net.in_dim());

    // Bench parameters.
    constexpr int kIter        = 256;        // rollouts per search
    constexpr int kSearches    = 64;         // number of independent searches per backend
    constexpr int kSrvWorkers  = 32;         // concurrent searcher threads for server bench

    // Warm-up.
    (void)run_search(&direct,   16, 0x1ULL);
    (void)run_search(&server_be, 16, 0x1ULL);

    // Direct (synchronous, single-threaded).
    auto t0 = std::chrono::steady_clock::now();
    int direct_actions = 0;
    for (int i = 0; i < kSearches; ++i) {
        auto r = run_search(&direct, kIter, 0x100ULL + i,
                            /*start_pos=*/i % 9);
        direct_actions += (r.best >= 0);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double direct_secs =
        std::chrono::duration<double>(t1 - t0).count();
    const double direct_rps = (kSearches * kIter) / direct_secs;

    // Server-backed: launch kSrvWorkers searchers concurrently. The server
    // batches their concurrent eval requests into single GPU forwards.
    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::future<int>> futures;
    futures.reserve(kSrvWorkers);
    const int per_worker = kSearches / kSrvWorkers;
    for (int w = 0; w < kSrvWorkers; ++w) {
        futures.push_back(std::async(std::launch::async, [&, w]() {
            int n = 0;
            for (int i = 0; i < per_worker; ++i) {
                auto r = run_search(&server_be, kIter,
                                    0x100ULL + w * per_worker + i,
                                    /*start_pos=*/(w + i) % 9);
                n += (r.best >= 0);
            }
            return n;
        }));
    }
    int server_actions = 0;
    for (auto& f : futures) server_actions += f.get();
    auto t3 = std::chrono::steady_clock::now();
    const double server_secs =
        std::chrono::duration<double>(t3 - t2).count();
    const double server_rps = (kSrvWorkers * per_worker * kIter) / server_secs;

    std::printf("\nMCTS bench (%d searches × %d rollouts each):\n", kSearches, kIter);
    std::printf("  direct  (1 thread)        : %.3f s   %.0f rollouts/sec\n",
                direct_secs, direct_rps);
    std::printf("  server  (%d worker threads): %.3f s   %.0f rollouts/sec  (%d batches)\n",
                kSrvWorkers, server_secs, server_rps, server.batches_run());
    std::fflush(stdout);

    // Don't fail on a specific ratio — just sanity that both made progress.
    BGA_CHECK(direct_actions > 0);
    BGA_CHECK(server_actions > 0);
    BGA_CHECK(server.batches_run() > 0);
}

int main() { return run_all("mcts + inference server integration"); }
