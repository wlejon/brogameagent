// MCTS + BatchedInferenceServer integration test, TX-net edition.
//
// Same shape as test_mcts_server.cpp but the underlying net is a
// SingleHeroNetTX served through the (now net-agnostic) BatchedInferenceServer
// + ServerBackend. Verifies that:
//   1. DirectBatchedNetBackend and ServerBackend produce identical search
//      results when the underlying TX net is the same.
//   2. A smoke bench prints per-backend rollouts/sec.

#include "parity_helpers.h"

#include <brogameagent/generic_mcts.h>
#include <brogameagent/learn/inference_backend.h>
#include <brogameagent/learn/inference_server.h>
#include <brogameagent/nn/net_tx.h>
#include <brogameagent/observation.h>

#include <any>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

using namespace bga_parity;
using brogameagent::nn::Device;
using brogameagent::nn::SingleHeroNetTX;
using brogameagent::learn::DirectBatchedNetBackend;
using brogameagent::learn::ServerBackend;
using brogameagent::learn::BatchedInferenceServer;
using brogameagent::learn::IInferenceBackend;
using brogameagent::learn::EvalResult;
namespace mcts = brogameagent::mcts;
namespace obs  = brogameagent::observation;

namespace {

// ─── Corridor env (small deterministic 1D grid) ────────────────────────────
//
// Same env as test_mcts_server.cpp but the observation is padded out to TX's
// observation::TOTAL layout so the TX forward path is fed a valid input.

struct CorridorState {
    int pos = 0;
    int goal = 10;
    bool done = false;
};

constexpr int kNumActions = 3;   // 0=left, 1=right, 2=stay

std::vector<float> build_obs_tx(const CorridorState& st) {
    // Fill a (TOTAL,) vector. Self block carries the corridor-shaped
    // features in its leading slots; entity slots are all flagged invalid
    // (first feature = 0) so masked-mean pool returns zero pooled vectors.
    std::vector<float> v(obs::TOTAL, 0.0f);
    v[0] = (st.pos > 0) ? 0.0f : 1.0f;        // behind wall indicator
    v[1] = 0.0f;
    v[2] = (st.pos + 1 >= st.goal) ? 1.0f : 0.0f;
    v[3] = static_cast<float>(st.goal - st.pos) /
           static_cast<float>(st.goal);
    // Validity flags for enemy/ally slots default to 0 from the assign().
    return v;
}

mcts::GenericEnv make_env(CorridorState& st) {
    mcts::GenericEnv env;
    env.num_actions = kNumActions;
    env.snapshot_fn = [&]() -> std::any { return st; };
    env.restore_fn  = [&](const std::any& s) { st = std::any_cast<CorridorState>(s); };
    env.observe_fn  = [&]() -> std::vector<float> { return build_obs_tx(st); };
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

// TX policy head emits a wider logit vector (move + atk + abil); we project
// it down to the corridor's 3 actions by taking the first kNumActions logits.
// This is consistent across both backends so equivalence still holds.
mcts::GenericPriorFn make_prior_fn(IInferenceBackend* backend) {
    return [backend](const std::vector<float>& obs_v,
                     const std::vector<int>& legal) -> std::vector<float> {
        const auto r = backend->evaluate(obs_v);
        std::vector<float> probs(kNumActions, 0.0f);
        std::vector<uint8_t> mask(kNumActions, 0);
        for (int a : legal) if (a >= 0 && a < kNumActions) mask[a] = 1;
        float m = -1e30f;
        for (int a = 0; a < kNumActions; ++a)
            if (mask[a] && r.logits[a] > m) m = r.logits[a];
        float s = 0.0f;
        for (int a = 0; a < kNumActions; ++a) {
            if (!mask[a]) { probs[a] = 0.0f; continue; }
            probs[a] = std::exp(r.logits[a] - m);
            s += probs[a];
        }
        if (s > 0.0f) for (int a = 0; a < kNumActions; ++a) probs[a] /= s;
        return probs;
    };
}

mcts::GenericValueFn make_value_fn(IInferenceBackend* backend) {
    return [backend](const std::vector<float>& obs_v) -> float {
        return backend->evaluate(obs_v).value;
    };
}

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
    cfg.rollout_depth = 0;
    cfg.seed          = seed;
    m.set_config(cfg);
    m.set_prior_fn(make_prior_fn(backend));
    m.set_value_fn(make_value_fn(backend));

    SearchOut out;
    out.best = m.search();
    out.visits = m.root_visits();
    return out;
}

SingleHeroNetTX make_net() {
    SingleHeroNetTX::Config cfg;
    cfg.d_model      = 16;
    cfg.d_ff         = 32;
    cfg.num_heads    = 2;
    cfg.num_blocks   = 1;
    cfg.self_hidden  = 16;
    cfg.trunk_hidden = 32;
    cfg.value_hidden = 16;
    cfg.seed         = 0xC0FFEE99ull;
    SingleHeroNetTX net;
    net.init(cfg);
    net.to(Device::GPU);
    return net;
}

} // namespace

// ─── Equivalence: direct vs server, same N rollouts, TX net ───────────────

BGA_PARITY_TEST(mcts_server_tx_search_equivalence) {
    SingleHeroNetTX net = make_net();
    DirectBatchedNetBackend direct(&net);

    BatchedInferenceServer::Config scfg;
    scfg.max_batch_size  = 8;
    scfg.max_wait_micros = 200;
    BatchedInferenceServer server(static_cast<brogameagent::learn::BatchedNet*>(&net), scfg);
    ServerBackend server_be(&server, &net);

    constexpr int kIter = 32;

    auto a = run_search(&direct,    kIter, 0xDEADBEEFull, /*start_pos=*/3);
    auto b = run_search(&server_be, kIter, 0xDEADBEEFull, /*start_pos=*/3);

    BGA_CHECK(a.visits.size() == b.visits.size());
    BGA_CHECK(a.best == b.best);
    for (size_t i = 0; i < a.visits.size(); ++i) {
        const float d = std::fabs(a.visits[i] - b.visits[i]);
        if (d > 1e-3f) {
            std::fprintf(stderr,
                "tx visit dist mismatch at action %zu: direct=%.5f server=%.5f\n",
                i, a.visits[i], b.visits[i]);
        }
        BGA_CHECK(d <= 1e-3f);
    }
}

// ─── Smoke bench: rollouts/sec under each backend, TX net ─────────────────

BGA_PARITY_TEST(mcts_server_tx_bench) {
    SingleHeroNetTX net = make_net();
    DirectBatchedNetBackend direct(&net);

    BatchedInferenceServer::Config scfg;
    scfg.max_batch_size  = 16;
    scfg.max_wait_micros = 100;
    BatchedInferenceServer server(static_cast<brogameagent::learn::BatchedNet*>(&net), scfg);
    ServerBackend server_be(&server, &net);

    constexpr int kIter      = 32;
    constexpr int kSearches  = 8;
    constexpr int kWorkers   = 8;

    (void)run_search(&direct,    8, 0x1ULL);
    (void)run_search(&server_be, 8, 0x1ULL);

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

    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::future<int>> futures;
    futures.reserve(kWorkers);
    const int per_worker = kSearches / kWorkers;
    for (int w = 0; w < kWorkers; ++w) {
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
    const double server_rps = (kWorkers * per_worker * kIter) / server_secs;

    std::printf("\nMCTS bench (TX net, %d searches × %d rollouts):\n",
                kSearches, kIter);
    std::printf("  direct  (1 thread)          : %.3f s   %.0f rollouts/sec\n",
                direct_secs, direct_rps);
    std::printf("  server  (%d worker threads)  : %.3f s   %.0f rollouts/sec  (%d batches)\n",
                kWorkers, server_secs, server_rps, server.batches_run());
    std::fflush(stdout);

    BGA_CHECK(direct_actions > 0);
    BGA_CHECK(server_actions > 0);
    BGA_CHECK(server.batches_run() > 0);
}

int main() { return run_all("mcts + inference server (TX net)"); }
