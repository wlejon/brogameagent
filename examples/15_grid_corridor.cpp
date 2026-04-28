// 15_grid_corridor — end-to-end demo of the grid-world training kit.
//
// A trivial 1D corridor env: agent starts at pos=0, goal at pos=10.
// Actions: 0=left, 1=right, 2=stay. Reward +1 on reaching the goal.
// We pair GenericMcts (using a learned PolicyValueNet as both prior and
// value head via WeightsHandle) with the GridTrainer harness, fed by
// MCTS-driven self-play. After enough episodes, the policy reliably
// chooses "right" at every cell — and the BestCrop pool captures the
// best run for replay.

#include "brogameagent/generic_mcts.h"
#include "brogameagent/grid/harness.h"
#include "brogameagent/grid/obs_window.h"
#include "brogameagent/grid/bc_ingest.h"
#include "brogameagent/learn/generic_replay_buffer.h"
#include "brogameagent/nn/ops.h"
#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/tensor.h"

#include <any>
#include <cstdio>
#include <random>
#include <vector>

using namespace brogameagent;

namespace {

struct CorridorState {
    int  pos  = 0;
    int  goal = 10;
    bool done = false;
};

// 3-cell egocentric observation: {behind, here, ahead} as solid/empty
// indicators (corridor is always empty so 0 everywhere; we keep the
// ObsWindow shape so the example exercises the rasterizer), plus a
// 1-float self block holding normalized goal-relative distance.
constexpr int kInDim = 3 + 1;

grid::ObsWindow make_obs_window() {
    grid::ObsWindowSpec s;
    s.cols_behind   = 1;
    s.cols_ahead    = 1;
    s.rows_up       = 0;
    s.rows_down     = 0;
    s.tile_channels = 1;
    s.self_block_size = 1;
    return grid::ObsWindow(s,
        [](int /*col*/, int /*row*/, float* o) { o[0] = 0.0f; return true; },
        {});
}

std::vector<float> build_obs(const grid::ObsWindow& win, const CorridorState& st) {
    float self[1] = { static_cast<float>(st.goal - st.pos) / static_cast<float>(st.goal) };
    std::vector<float> out(static_cast<size_t>(win.out_dim()), 0.0f);
    win.build(st.pos, 0, self, 1, out.data());
    return out;
}

mcts::GenericEnv make_corridor_env(CorridorState& st, const grid::ObsWindow& win) {
    mcts::GenericEnv env;
    env.num_actions = 3;
    env.snapshot_fn = [&]() -> std::any { return st; };
    env.restore_fn  = [&](const std::any& s) { st = std::any_cast<CorridorState>(s); };
    env.observe_fn  = [&]() -> std::vector<float> { return build_obs(win, st); };
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

// Forward the current published weights through PolicyValueNet to compute
// (value, masked policy probabilities) for use as prior + value in MCTS.
struct NetEval {
    nn::PolicyValueNet  net;
    nn::WeightsHandle*  handle = nullptr;
    uint64_t            cached_version = 0;

    void refresh(const nn::PolicyValueNet::Config& cfg) {
        if (!handle) return;
        uint64_t v = 0;
        auto blob = handle->snapshot(&v);
        if (!blob) return;
        if (v != cached_version) {
            net.init(cfg);
            net.load(*blob);
            cached_version = v;
        }
    }

    void forward(const std::vector<float>& obs,
                 std::vector<int>&         legal,
                 float&                    value,
                 std::vector<float>&       probs) {
        nn::Tensor x = nn::Tensor::vec(net.in_dim());
        for (int i = 0; i < net.in_dim(); ++i)
            x[i] = (i < static_cast<int>(obs.size())) ? obs[static_cast<size_t>(i)] : 0.0f;
        nn::Tensor logits = nn::Tensor::vec(net.num_actions());
        net.forward(x, value, logits);

        std::vector<float> mask(static_cast<size_t>(net.num_actions()), 0.0f);
        for (int a : legal)
            if (a >= 0 && a < net.num_actions()) mask[static_cast<size_t>(a)] = 1.0f;
        nn::Tensor probs_t = nn::Tensor::vec(net.num_actions());
        nn::softmax_forward(logits, probs_t, mask.empty() ? nullptr : mask.data());
        probs.resize(static_cast<size_t>(net.num_actions()));
        for (int i = 0; i < net.num_actions(); ++i) probs[static_cast<size_t>(i)] = probs_t[i];
    }
};

} // namespace

int main() {
    // ─── Setup ────────────────────────────────────────────────────────
    grid::ObsWindow win = make_obs_window();
    if (win.out_dim() != kInDim) {
        std::fprintf(stderr, "obs dim mismatch: %d vs %d\n", win.out_dim(), kInDim);
        return 1;
    }

    grid::GridTrainerConfig hcfg;
    hcfg.net.in_dim       = kInDim;
    hcfg.net.hidden       = {32, 32};
    hcfg.net.value_hidden = 16;
    hcfg.net.num_actions  = 3;
    hcfg.net.seed         = 0xC0DE1234ULL;
    hcfg.buffer_capacity  = 8192;
    hcfg.trainer.batch    = 16;
    hcfg.trainer.lr       = 5e-3f;
    hcfg.trainer.publish_every = 50;
    hcfg.best_window      = 16;

    grid::GridTrainer trainer(std::move(hcfg));

    // ─── BC warmup: a hand-coded "always go right" demonstration ──────
    {
        CorridorState st;
        auto env = make_corridor_env(st, win);
        auto policy = [](const std::vector<float>&, const std::vector<int>&) { return 1; };
        grid::BCConfig cfg;
        cfg.rollout_horizon = 32;
        cfg.min_return      = 0.5f;
        cfg.gamma           = 0.99f;
        std::vector<std::any> starts = { CorridorState{} };
        auto sits = grid::generate_bc_situations(env, policy, starts, cfg);
        trainer.warmup_with(sits);
        std::printf("BC warmup: %zu situations\n", sits.size());
    }

    // ─── Self-play loop ───────────────────────────────────────────────
    NetEval evaluator;
    evaluator.handle = &trainer.weights();

    CorridorState st;
    auto env = make_corridor_env(st, win);

    auto prior_fn = [&](const std::vector<float>& obs,
                        const std::vector<int>&   legal) {
        evaluator.refresh(trainer.net().config());
        std::vector<float> probs;
        float v;
        std::vector<int> legal_copy = legal;
        evaluator.forward(obs, legal_copy, v, probs);
        return probs;
    };
    auto value_fn = [&](const std::vector<float>& obs) {
        evaluator.refresh(trainer.net().config());
        std::vector<float> probs;
        float v = 0.0f;
        std::vector<int> empty;
        evaluator.forward(obs, empty, v, probs);
        return v;
    };

    mcts::GenericMcts m(env);
    mcts::GenericMctsConfig mcfg;
    mcfg.iterations    = 64;
    mcfg.c_puct        = 1.5f;
    mcfg.gamma         = 0.99f;
    mcfg.rollout_depth = 16;
    mcfg.seed          = 1;
    m.set_config(mcfg);
    m.set_prior_fn(prior_fn);
    m.set_value_fn(value_fn);

    constexpr int kEpisodes = 80;
    int wins = 0;
    for (int ep = 0; ep < kEpisodes; ++ep) {
        st = CorridorState{};
        std::vector<learn::GenericSituation> ep_sits;
        std::vector<grid::FailureStep>       ep_tail;
        std::vector<int>                     ep_actions;
        std::any                             ep_start = st;
        float                                ep_return = 0.0f;
        int                                  ep_depth  = 0;

        for (int t = 0; t < 32 && !st.done; ++t) {
            auto obs   = build_obs(win, st);
            auto legal = env.legal_actions_fn();

            // Run a search from the current state.
            CorridorState saved = st;
            int chosen = m.search();
            st = saved;       // search restores; but be defensive.
            if (chosen < 0) break;

            // Build a training situation (visit-distribution policy target).
            auto visits = m.root_visits();
            learn::GenericSituation sit;
            sit.obs           = obs;
            sit.policy_target = visits;
            sit.action_mask.assign(3, 0.0f);
            for (int a : legal) if (a >= 0 && a < 3) sit.action_mask[static_cast<size_t>(a)] = 1.0f;
            sit.value_target  = 0.0f;   // filled with discounted return below
            ep_sits.push_back(std::move(sit));
            ep_actions.push_back(chosen);

            // Cheap signature: just the position bucket.
            char sigbuf[8];
            std::snprintf(sigbuf, sizeof(sigbuf), "%d", st.pos);
            ep_tail.push_back({ sigbuf, chosen });

            // Apply.
            auto sr = env.step_fn(chosen);
            ep_return += sr.reward;
            ++ep_depth;
            m.advance_root(chosen);
            if (sr.done) break;
        }

        // Discounted return-to-go, clipped to [-1, 1].
        float running = 0.0f;
        for (int i = static_cast<int>(ep_sits.size()) - 1; i >= 0; --i) {
            running = (i == static_cast<int>(ep_sits.size()) - 1 ? ep_return
                                                                  : running) * mcfg.gamma;
            // The above isn't quite right (we'd like per-step rewards) —
            // approximate: terminal reward only, propagated via gamma.
            ep_sits[static_cast<size_t>(i)].value_target =
                std::max(-1.0f, std::min(1.0f, running));
        }

        for (auto& s : ep_sits) trainer.ingest_situation(std::move(s));

        grid::EpisodeSummary es;
        es.total_return   = ep_return;
        es.depth          = ep_depth;
        es.failed         = !st.done || ep_return <= 0.0f;
        es.start_snapshot = ep_start;
        es.action_prefix  = ep_actions;
        es.failure_tail   = es.failed ? ep_tail : std::vector<grid::FailureStep>{};
        trainer.ingest_episode(std::move(es));

        if (st.done && ep_return > 0.0f) ++wins;

        // Train.
        trainer.step_sync(/*sgd_steps*/ 16);

        m.reset();   // start each episode with a fresh tree.

        if ((ep + 1) % 10 == 0) {
            auto stx = trainer.stats();
            std::printf("ep=%3d wins=%d steps=%d publishes=%d buf=%d trail=%.3f\n",
                ep + 1, wins, stx.total_steps, stx.total_publishes,
                stx.buffer_size, stx.trailing_mean_return);
        }
    }

    auto final_stats = trainer.stats();
    std::printf("\nfinal: episodes=%d wins=%d steps=%d publishes=%d best=%d failures=%d\n",
        kEpisodes, wins,
        final_stats.total_steps, final_stats.total_publishes,
        trainer.best_crop().size(), trainer.failure_tape().size());
    return 0;
}
