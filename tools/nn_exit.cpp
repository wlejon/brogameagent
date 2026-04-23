// nn_exit — end-to-end ExIt loop.
//
// Iteration k:
//   - Run N episodes using MCTS with PUCT + NeuralPrior(k) + NeuralEvaluator(k).
//   - Capture (obs, legal-mask, MCTS π, final return) tuples into the replay buffer.
//   - SGD M minibatches on that buffer.
//   - Evaluate the updated net against a frozen scripted baseline over E episodes.
//   - Log: iter, gen_ms, train_ms, win_rate_vs_baseline, mean_hp_delta.
//   - Save the net to F_iter.bgnn.
//
// The trainer publishes to a WeightsHandle in the foreground (not a separate
// thread here — this CLI is demonstrating correctness, not concurrency).
// Switching this to async requires only replacing the synchronous
// `tr.step_n` loop with a std::thread that calls step() forever.
//
// Usage:
//   nn_exit [--iters K] [--episodes N] [--iterations M] [--max-ticks T]
//           [--steps S] [--eval E] [--out-prefix P] [--seed X]

#include "brogameagent/agent.h"
#include "brogameagent/learn/neural_adapters.h"
#include "brogameagent/learn/replay_buffer.h"
#include "brogameagent/learn/search_trace.h"
#include "brogameagent/learn/trainer.h"
#include "brogameagent/mcts.h"
#include "brogameagent/nn/net.h"
#include "brogameagent/world.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace brogameagent;

namespace {

struct Args {
    int iters        = 3;
    int episodes     = 6;
    int iterations   = 80;     // MCTS iterations per decision
    int max_ticks    = 200;
    int train_steps  = 1500;
    int eval_episodes= 8;
    int buf_cap      = 8192;
    uint64_t seed    = 0xEEE5CAFEULL;
    std::string out_prefix = "exit_iter";
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--iters")      a.iters       = std::atoi(need());
        else if (k == "--episodes")   a.episodes    = std::atoi(need());
        else if (k == "--iterations") a.iterations  = std::atoi(need());
        else if (k == "--max-ticks")  a.max_ticks   = std::atoi(need());
        else if (k == "--steps")      a.train_steps = std::atoi(need());
        else if (k == "--eval")       a.eval_episodes = std::atoi(need());
        else if (k == "--buf")        a.buf_cap     = std::atoi(need());
        else if (k == "--seed")       a.seed        = std::strtoull(need(), nullptr, 0);
        else if (k == "--out-prefix") a.out_prefix  = need();
        else if (k == "--help" || k == "-h") return false;
    }
    return true;
}

struct DuelScene {
    World world;
    Agent hero;
    Agent enemy;
};

std::unique_ptr<DuelScene> build_duel(uint64_t seed) {
    auto s = std::make_unique<DuelScene>();
    s->hero.unit().id = 1; s->hero.unit().teamId = 0;
    s->hero.unit().hp = 100; s->hero.unit().maxHp = 100;
    s->hero.unit().damage = 10; s->hero.unit().attackRange = 3;
    s->hero.unit().attacksPerSec = 2;
    s->hero.setPosition(-1.0f, 0); s->hero.setMaxAccel(30); s->hero.setMaxTurnRate(10);

    s->enemy.unit().id = 2; s->enemy.unit().teamId = 1;
    s->enemy.unit().hp = 80; s->enemy.unit().maxHp = 80;
    s->enemy.unit().damage = 8; s->enemy.unit().attackRange = 3;
    s->enemy.unit().attacksPerSec = 1.5f;
    s->enemy.setPosition(1.0f, 0); s->enemy.setMaxAccel(30); s->enemy.setMaxTurnRate(10);

    s->world.addAgent(&s->hero);
    s->world.addAgent(&s->enemy);
    s->world.seed(seed);
    return s;
}

// Run one MCTS-driven episode, optionally with neural prior+evaluator.
// Captures situations into `buf` with discounted-return value_target.
// Returns (+1 hero win, -1 loss, 0 draw) and final hp_delta via out-params.
struct EpResult {
    int outcome = 0;
    float hp_delta = 0.0f;
    int situations = 0;
};

EpResult run_one(uint64_t ep_seed, int iterations, int max_ticks,
                 std::shared_ptr<nn::SingleHeroNet> net,
                 nn::WeightsHandle* handle,
                 learn::ReplayBuffer* buf_opt)
{
    EpResult r;
    auto s = build_duel(ep_seed);

    mcts::MctsConfig cfg;
    cfg.iterations = iterations;
    cfg.rollout_horizon = 12;
    cfg.action_repeat = 4;
    cfg.seed = ep_seed;
    cfg.use_leaf_value = (net != nullptr);  // rely on NN value if present
    cfg.prior_c = (net != nullptr) ? 1.5f : 0.0f;

    mcts::Mcts engine(cfg);
    if (net) {
        engine.set_evaluator(std::make_shared<learn::NeuralEvaluator>(net, handle));
        engine.set_prior(std::make_shared<learn::NeuralPrior>(net, handle));
    } else {
        engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    }
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    std::vector<learn::Situation> captured;
    const float dt = cfg.sim_dt;
    for (int t = 0; t < max_ticks; ++t) {
        if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        mcts::CombatAction act = engine.search(s->world, s->hero);
        if (buf_opt) {
            if (auto* root = engine.last_root())
                captured.push_back(learn::make_situation(s->world, s->hero, *root));
        }
        mcts::CombatAction o = s->enemy.unit().alive()
            ? mcts::policy_aggressive(s->enemy, s->world) : mcts::CombatAction{};
        for (int w = 0; w < cfg.action_repeat; ++w) {
            mcts::apply(s->hero, s->world, act, dt);
            mcts::apply(s->enemy, s->world, o, dt);
            s->world.stepProjectiles(dt);
            s->world.cullProjectiles();
            if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        }
        engine.advance_root(act);
    }

    bool ha = s->hero.unit().alive(), ea = s->enemy.unit().alive();
    if (ha && !ea)      r.outcome = +1;
    else if (!ha && ea) r.outcome = -1;
    else                r.outcome =  0;
    float hh = ha ? s->hero.unit().hp / s->hero.unit().maxHp : 0.0f;
    float eh = ea ? s->enemy.unit().hp / s->enemy.unit().maxHp : 0.0f;
    r.hp_delta = hh - eh;

    if (buf_opt) {
        float final_ret = hh - eh;
        if (!ha) final_ret = -1.0f; else if (!ea) final_ret = 1.0f;
        const float gamma = 0.97f;
        float g = final_ret;
        for (int i = static_cast<int>(captured.size()) - 1; i >= 0; --i) {
            captured[i].value_target = g;
            g *= gamma;
        }
        for (auto& sit : captured) buf_opt->push(sit);
        r.situations = static_cast<int>(captured.size());
    }
    return r;
}

void save_net(const nn::SingleHeroNet& net, const std::string& path) {
    auto blob = net.save();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    // Init net (iteration-0 weights).
    auto net = std::make_shared<nn::SingleHeroNet>();
    nn::SingleHeroNet::Config ncfg;
    ncfg.enc.hidden = 32;
    ncfg.enc.embed_dim = 32;
    ncfg.trunk_hidden = 64;
    ncfg.value_hidden = 32;
    ncfg.seed = a.seed ^ 0xA1A2A3A4ULL;
    net->init(ncfg);

    nn::WeightsHandle handle;
    handle.publish(net->save(), 0);

    learn::ReplayBuffer buf(a.buf_cap);

    // Baseline: frozen scripted (MCTS + HpDelta + AggressiveRollout) vs same.
    // Represents iteration-0 behavior.

    std::printf("iter\tphase\tmetric\tvalue\n");
    for (int it = 0; it < a.iters; ++it) {
        // === Generate ===
        std::printf("%d\tgen\tstart_buf\t%zu\n", it, buf.size());
        auto tg0 = std::chrono::steady_clock::now();
        int wins = 0, losses = 0, total_sit = 0;
        float hpd_sum = 0.0f;
        // First iteration uses no net (handle has untrained weights — skip
        // neural bits in favor of classical MCTS for cleaner iteration-0
        // data). Subsequent iterations use net.
        auto gen_net = (it == 0) ? std::shared_ptr<nn::SingleHeroNet>() : net;
        auto gen_handle = (it == 0) ? nullptr : &handle;
        for (int ep = 0; ep < a.episodes; ++ep) {
            auto r = run_one(a.seed + it * 1000 + ep, a.iterations, a.max_ticks,
                             gen_net, gen_handle, &buf);
            if (r.outcome > 0) ++wins;
            else if (r.outcome < 0) ++losses;
            total_sit += r.situations;
            hpd_sum += r.hp_delta;
        }
        auto tg1 = std::chrono::steady_clock::now();
        long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tg1-tg0).count();
        std::printf("%d\tgen\twins\t%d\n", it, wins);
        std::printf("%d\tgen\tlosses\t%d\n", it, losses);
        std::printf("%d\tgen\tsituations\t%d\n", it, total_sit);
        std::printf("%d\tgen\tmean_hp_delta\t%.4f\n", it, hpd_sum / a.episodes);
        std::printf("%d\tgen\tms\t%ld\n", it, gen_ms);

        // === Train ===
        if (buf.size() == 0) continue;
        learn::ExItTrainer tr;
        tr.set_net(net.get());
        tr.set_buffer(&buf);
        tr.set_weights_handle(&handle);
        learn::TrainerConfig tcfg;
        tcfg.lr = 0.005f;
        tcfg.momentum = 0.9f;
        tcfg.batch = 32;
        tcfg.rng_seed = a.seed ^ (0xB1B2B3B4ULL + it);
        tcfg.publish_every = 250;
        tr.set_config(tcfg);

        auto tt0 = std::chrono::steady_clock::now();
        learn::TrainStep last;
        for (int s = 0; s < a.train_steps; ++s) last = tr.step();
        auto tt1 = std::chrono::steady_clock::now();
        long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tt1-tt0).count();
        std::printf("%d\ttrain\tloss_v\t%.4f\n", it, last.loss_value);
        std::printf("%d\ttrain\tloss_p\t%.4f\n", it, last.loss_policy);
        std::printf("%d\ttrain\tms\t%ld\n", it, train_ms);
        // Publish final weights.
        handle.publish(net->save(), static_cast<uint64_t>((it + 1) * a.train_steps));

        // === Eval vs frozen baseline (classical MCTS, no NN) ===
        // Plays the net-driven hero against the same world layout; the
        // opponent remains the scripted aggressive policy. We compare hp
        // delta to iteration-0 distribution.
        auto te0 = std::chrono::steady_clock::now();
        int ew = 0, el = 0;
        float ehpd = 0;
        for (int ep = 0; ep < a.eval_episodes; ++ep) {
            auto r = run_one(a.seed ^ 0xEEEE + ep, a.iterations, a.max_ticks,
                             net, &handle, nullptr);
            if (r.outcome > 0) ++ew;
            else if (r.outcome < 0) ++el;
            ehpd += r.hp_delta;
        }
        auto te1 = std::chrono::steady_clock::now();
        long eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(te1-te0).count();
        std::printf("%d\teval\twins\t%d\n", it, ew);
        std::printf("%d\teval\tlosses\t%d\n", it, el);
        std::printf("%d\teval\tmean_hp_delta\t%.4f\n", it, ehpd / a.eval_episodes);
        std::printf("%d\teval\twin_rate\t%.3f\n", it,
                    a.eval_episodes > 0 ? static_cast<float>(ew) / a.eval_episodes : 0.0f);
        std::printf("%d\teval\tms\t%ld\n", it, eval_ms);

        char path[256];
        std::snprintf(path, sizeof(path), "%s_%d.bgnn", a.out_prefix.c_str(), it);
        save_net(*net, path);
        std::printf("%d\tsave\tpath\t%s\n", it, path);
    }

    return 0;
}
