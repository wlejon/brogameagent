// nn_pretrain_ae_gpu — GPU autoencoder pretraining for DeepSetsEncoder.
//
// Mirrors the shape of nn_pretrain_ae.cpp but trains on Device::GPU end-to-end:
//   1. Generate observations via short MCTS duels (same as the CPU tool).
//   2. Construct DeepSetsAutoencoder, call to(Device::GPU) once.
//   3. Per step: upload observation → forward → mse_vec_forward → mse_vec_backward
//      → backward → sgd_step (all on GPU).
//   4. Periodically save checkpoint via save() (sync from device).
//
// Loss semantics differ from the CPU tool: this trainer uses mse_vec_*_gpu
// (plain mean-of-squared-diffs over the whole reconstruction, no masking),
// matching the task spec. The CPU tool uses a masked 0.5*d^2 reconstruction
// loss; both are valid pretraining objectives.

#include "brogameagent/agent.h"
#include "brogameagent/learn/replay_buffer.h"
#include "brogameagent/learn/search_trace.h"
#include "brogameagent/mcts.h"
#include "brogameagent/nn/autoencoder.h"
#include "brogameagent/nn/gpu/ops.h"
#include "brogameagent/nn/gpu/runtime.h"
#include "brogameagent/nn/gpu/tensor.h"
#include "brogameagent/nn/tensor.h"
#include "brogameagent/observation.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace brogameagent;

namespace {

struct Args {
    std::string out;
    std::string out_ae;
    int      epochs    = 5;
    float    lr        = 0.01f;
    float    momentum  = 0.9f;
    int      embed_dim = 32;
    int      hidden    = 32;
    int      dec_hidden= 32;
    uint64_t seed      = 0xAE12345678ULL;
    bool     verbose   = false;
    int      log_every = 100;
    // self-play data-gen knobs.
    int      episodes   = 16;
    int      iterations = 40;
    int      max_ticks  = 200;
};

void print_help() {
    std::printf(
        "nn_pretrain_ae_gpu — GPU pretraining of DeepSetsEncoder.\n"
        "\n"
        "Required:\n"
        "  --out <path.bgnn>      encoder-only weight blob\n"
        "\n"
        "Options:\n"
        "  --out_ae <path>        also save full autoencoder blob\n"
        "  --epochs N             epochs over shuffled pool (default 5)\n"
        "  --lr F                 learning rate (default 0.01)\n"
        "  --momentum F           SGD momentum (default 0.9)\n"
        "  --embed_dim N          encoder per-stream embed (default 32)\n"
        "  --hidden N             encoder per-stream hidden (default 32)\n"
        "  --dec_hidden N         decoder per-stream hidden (default 32)\n"
        "  --episodes N           self-play episodes (default 16)\n"
        "  --iterations N         MCTS iters per decision (default 40)\n"
        "  --max-ticks N          max ticks per episode (default 200)\n"
        "  --log-every N          print loss every N steps (default 100)\n"
        "  --seed N               RNG seed\n"
        "  --verbose              print per-step loss\n"
        "  --help                 this message\n");
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--out")        a.out        = need() ? argv[i] : "";
        else if (k == "--out_ae")     a.out_ae     = need() ? argv[i] : "";
        else if (k == "--epochs")     a.epochs     = std::atoi(need());
        else if (k == "--lr")         a.lr         = static_cast<float>(std::atof(need()));
        else if (k == "--momentum")   a.momentum   = static_cast<float>(std::atof(need()));
        else if (k == "--embed_dim")  a.embed_dim  = std::atoi(need());
        else if (k == "--hidden")     a.hidden     = std::atoi(need());
        else if (k == "--dec_hidden") a.dec_hidden = std::atoi(need());
        else if (k == "--seed")       a.seed       = std::strtoull(need(), nullptr, 0);
        else if (k == "--episodes")   a.episodes   = std::atoi(need());
        else if (k == "--iterations") a.iterations = std::atoi(need());
        else if (k == "--max-ticks")  a.max_ticks  = std::atoi(need());
        else if (k == "--log-every")  a.log_every  = std::atoi(need());
        else if (k == "--verbose")    a.verbose    = true;
        else if (k == "--help" || k == "-h") { print_help(); return false; }
        else { std::fprintf(stderr, "unknown flag: %s\n", k.c_str()); print_help(); return false; }
    }
    if (a.out.empty()) {
        std::fprintf(stderr, "error: --out is required\n\n");
        print_help();
        return false;
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

int gen_episode(const Args& a, int ep_idx, learn::ReplayBuffer& buf) {
    uint64_t ep_seed = a.seed + static_cast<uint64_t>(ep_idx);
    auto s = build_duel(ep_seed);

    mcts::MctsConfig cfg;
    cfg.iterations = a.iterations;
    cfg.rollout_horizon = 16;
    cfg.action_repeat = 4;
    cfg.seed = ep_seed;

    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_aggressive);

    const float dt = cfg.sim_dt;
    int captured = 0;
    for (int t = 0; t < a.max_ticks; ++t) {
        if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        mcts::CombatAction act = engine.search(s->world, s->hero);
        const mcts::Node* root = engine.last_root();
        if (root) {
            buf.push(learn::make_situation(s->world, s->hero, *root));
            ++captured;
        }
        mcts::CombatAction o_act = s->enemy.unit().alive()
            ? mcts::policy_aggressive(s->enemy, s->world) : mcts::CombatAction{};
        for (int w = 0; w < cfg.action_repeat; ++w) {
            mcts::apply(s->hero, s->world, act, dt);
            mcts::apply(s->enemy, s->world, o_act, dt);
            s->world.stepProjectiles(dt);
            s->world.cullProjectiles();
            if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        }
        engine.advance_root(act);
    }
    return captured;
}

void to_tensor(const learn::Situation& sit, nn::Tensor& out) {
    if (out.size() != observation::TOTAL) out.resize(observation::TOTAL, 1);
    for (int i = 0; i < observation::TOTAL; ++i) out[i] = sit.obs[i];
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    nn::gpu::cuda_init();

    // ─── Gather observations ───────────────────────────────────────────────
    std::printf("phase\tstep\tdetail\n");
    std::printf("gen\t0\tcollecting %d episodes @ %d iter\n",
                a.episodes, a.iterations);
    auto t0 = std::chrono::steady_clock::now();
    learn::ReplayBuffer buf(static_cast<size_t>(a.episodes) * a.max_ticks + 8);
    int total = 0;
    for (int i = 0; i < a.episodes; ++i) total += gen_episode(a, i, buf);
    auto t1 = std::chrono::steady_clock::now();
    long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::printf("gen\t1\tcaptured=%d buf=%zu gen_ms=%ld\n", total, buf.size(), gen_ms);
    if (buf.size() == 0) {
        std::fprintf(stderr, "empty buffer — nothing to train on\n");
        return 1;
    }

    const auto& all = buf.all();
    std::vector<int> indices(all.size());
    for (size_t i = 0; i < all.size(); ++i) indices[i] = static_cast<int>(i);
    std::mt19937_64 rng(a.seed ^ 0xDEADBEEFULL);
    std::shuffle(indices.begin(), indices.end(), rng);

    // ─── Init AE on GPU ────────────────────────────────────────────────────
    nn::DeepSetsAutoencoder ae;
    nn::DeepSetsAutoencoder::Config cfg;
    cfg.enc.embed_dim = a.embed_dim;
    cfg.enc.hidden    = a.hidden;
    cfg.dec_hidden    = a.dec_hidden;
    cfg.seed          = a.seed ^ 0xC0FFEE00ULL;
    ae.init(cfg);
    ae.to(nn::Device::GPU);
    std::printf("net\tparams\t%d\tdevice=GPU\n", ae.num_params());

    nn::Tensor x_host = nn::Tensor::vec(observation::TOTAL);
    nn::gpu::GpuTensor x_g(observation::TOTAL, 1);
    nn::gpu::GpuTensor x_hat_g(observation::TOTAL, 1);
    nn::gpu::GpuTensor target_g(observation::TOTAL, 1);
    nn::gpu::GpuTensor dXh_g(observation::TOTAL, 1);

    // ─── Train ─────────────────────────────────────────────────────────────
    std::printf("train\tepoch\tloss\n");
    auto tt0 = std::chrono::steady_clock::now();
    int step = 0;
    for (int ep = 0; ep < a.epochs; ++ep) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double ep_loss_sum = 0.0;
        int    ep_samples  = 0;

        for (int idx : indices) {
            to_tensor(all[idx], x_host);
            nn::gpu::upload(x_host, x_g);
            // target = x (autoencoder reconstruction). Reuse upload to a
            // separate buffer so the backward kernel can read both.
            nn::gpu::upload(x_host, target_g);

            ae.zero_grad();
            ae.forward(x_g, x_hat_g);
            const float loss = nn::gpu::mse_vec_forward_gpu(x_hat_g, target_g);
            nn::gpu::mse_vec_backward_gpu(x_hat_g, target_g, dXh_g);
            ae.backward(dXh_g);
            ae.sgd_step(a.lr, a.momentum);

            ep_loss_sum += loss;
            ++ep_samples;
            ++step;
            if (a.verbose && (step % a.log_every == 0)) {
                std::printf("step\t%d\tloss=%.6f\n", step, loss);
            }
        }
        const float train_loss = ep_samples > 0
            ? static_cast<float>(ep_loss_sum / ep_samples) : 0.0f;
        std::printf("epoch\t%d\tloss=%.6f\n", ep + 1, train_loss);
    }
    nn::gpu::cuda_sync();
    auto tt1 = std::chrono::steady_clock::now();
    long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tt1-tt0).count();
    std::printf("train\tdone\ttrain_ms=%ld\n", train_ms);

    // ─── Save (save_to syncs from device) ──────────────────────────────────
    auto enc_blob = ae.save_encoder();
    {
        std::ofstream f(a.out, std::ios::binary);
        f.write(reinterpret_cast<const char*>(enc_blob.data()),
                static_cast<std::streamsize>(enc_blob.size()));
    }
    std::printf("save\tenc\t%s\tbytes=%zu\n", a.out.c_str(), enc_blob.size());

    if (!a.out_ae.empty()) {
        auto ae_blob = ae.save();
        std::ofstream f(a.out_ae, std::ios::binary);
        f.write(reinterpret_cast<const char*>(ae_blob.data()),
                static_cast<std::streamsize>(ae_blob.size()));
        std::printf("save\tae\t%s\tbytes=%zu\n", a.out_ae.c_str(), ae_blob.size());
    }

    return 0;
}
