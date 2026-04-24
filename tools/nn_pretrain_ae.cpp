// nn_pretrain_ae — unsupervised autoencoder pretraining for the
// DeepSetsEncoder. Produces a .bgnn-style encoder-only blob that can be
// loaded into a SingleHeroNet via SingleHeroNet::load_encoder_only before
// supervised value/policy fine-tuning.
//
// Observation source:
//   `.bgar` replay files capture full world state, not pre-computed
//   observation vectors. The project's existing `nn_train_value` tool does
//   not parse replay files either — it generates training data by running
//   MCTS self-play and sampling observations at each decision window. We
//   mirror that pattern here: run short MCTS duels and collect the
//   resulting observations from the ReplayBuffer. This reuses the
//   replay_buffer / Situation path, which is the actual "replay" path used
//   elsewhere in the codebase for NN training. A future extension could
//   read genuine `.bgar` files, rebuild the World at each indexed frame,
//   and call observation::build — but that is out of scope here.
//
// Usage: nn_pretrain_ae --out <path.bgnn> [options]

#include "brogameagent/agent.h"
#include "brogameagent/learn/replay_buffer.h"
#include "brogameagent/learn/search_trace.h"
#include "brogameagent/mcts.h"
#include "brogameagent/nn/autoencoder.h"
#include "brogameagent/nn/net.h"
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
    std::string replays;     // unused if empty — see note above
    std::string out;
    std::string out_ae;
    int      epochs    = 5;
    int      batch     = 64;
    float    lr        = 0.01f;
    float    momentum  = 0.9f;
    int      embed_dim = 32;
    int      hidden    = 32;
    int      dec_hidden= 32;
    uint64_t seed      = 0xAE12345678ULL;
    bool     verbose   = false;
    // self-play data-gen knobs (used when we synthesize observations).
    int      episodes   = 16;
    int      iterations = 40;
    int      max_ticks  = 200;
};

void print_help() {
    std::printf(
        "nn_pretrain_ae — pretrain DeepSetsEncoder as an autoencoder.\n"
        "\n"
        "Required:\n"
        "  --out <path.bgnn>       encoder-only weight blob (DeepSetsEncoder::save_to format)\n"
        "\n"
        "Options:\n"
        "  --replays <path>        replay source glob (currently ignored — data is\n"
        "                          generated via MCTS self-play; see source comment)\n"
        "  --out_ae <path>         also save full autoencoder blob\n"
        "  --epochs N              epochs over shuffled pool (default 5)\n"
        "  --batch N               minibatch size (default 64)\n"
        "  --lr F                  learning rate (default 0.01)\n"
        "  --momentum F            SGD momentum (default 0.9)\n"
        "  --embed_dim N           encoder per-stream embed (default 32)\n"
        "  --hidden N              encoder per-stream hidden (default 32)\n"
        "  --dec_hidden N          decoder per-stream hidden (default 32)\n"
        "  --seed N                RNG seed (default %llu)\n"
        "  --episodes N            self-play episodes for data gen (default 16)\n"
        "  --iterations N          MCTS iterations per decision (default 40)\n"
        "  --max-ticks N           max ticks per episode (default 200)\n"
        "  --verbose               print per-step loss\n"
        "  --help                  this message\n",
        static_cast<unsigned long long>(0xAE12345678ULL));
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--replays")    a.replays    = need() ? argv[i] : "";
        else if (k == "--out")        a.out        = need() ? argv[i] : "";
        else if (k == "--out_ae")     a.out_ae     = need() ? argv[i] : "";
        else if (k == "--epochs")     a.epochs     = std::atoi(need());
        else if (k == "--batch")      a.batch      = std::atoi(need());
        else if (k == "--lr")         a.lr         = static_cast<float>(std::atof(need()));
        else if (k == "--momentum")   a.momentum   = static_cast<float>(std::atof(need()));
        else if (k == "--embed_dim")  a.embed_dim  = std::atoi(need());
        else if (k == "--hidden")     a.hidden     = std::atoi(need());
        else if (k == "--dec_hidden") a.dec_hidden = std::atoi(need());
        else if (k == "--seed")       a.seed       = std::strtoull(need(), nullptr, 0);
        else if (k == "--episodes")   a.episodes   = std::atoi(need());
        else if (k == "--iterations") a.iterations = std::atoi(need());
        else if (k == "--max-ticks")  a.max_ticks  = std::atoi(need());
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

// Convert Situation.obs (std::array) into a Tensor of size observation::TOTAL.
void to_tensor(const learn::Situation& sit, nn::Tensor& out) {
    if (out.size() != observation::TOTAL) out.resize(observation::TOTAL, 1);
    for (int i = 0; i < observation::TOTAL; ++i) out[i] = sit.obs[i];
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    // ─── Gather observations ───────────────────────────────────────────────
    std::printf("phase\tstep\tdetail\n");
    std::printf("gen\t0\tcollecting %d episodes @ %d iter (replays flag=\"%s\", ignored)\n",
                a.episodes, a.iterations, a.replays.c_str());
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

    // 10% held-out validation.
    std::mt19937_64 rng(a.seed ^ 0xDEADBEEFULL);
    std::shuffle(indices.begin(), indices.end(), rng);
    const int n_val = std::max(1, static_cast<int>(indices.size() / 10));
    std::vector<int> val(indices.begin(), indices.begin() + n_val);
    std::vector<int> train(indices.begin() + n_val, indices.end());
    std::printf("split\t0\ttrain=%zu val=%zu\n", train.size(), val.size());

    // ─── Init AE ───────────────────────────────────────────────────────────
    nn::DeepSetsAutoencoder ae;
    nn::DeepSetsAutoencoder::Config cfg;
    cfg.enc.embed_dim = a.embed_dim;
    cfg.enc.hidden    = a.hidden;
    cfg.dec_hidden    = a.dec_hidden;
    cfg.seed          = a.seed ^ 0xC0FFEE00ULL;
    ae.init(cfg);
    std::printf("net\tparams\t%d\n", ae.num_params());

    nn::Tensor x     = nn::Tensor::vec(observation::TOTAL);
    nn::Tensor x_hat = nn::Tensor::vec(observation::TOTAL);
    nn::Tensor dXh   = nn::Tensor::vec(observation::TOTAL);

    auto eval_loss = [&](const std::vector<int>& idxs) {
        double sum = 0.0;
        for (int i : idxs) {
            to_tensor(all[i], x);
            ae.forward(x, x_hat);
            sum += reconstruction_loss(x, x_hat, dXh);
        }
        return idxs.empty() ? 0.0f : static_cast<float>(sum / idxs.size());
    };

    // ─── Train ─────────────────────────────────────────────────────────────
    std::printf("train\tepoch\tloss\tval_loss\n");
    auto tt0 = std::chrono::steady_clock::now();
    for (int ep = 0; ep < a.epochs; ++ep) {
        std::shuffle(train.begin(), train.end(), rng);
        double ep_loss_sum = 0.0;
        int    ep_samples  = 0;

        for (size_t bstart = 0; bstart < train.size(); bstart += a.batch) {
            const size_t bend = std::min(train.size(), bstart + static_cast<size_t>(a.batch));
            ae.zero_grad();
            double batch_loss = 0.0;
            for (size_t i = bstart; i < bend; ++i) {
                to_tensor(all[train[i]], x);
                ae.forward(x, x_hat);
                batch_loss += reconstruction_loss(x, x_hat, dXh);
                ae.backward(dXh);
            }
            // Scale grads by 1/batch for mean-loss SGD.
            const float scale = 1.0f / static_cast<float>(bend - bstart);
            ae.sgd_step(a.lr * scale, a.momentum);
            ep_loss_sum += batch_loss;
            ep_samples  += static_cast<int>(bend - bstart);
            if (a.verbose) {
                std::printf("step\t%zu\tbatch_loss=%.6f\n",
                            bstart / a.batch,
                            ep_samples > 0 ? batch_loss / (bend - bstart) : 0.0);
            }
        }

        const float train_loss = ep_samples > 0
            ? static_cast<float>(ep_loss_sum / ep_samples) : 0.0f;
        const float val_loss = eval_loss(val);
        std::printf("epoch\t%d\tloss=%.6f\tval_loss=%.6f\n", ep + 1, train_loss, val_loss);
    }
    auto tt1 = std::chrono::steady_clock::now();
    long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tt1-tt0).count();
    std::printf("train\tdone\ttrain_ms=%ld\n", train_ms);

    // ─── Save encoder blob ────────────────────────────────────────────────
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

    // ─── Sanity check: round-trip encoder blob into a fresh SingleHeroNet ──
    nn::SingleHeroNet net;
    nn::SingleHeroNet::Config ncfg;
    ncfg.enc.embed_dim = a.embed_dim;
    ncfg.enc.hidden    = a.hidden;
    ncfg.trunk_hidden  = 64;
    ncfg.value_hidden  = 32;
    ncfg.seed          = a.seed ^ 0xFEEDFACEULL;
    net.init(ncfg);
    net.load_encoder_only(enc_blob);

    // Forward the same observation through AE encoder and through the net's
    // encoder (by construction: compare decoded output's first SELF_FEATURES
    // block is insufficient — instead we compare the AE forward-reconstruction
    // against a rebuilt one using the same encoder weights).
    nn::DeepSetsAutoencoder ae2;
    ae2.init(cfg);
    // Overwrite ae2's encoder with the freshly-serialized encoder blob.
    {
        size_t off = 0;
        ae2.encoder().load_from(enc_blob.data(), off, enc_blob.size());
    }
    // Also copy decoder from ae -> ae2 so the end-to-end is meaningful.
    {
        std::vector<uint8_t> dec_blob;
        ae.decoder().save_to(dec_blob);
        size_t off = 0;
        ae2.decoder().load_from(dec_blob.data(), off, dec_blob.size());
    }
    to_tensor(all[train.empty() ? val[0] : train[0]], x);
    nn::Tensor x_hat_a = nn::Tensor::vec(observation::TOTAL);
    nn::Tensor x_hat_b = nn::Tensor::vec(observation::TOTAL);
    ae.forward(x, x_hat_a);
    ae2.forward(x, x_hat_b);
    float max_err = 0.0f;
    for (int i = 0; i < observation::TOTAL; ++i)
        max_err = std::max(max_err, std::fabs(x_hat_a[i] - x_hat_b[i]));
    std::printf("check\troundtrip\tmax_abs_err=%.6e\t%s\n",
                max_err, max_err < 1e-5f ? "PASS" : "FAIL");
    return 0;
}
