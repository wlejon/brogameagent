// nn_train_value — Phase-2 tool.
//
// 1) Generate episodes by running MCTS with HpDeltaEvaluator against a
//    scripted opponent. Every decision window, capture the hero's
//    observation + MCTS-derived policy targets + (eventually) discounted
//    return.
// 2) Train SingleHeroNet to match those targets via ExItTrainer.
// 3) Save weights to a .bgnn file.
//
// Usage:
//   nn_train_value [--episodes N] [--iterations M] [--steps S] [--out F]
//                  [--seed X]
//
// Output: TSV rows — one per training step sampled at a log cadence.

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
    int   episodes   = 8;
    int   iterations = 80;
    int   max_ticks  = 200;
    int   train_steps= 2000;
    int   log_every  = 100;
    uint64_t seed    = 0xC0DE1337ULL;
    std::string out  = "value.bgnn";
    int   buf_cap    = 4096;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--episodes")   a.episodes    = std::atoi(need());
        else if (k == "--iterations") a.iterations  = std::atoi(need());
        else if (k == "--max-ticks")  a.max_ticks   = std::atoi(need());
        else if (k == "--steps")      a.train_steps = std::atoi(need());
        else if (k == "--log-every")  a.log_every   = std::atoi(need());
        else if (k == "--buf")        a.buf_cap     = std::atoi(need());
        else if (k == "--seed")       a.seed        = std::strtoull(need(), nullptr, 0);
        else if (k == "--out")        a.out         = need();
        else if (k == "--help" || k == "-h") {
            std::fprintf(stderr, "nn_train_value: see source for flags\n");
            return false;
        }
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

// Run one episode; push Situations into buf. value_target is set after we
// know the final return using a backward pass over captured decisions.
int run_episode_and_capture(const Args& a, int ep_idx, learn::ReplayBuffer& buf) {
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

    std::vector<learn::Situation> captured;

    const float dt = cfg.sim_dt;
    for (int t = 0; t < a.max_ticks; ++t) {
        if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        mcts::CombatAction act = engine.search(s->world, s->hero);
        const mcts::Node* root = engine.last_root();
        if (root) captured.push_back(learn::make_situation(s->world, s->hero, *root));

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

    // Compute final return in [-1, 1]: HP-delta at episode end.
    float hero_hp  = s->hero.unit().alive()  ? s->hero.unit().hp  / s->hero.unit().maxHp  : 0.0f;
    float enemy_hp = s->enemy.unit().alive() ? s->enemy.unit().hp / s->enemy.unit().maxHp : 0.0f;
    float final_return = hero_hp - enemy_hp;
    if (!s->hero.unit().alive())       final_return = -1.0f;
    else if (!s->enemy.unit().alive()) final_return =  1.0f;

    // Backward discounted return assignment.
    const float gamma = 0.97f;
    float g = final_return;
    for (int i = static_cast<int>(captured.size()) - 1; i >= 0; --i) {
        captured[i].value_target = g;
        g *= gamma;
    }
    for (auto& sit : captured) buf.push(sit);
    return static_cast<int>(captured.size());
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    learn::ReplayBuffer buf(a.buf_cap);
    std::printf("phase\tstep\tdetail\n");
    std::printf("gen\t0\tcollecting %d episodes @ %d iter\n", a.episodes, a.iterations);
    auto t0 = std::chrono::steady_clock::now();
    int total = 0;
    for (int i = 0; i < a.episodes; ++i) {
        total += run_episode_and_capture(a, i, buf);
    }
    auto t1 = std::chrono::steady_clock::now();
    long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::printf("gen\t1\tcaptured=%d buf=%zu gen_ms=%ld\n", total, buf.size(), gen_ms);

    if (buf.size() == 0) {
        std::fprintf(stderr, "empty buffer — nothing to train on\n");
        return 1;
    }

    nn::SingleHeroNet net;
    nn::SingleHeroNet::Config ncfg;
    ncfg.enc.hidden = 32;
    ncfg.enc.embed_dim = 32;
    ncfg.trunk_hidden = 64;
    ncfg.value_hidden = 32;
    ncfg.seed = a.seed ^ 0xA1A2A3A4ULL;
    net.init(ncfg);
    std::printf("net\tparams\t%d\n", net.num_params());

    learn::ExItTrainer tr;
    tr.set_net(&net);
    tr.set_buffer(&buf);
    learn::TrainerConfig tcfg;
    tcfg.lr = 0.005f;
    tcfg.momentum = 0.9f;
    tcfg.batch = 32;
    tcfg.rng_seed = a.seed ^ 0xB1B2B3B4ULL;
    tcfg.publish_every = 0;  // file-based save at end
    tr.set_config(tcfg);

    std::printf("train\tstep\tloss_v\tloss_p\tloss_total\n");
    auto tt0 = std::chrono::steady_clock::now();
    for (int s = 0; s < a.train_steps; ++s) {
        auto r = tr.step();
        if ((s + 1) % a.log_every == 0 || s == 0) {
            std::printf("train\t%d\t%.4f\t%.4f\t%.4f\n",
                        s + 1, r.loss_value, r.loss_policy, r.loss_total);
        }
    }
    auto tt1 = std::chrono::steady_clock::now();
    long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tt1-tt0).count();
    std::printf("train\tdone\ttrain_ms=%ld\n", train_ms);

    auto blob = net.save();
    std::ofstream f(a.out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
    std::printf("save\tout\t%s\tbytes=%zu\n", a.out.c_str(), blob.size());
    return 0;
}
