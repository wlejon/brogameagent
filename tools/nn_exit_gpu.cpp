// nn_exit_gpu — GPU ExIt training driver for PolicyValueNet via
// GenericExItTrainer.
//
// Mirrors the spirit of tools/nn_exit.cpp but runs end-to-end on Device::GPU
// using the new GPU code path through GenericExItTrainer. The CPU tool
// trains a SingleHeroNet via ExItTrainer; this tool drives the *generic*
// trainer (which is the one we just retrofitted for GPU), so the net it
// trains is PolicyValueNet rather than SingleHeroNet. The training data
// pipeline is intentionally simple: a synthesized supervised dataset over
// PolicyValueNet's bring-your-own-observation/action interface, which keeps
// the focus on demonstrating the GPU training loop rather than on the
// MCTS-data-generation orchestration that nn_exit.cpp showcases.
//
// Usage:
//   nn_exit_gpu [--steps N] [--batch B] [--lr F] [--in-dim D] [--head-sizes
//                "h0,h1,..."] [--samples S] [--log-every K] [--out PATH]
//                [--seed X]

#include "brogameagent/learn/generic_replay_buffer.h"
#include "brogameagent/learn/generic_trainer.h"
#include "brogameagent/nn/device.h"
#include "brogameagent/nn/gpu/runtime.h"
#include "brogameagent/nn/policy_value_net.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace brogameagent;

namespace {

struct Args {
    int  steps      = 500;
    int  batch      = 32;
    float lr        = 0.05f;
    float momentum  = 0.9f;
    int  in_dim     = 16;
    std::vector<int> head_sizes = {6, 4};
    int  samples    = 256;
    int  log_every  = 50;
    std::string out = "exit_gpu.bgnn";
    uint64_t seed   = 0xEEE5CAFEULL;
};

void print_help() {
    std::printf(
        "nn_exit_gpu — GPU training driver for PolicyValueNet via\n"
        "GenericExItTrainer (Device::GPU).\n"
        "\n"
        "Options:\n"
        "  --steps N         training steps (default 500)\n"
        "  --batch B         minibatch size (default 32)\n"
        "  --lr F            SGD learning rate (default 0.05)\n"
        "  --momentum F      SGD momentum (default 0.9)\n"
        "  --in-dim D        observation dim (default 16)\n"
        "  --head-sizes CSV  comma-separated head sizes (default 6,4)\n"
        "  --samples S       number of training tuples (default 256)\n"
        "  --log-every K     print loss every K steps (default 50)\n"
        "  --out PATH        save final checkpoint here (default exit_gpu.bgnn)\n"
        "  --seed N          RNG seed\n"
        "  --help            this message\n");
}

bool parse_csv(const std::string& s, std::vector<int>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::atoi(tok.c_str()));
    }
    return !out.empty();
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--steps")      a.steps     = std::atoi(need());
        else if (k == "--batch")      a.batch     = std::atoi(need());
        else if (k == "--lr")         a.lr        = static_cast<float>(std::atof(need()));
        else if (k == "--momentum")   a.momentum  = static_cast<float>(std::atof(need()));
        else if (k == "--in-dim")     a.in_dim    = std::atoi(need());
        else if (k == "--head-sizes") {
            const char* v = need();
            if (!v || !parse_csv(v, a.head_sizes)) {
                std::fprintf(stderr, "bad --head-sizes\n"); return false;
            }
        }
        else if (k == "--samples")    a.samples   = std::atoi(need());
        else if (k == "--log-every")  a.log_every = std::atoi(need());
        else if (k == "--out")        a.out       = need();
        else if (k == "--seed")       a.seed      = std::strtoull(need(), nullptr, 0);
        else if (k == "--help" || k == "-h") { print_help(); return false; }
        else { std::fprintf(stderr, "unknown flag: %s\n", k.c_str()); print_help(); return false; }
    }
    return true;
}

// Synthesize a deterministic supervised dataset:
//   obs    : random in [-1, 1]^in_dim
//   target : per-head one-hot picked by hashing the obs
//   value  : tanh-bounded scalar derived from the same hash.
// The dataset is small enough to overfit, so the trainer should drive loss
// down monotonically — a signal that the GPU forward/loss/backward/optim
// chain is wired correctly.
void synthesize(const Args& a, learn::GenericReplayBuffer& buf,
                int total_actions) {
    std::mt19937_64 rng(a.seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    int n_heads = static_cast<int>(a.head_sizes.size());
    std::vector<int> head_offsets(n_heads + 1, 0);
    for (int i = 0; i < n_heads; ++i)
        head_offsets[i + 1] = head_offsets[i] + a.head_sizes[i];

    for (int s = 0; s < a.samples; ++s) {
        learn::GenericSituation g;
        g.obs.resize(a.in_dim);
        float acc = 0.0f;
        for (int j = 0; j < a.in_dim; ++j) {
            const float v = u(rng);
            g.obs[j] = v;
            acc += v * static_cast<float>(j + 1);
        }
        g.policy_target.assign(total_actions, 0.0f);
        for (int h = 0; h < n_heads; ++h) {
            // Hash acc + head index → integer in [0, head_size).
            uint64_t hv = static_cast<uint64_t>(static_cast<int64_t>(acc * 1000.0f))
                          ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(h + 1));
            int pick = static_cast<int>(hv % static_cast<uint64_t>(a.head_sizes[h]));
            g.policy_target[head_offsets[h] + pick] = 1.0f;
        }
        // Bound value target into (-1, 1) via a soft squash.
        const float t = acc / static_cast<float>(a.in_dim);
        g.value_target = t / (1.0f + std::abs(t));
        buf.push(std::move(g));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;
    if (a.head_sizes.empty()) {
        std::fprintf(stderr, "head_sizes empty\n");
        return 2;
    }

    nn::gpu::cuda_init();

    int total_actions = 0;
    for (int h : a.head_sizes) total_actions += h;

    // Build the net and move to GPU.
    nn::PolicyValueNet net;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = a.in_dim;
    cfg.hidden = {64, 64};
    cfg.value_hidden = 32;
    cfg.head_sizes = a.head_sizes;
    cfg.seed = a.seed ^ 0xA1A2A3A4ULL;
    net.init(cfg);
    net.to(nn::Device::GPU);

    std::printf("net\tparams\t%d\tdevice=GPU\n", net.num_params());

    // Synthesize supervised data (mirrors the "load training data" step the
    // CPU tool does, without pulling in the world/MCTS pipeline).
    learn::GenericReplayBuffer buf(static_cast<size_t>(a.samples));
    synthesize(a, buf, total_actions);
    std::printf("data\tsamples\t%zu\n", buf.size());

    // Train.
    learn::GenericExItTrainer tr;
    learn::GenericTrainerConfig tcfg;
    tcfg.batch = a.batch;
    tcfg.lr = a.lr;
    tcfg.momentum = a.momentum;
    tcfg.publish_every = 0;
    tcfg.device = nn::Device::GPU;
    tcfg.rng_seed = a.seed ^ 0xB1B2B3B4ULL;
    tr.set_net(&net);
    tr.set_buffer(&buf);
    tr.set_config(tcfg);

    std::printf("train\tstep\tloss_total\tloss_value\tloss_policy\n");
    auto t0 = std::chrono::steady_clock::now();
    learn::GenericTrainStep first = tr.step();
    std::printf("train\t%d\t%.6f\t%.6f\t%.6f\n",
                1, first.loss_total, first.loss_value, first.loss_policy);
    learn::GenericTrainStep last = first;
    for (int s = 2; s <= a.steps; ++s) {
        last = tr.step();
        if (a.log_every > 0 && (s % a.log_every == 0 || s == a.steps)) {
            std::printf("train\t%d\t%.6f\t%.6f\t%.6f\n",
                        s, last.loss_total, last.loss_value, last.loss_policy);
        }
    }
    nn::gpu::cuda_sync();
    auto t1 = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::printf("train\tdone\tms=%ld\tinitial=%.6f\tfinal=%.6f\n",
                ms, first.loss_total, last.loss_total);

    // Save (PolicyValueNet::save() syncs from device).
    auto blob = net.save();
    std::ofstream f(a.out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
    std::printf("save\t%s\tbytes=%zu\n", a.out.c_str(), blob.size());
    return 0;
}
