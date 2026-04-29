// GPU smoke test for GenericExItTrainer.
//
// Trains a tiny PolicyValueNet (small trunk, small action space) on a fixed
// supervised mapping for 20 steps on Device::GPU. Verifies:
//   - the trainer accepts the GPU device flag;
//   - per-step loss values are finite;
//   - the loss after N steps is strictly less than the initial loss.
//
// This is the "milestone" test the task spec calls for. It exercises the
// upload → forward → softmax_xent_fused_gpu (per-head) → backward → sgd_step
// rhythm end-to-end without depending on any sim/world code.

#include <brogameagent/learn/generic_replay_buffer.h>
#include <brogameagent/learn/generic_trainer.h>
#include <brogameagent/nn/device.h>
#include <brogameagent/nn/gpu/runtime.h>
#include <brogameagent/nn/policy_value_net.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using brogameagent::nn::Device;
using brogameagent::nn::PolicyValueNet;
using brogameagent::learn::GenericExItTrainer;
using brogameagent::learn::GenericReplayBuffer;
using brogameagent::learn::GenericSituation;
using brogameagent::learn::GenericTrainerConfig;
using brogameagent::learn::GenericTrainStep;

namespace {

void push(GenericReplayBuffer& buf,
          std::vector<float> obs, int a0, int a1, float v) {
    GenericSituation s;
    s.obs = std::move(obs);
    s.policy_target.assign(7, 0.0f);
    s.policy_target[a0] = 1.0f;
    s.policy_target[3 + a1] = 1.0f;
    s.value_target = v;
    buf.push(std::move(s));
}

}  // namespace

int main() {
    brogameagent::nn::gpu::cuda_init();

    // Two-head net (head sizes 3, 4). Small trunk so the test is fast.
    PolicyValueNet net;
    PolicyValueNet::Config cfg;
    cfg.in_dim = 4;
    cfg.hidden = {16, 16};
    cfg.value_hidden = 8;
    cfg.head_sizes = {3, 4};
    cfg.seed = 0xABCDEF42ULL;
    net.init(cfg);
    net.to(Device::GPU);

    GenericReplayBuffer buf(16);
    push(buf, {1, 0, 0, 0}, 0, 0,  0.5f);
    push(buf, {0, 1, 0, 0}, 1, 1, -0.3f);
    push(buf, {0, 0, 1, 0}, 2, 2,  0.8f);
    push(buf, {0, 0, 0, 1}, 0, 3, -0.6f);

    GenericExItTrainer tr;
    GenericTrainerConfig tcfg;
    tcfg.batch = 4;
    tcfg.lr = 0.05f;
    tcfg.momentum = 0.9f;
    tcfg.publish_every = 0;
    tcfg.device = Device::GPU;
    tr.set_net(&net);
    tr.set_buffer(&buf);
    tr.set_config(tcfg);

    GenericTrainStep first = tr.step();
    GenericTrainStep last = first;
    for (int i = 0; i < 19; ++i) last = tr.step();

    auto finite = [](float x) { return std::isfinite(x); };
    if (!finite(first.loss_total) || !finite(last.loss_total)) {
        std::fprintf(stderr,
            "non-finite loss: first=%.6f last=%.6f\n",
            first.loss_total, last.loss_total);
        return 1;
    }
    if (first.samples != 4) {
        std::fprintf(stderr, "first.samples=%d expected 4\n", first.samples);
        return 1;
    }
    if (last.samples != 4) {
        std::fprintf(stderr, "last.samples=%d expected 4\n", last.samples);
        return 1;
    }
    if (!(last.loss_total < first.loss_total)) {
        std::fprintf(stderr,
            "loss did not decrease: first=%.6f last=%.6f\n",
            first.loss_total, last.loss_total);
        return 1;
    }

    std::printf("ok first.loss_total=%.6f last.loss_total=%.6f\n",
                first.loss_total, last.loss_total);
    std::printf("   first.loss_value=%.6f last.loss_value=%.6f\n",
                first.loss_value, last.loss_value);
    std::printf("   first.loss_policy=%.6f last.loss_policy=%.6f\n",
                first.loss_policy, last.loss_policy);
    return 0;
}
